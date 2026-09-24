"""Coordinate layer: give rhythm events x/y positions with the AUTO-OSU coordinate diffusion model.

The model (a DiT, see `autoosu.ml.coord`) was trained on ranked osu!standard maps to denoise the
positions of a token sequence: one token per circle, slider head, slider anchor, slider end and
spinner, each with its time and a one-hot type. At generation time the sequence is built from our
rhythm events, positions start as pure noise, and the model is conditioned on the star rating and
circle size. Sliders are then fitted into the playfield (`autoosu.sliderpath.fit_slider`).

Sequence format (must match the training data loader in coord/osu_diffusion/utils/data_loading.py):
    features = [x, y, time_ms, one-hot(16)]
    types: 0 circle, 1 circle+NC, 2 spinner, 3 spinner end, 4 slider head, 5 slider head+NC,
           6 bezier anchor, 7 perfect-circle anchor, 8 catmull anchor, 9 red anchor, 10 last anchor,
           11-15 slider end with repeat class (1, 2, 3, even>=4, odd>=5)
    context = [sinusoidal(time*0.1, 128), sinusoidal(distance to previous, 128), one-hot(16)]  (272)
Distances default to zero; optional two-pass control restores the trained pixel-distance condition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..beatmap import PLAYFIELD_H, PLAYFIELD_W, Circle, HitObject, Slider, Spinner
from ..difficulty import DifficultyPreset
from ..placement import SvSection, circle_radius
from ..rhythm import RhythmEvent
from ..sliderpath import fit_slider, path_points
from ..timing import Timing
from .coord import DiT_models, create_diffusion, timestep_embedding

N_TYPES = 16
CONTEXT_SIZE = 272
T_CIRCLE, T_SPINNER, T_SPINNER_END, T_HEAD, T_BEZIER, T_PERFECT, T_CATMULL, T_RED, T_LAST, T_END = 0, 2, 3, 4, 6, 7, 8, 9, 10, 11
PLAYFIELD = np.array([PLAYFIELD_W, PLAYFIELD_H], dtype=np.float32)
ProgressFn = Callable[[float, str], None]


def repeat_type(repeats: int) -> int:
    if repeats < 4:
        return repeats - 1
    return 3 if repeats % 2 == 0 else 4


# --------------------------------------------------------------------------- model

@dataclass
class CoordModel:
    net: torch.nn.Module
    device: str
    num_diff_classes: int = 26
    max_difficulty: float = 12.0
    num_cs_classes: int = 22
    arch: str = "DiT-B"

    @property
    def num_tokens(self) -> int:
        return self.num_diff_classes + self.num_cs_classes

    def class_vector(self, star: Optional[float], cs: Optional[float]) -> torch.Tensor:
        """One-hot difficulty class + one-hot circle-size class (unknown slots when None)."""
        v = torch.zeros(self.num_tokens)
        nd, ncs = self.num_diff_classes, self.num_cs_classes
        if star is None:
            v[nd - 1] = 1
        else:
            v[int(np.clip(int(star * (nd - 2) / self.max_difficulty), 0, nd - 2))] = 1
        if cs is None:
            v[nd + ncs - 1] = 1
        else:
            v[nd + int(np.clip(int(cs * (ncs - 2) / 10), 0, ncs - 2))] = 1
        return v


def export_coord_model(model_ema_pkl: str | Path, tokenizer_pkl: str | Path, out: str | Path,
                       arch: str = "DiT-B") -> Path:
    """Pack the training checkpoint (EMA weights + tokenizer state) into one safe fp16 .pt file."""
    ema = torch.load(model_ema_pkl, map_location="cpu", weights_only=False)
    tok = torch.load(tokenizer_pkl, map_location="cpu", weights_only=False)
    meta = {"arch": arch, "num_diff_classes": int(tok["num_diff_classes"]), "max_difficulty": float(tok["max_difficulty"]),
            "num_cs_classes": int(tok["num_cs_classes"]), "context_size": CONTEXT_SIZE, "format": "autoosu-coord-v1"}
    state = {k: v.to(torch.float16) for k, v in ema.items()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "state_dict": state}, out)
    return out


def load_coord_model(path: str | Path, device: Optional[str] = None) -> CoordModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    cm = CoordModel(net=None, device=device, num_diff_classes=meta["num_diff_classes"],
                    max_difficulty=meta["max_difficulty"], num_cs_classes=meta["num_cs_classes"], arch=meta["arch"])
    net = DiT_models[meta["arch"]](context_size=meta.get("context_size", CONTEXT_SIZE), class_size=cm.num_tokens)
    net.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    cm.net = net.to(device).eval()
    return cm


# --------------------------------------------------------------------------- sequence

@dataclass
class SeqSlider:
    event: RhythmEvent
    curve: str                     # L | B | P
    anchor_idx: List[int]          # sequence indices of head + anchors in path order
    end_idx: int                   # index of the slider-end token
    length: float                  # required pixel length of one slide


@dataclass
class Sequence:
    times: np.ndarray              # (N,) ms
    types: np.ndarray              # (N,) int
    object_idx: List[int]          # sequence index of every event's first token
    sliders: List[SeqSlider] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)


def _slider_structure(length: float, diam: float, rng: np.random.Generator) -> Tuple[str, int]:
    """Curve type and number of middle anchors for a slider of the given pixel length."""
    if length < 1.2 * diam:
        return ("L", 0) if rng.random() < 0.6 else ("B", 1)
    if length < 2.5 * diam:
        return [("B", 1), ("P", 1), ("L", 0)][int(rng.choice(3, p=[0.5, 0.35, 0.15]))]
    return [("B", 1), ("P", 1), ("B", 2)][int(rng.choice(3, p=[0.35, 0.3, 0.35]))]


def build_sequence(events: Sequence[RhythmEvent], preset: DifficultyPreset, timing: Timing,
                   sv_sections: Sequence[SvSection], rng: np.random.Generator) -> Sequence:
    diam = 2 * circle_radius(preset.cs)
    times: List[float] = []
    types: List[int] = []
    object_idx: List[int] = []
    sliders: List[SeqSlider] = []

    def sv_at(t: int) -> float:
        for start, end, sv in sv_sections:
            if start <= t < end:
                return sv
        return 1.0

    for ev in events:
        object_idx.append(len(times))
        if ev.kind == "spinner":
            times += [ev.time, ev.end_time]
            types += [T_SPINNER, T_SPINNER_END]
        elif ev.kind == "slider":
            span = (ev.end_time - ev.time) / max(1, ev.repeats)
            length = max(1.0, preset.slider_multiplier * 100.0 * sv_at(ev.time) * span / timing.beat_length)
            curve, n_mid = _slider_structure(length, diam, rng)
            head = len(times)
            times.append(ev.time)
            types.append(T_HEAD + (1 if ev.new_combo else 0))
            for k in range(n_mid):
                times.append(ev.time + (k + 1) / (n_mid + 1) * span)
                types.append(T_PERFECT if curve == "P" else T_BEZIER)
            times.append(ev.time + span)
            types.append(T_LAST)
            end = len(times)
            times.append(ev.end_time)
            types.append(T_END + repeat_type(max(1, ev.repeats)))
            sliders.append(SeqSlider(ev, curve, list(range(head, end)), end, length))
        else:
            times.append(ev.time)
            types.append(T_CIRCLE + (1 if ev.new_combo else 0))
    return Sequence(np.asarray(times, dtype=np.float32), np.asarray(types, dtype=np.int64), object_idx, sliders)


def sequence_context(seq: Sequence, distances: Optional[np.ndarray] = None) -> torch.Tensor:
    """(272, N): time embedding, raw pixel-distance embedding, one-hot type."""
    t = torch.from_numpy(seq.times)
    onehot = torch.zeros(N_TYPES, len(seq))
    onehot[torch.from_numpy(seq.types), torch.arange(len(seq))] = 1
    d = torch.zeros_like(t) if distances is None else torch.as_tensor(distances, dtype=torch.float32)
    if d.shape != t.shape or not torch.isfinite(d).all() or torch.any(d < 0):
        raise ValueError("Distances must be finite nonnegative pixels, one per token")
    return torch.cat([timestep_embedding(t * 0.1, 128).T, timestep_embedding(d, 128).T, onehot], 0)


def distance_condition(seq: Sequence, positions: np.ndarray, scale=1., regions=()) -> np.ndarray:
    from ..controls import envelope
    positions = np.asarray(positions, dtype=np.float32)
    if positions.shape != (len(seq), 2) or not np.isfinite(positions).all():
        raise ValueError("Reference positions must have shape (tokens, 2)")
    previous = np.vstack(([256., 192.], positions[:-1]))
    distances = np.linalg.norm(positions - previous, axis=1).astype(np.float32)
    # Only incoming circle/slider-head distances are scaled. Slider internals,
    # end tokens and spinner tokens retain their own reference distances.
    heads = np.isin(seq.types, [T_CIRCLE, T_CIRCLE+1, T_HEAD, T_HEAD+1])
    factors = scale * (1 + .2 * envelope(seq.times, regions))
    distances[heads] *= factors[heads]
    return distances


# --------------------------------------------------------------------------- sampling

def _to_pixels(x: torch.Tensor) -> np.ndarray:
    """(2, N) in [-1, 1] -> (N, 2) pixels."""
    return ((x.T.float().cpu().numpy() + 1) / 2) * PLAYFIELD


def _from_pixels(p: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy((p / PLAYFIELD * 2 - 1).T).to(like.device, like.dtype)


@torch.no_grad()
def sample_positions(cm: CoordModel, seq: Sequence, star: Optional[float], cs: Optional[float], *,
                     steps: int = 100, seed: int = 0, cfg_scale: float = 1.0, max_seq_len: int = 1024,
                     overlap: int = 128, band: int = 128, progress: Optional[ProgressFn] = None,
                     distances: Optional[np.ndarray] = None) -> np.ndarray:
    """Denoise positions for the whole sequence; returns (N, 2) pixel coordinates."""
    n = len(seq)
    device = cm.device
    diffusion = create_diffusion(timestep_respacing=[int(steps)], diffusion_steps=1000, noise_schedule="squaredcos_cap_v2")
    c_all = sequence_context(seq, distances).to(device)
    y = cm.class_vector(star, cs).to(device).unsqueeze(0)
    y_null = cm.class_vector(None, None).to(device).unsqueeze(0)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise_all = torch.randn(1, 2, n, generator=gen).to(device)
    out = noise_all.clone()

    # windows of max_seq_len with `overlap` tokens of context from the previous window
    step = max(1, max_seq_len - 2 * overlap)
    starts = [0]
    while starts[-1] + max_seq_len < n:
        starts.append(starts[-1] + step)
    total_steps = steps * len(starts)
    done = 0

    def model_fn(x, t, **kw):
        if cfg_scale == 1.0:
            return cm.net(x, t, **kw)
        return cm.net.forward_with_cfg(x, t, cfg_scale=cfg_scale, **kw)

    for wi, s in enumerate(starts):
        e = min(s + max_seq_len, n)
        L = e - s
        keep = overlap if s > 0 else 0          # tokens fixed from the previous window
        mask = torch.zeros(1, 2, L, dtype=torch.bool, device=device)
        mask[:, :, keep:] = True
        z = out[:, :, s:e].clone()
        z[mask] = noise_all[:, :, s:e][mask]     # (re)start the generated part from noise
        c = c_all[:, s:e].unsqueeze(0)
        attn = torch.ones(L, L, dtype=torch.bool, device=device)
        for i in range(L):
            attn[max(0, i - band):min(L, i + band), i] = False
        window_sliders = [sl for sl in seq.sliders if sl.anchor_idx[0] >= s and sl.end_idx < e]

        def denoised_fn(x, _z=z, _mask=mask, _sl=window_sliders, _s=s):
            x = torch.where(_mask, x, _z)
            if _sl:
                p = _to_pixels(x[0])
                for sl in _sl:
                    idx = [i - _s for i in sl.anchor_idx]
                    pts = path_points(sl.curve, p[idx])
                    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                    cum = np.concatenate([[0.0], np.cumsum(seg)])
                    d = min(sl.length, cum[-1])
                    i = max(0, min(int(np.searchsorted(cum, d, side="right")) - 1, len(seg) - 1))
                    w = (d - cum[i]) / seg[i] if seg[i] > 0 else 0.0
                    p[sl.end_idx - _s] = pts[i] + (pts[i + 1] - pts[i]) * w
                x = _from_pixels(p, x).unsqueeze(0)
            return x

        if cfg_scale == 1.0:
            kwargs = dict(c=c, y=y, attn_mask=attn)
            z_in = denoised_fn(z)
        else:
            kwargs = dict(c=torch.cat([c, c]), y=torch.cat([y, y_null]), attn_mask=attn)
            z_in = torch.cat([denoised_fn(z)] * 2)
            mask = torch.cat([mask, mask])
            z = torch.cat([z, z])
        torch.manual_seed(seed + wi)
        use_amp = device.startswith("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            sample = None
            for k, res in enumerate(diffusion.p_sample_loop_progressive(
                    model_fn, z_in.shape, noise=z_in, clip_denoised=True, denoised_fn=denoised_fn,
                    model_kwargs=kwargs, device=device, progress=False)):
                sample = res["sample"]
                done += 1
                if progress and (k % 5 == 0 or k == steps - 1):
                    progress(done / total_steps, f"window {wi + 1}/{len(starts)} step {k + 1}/{steps}")
        out[:, :, s:e] = sample[:1].float()
    return np.clip(_to_pixels(out[0]), 0, PLAYFIELD)


# --------------------------------------------------------------------------- objects

@dataclass
class Placement:
    objects: List[HitObject]
    sv_overrides: List[Tuple[int, int, float]] = field(default_factory=list)   # start, end, sv multiplier


def place_with_model(events: List[RhythmEvent], preset: DifficultyPreset, timing: Timing, cm: CoordModel,
                     sv_sections: Sequence[SvSection] = (), *, star: Optional[float] = None, seed: int = 0,
                     steps: int = 100, cfg_scale: float = 1.0, progress: Optional[ProgressFn] = None,
                     spacing_scale: Optional[float] = None, highlight_regions=()) -> Placement:
    rng = np.random.default_rng(seed)
    seq = build_sequence(events, preset, timing, sv_sections, rng)
    if len(seq) == 0:
        return Placement([])
    controlled = spacing_scale is not None or bool(highlight_regions)
    first_progress = (lambda f, m: progress(f*.5, "reference: " + m)) if controlled and progress else progress
    pos = sample_positions(cm, seq, star, preset.cs, steps=steps, seed=seed, cfg_scale=cfg_scale, progress=first_progress)
    if controlled:
        distances = distance_condition(seq, pos, 1. if spacing_scale is None else spacing_scale, highlight_regions)
        second_progress = (lambda f, m: progress(.5+f*.5, "controlled: " + m)) if progress else None
        pos = sample_positions(cm, seq, star, preset.cs, steps=steps, seed=seed, cfg_scale=cfg_scale,
                               progress=second_progress, distances=distances)
    by_head = {sl.anchor_idx[0]: sl for sl in seq.sliders}
    objects: List[HitObject] = []
    overrides: List[Tuple[int, int, float]] = []
    for ev, i in zip(events, seq.object_idx):
        x, y = (int(round(v)) for v in pos[i])
        if ev.kind == "spinner":
            objects.append(Spinner(PLAYFIELD_W // 2, PLAYFIELD_H // 2, ev.time, True, ev.hitsound, end=ev.end_time))
        elif ev.kind == "slider":
            sl = by_head[i]
            fit = fit_slider(sl.curve, pos[sl.anchor_idx], sl.length)
            hx, hy = fit.anchors[0]
            end = (hx, hy) if ev.repeats % 2 == 0 else (int(round(fit.end[0])), int(round(fit.end[1])))
            objects.append(Slider(hx, hy, ev.time, ev.new_combo, ev.hitsound, curve_type=sl.curve,
                                  points=list(fit.anchors[1:]), repeats=max(1, ev.repeats), length=fit.length,
                                  duration=ev.end_time - ev.time, end_x=end[0], end_y=end[1]))
            if fit.sv_scale < 0.999:
                overrides.append((ev.time, ev.end_time, fit.sv_scale))
        else:
            objects.append(Circle(x, y, ev.time, ev.new_combo, ev.hitsound))
    return Placement(objects, overrides)
