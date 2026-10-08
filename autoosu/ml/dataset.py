"""Tick dataset: a beatmap becomes a sequence of 1/4-beat grid ticks. Each tick has a class label,
an audio patch around it, its position in the measure and a few scalar features (section loudness,
beat length, local object density). The same tensors feed both the autoregressive and the masked
model; the training loop derives the model inputs (shifted labels or masked labels).

Classes: 0 none, 1 circle, 2 slider head, 3 slider body, 4 slider end, 5 spinner (body).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:          # tick construction / statistics work without torch
    torch = None
    Dataset = object

from .osu_parse import OBJ_CIRCLE, OBJ_SLIDER, OBJ_SPINNER
from .prepare_data import HOP, SR
from .skills import conditioned_vector

GRID = 4                          # ticks per beat
N_CLASSES = 6
BOS = N_CLASSES                   # autoregressive: token before the first tick
MASK = N_CLASSES + 1              # masked model: hidden label
N_INPUT_TOKENS = N_CLASSES + 2
PATCH = 16                        # audio frames per tick patch (8 before, 8 after; 10 ms each)
FRAME_MS = HOP * 1000.0 / SR      # 10 ms
MAX_METER_SLOTS = 64              # meter (<=16) * GRID
COND_DIM = 6                      # sr, cs, ar, od, hp, year
EXTRA_DIM = 4                     # loudness, log beat length, density, density-known flag
DENSITY_SCALE = 4.0               # objects per measure that maps to feature value 1.0
DENSITY_WINDOW_MEASURES = 8       # +- window for the local density estimate
START_CLASSES = (1, 2)            # tick classes that start an object (spinner starts counted separately)


@dataclass
class TickSeq:
    times: np.ndarray       # (M,) ms
    metrical: np.ndarray    # (M,) int, tick index inside its measure
    labels: np.ndarray      # (M,) int
    beat_ms: np.ndarray     # (M,) float, beat length in force


def build_ticks(red_lines: np.ndarray, objects: Optional[np.ndarray], end_ms: float) -> TickSeq:
    """Lay a 1/GRID grid over every red-line segment and assign object classes to ticks."""
    red = red_lines[np.argsort(red_lines[:, 0])]
    times, met, bls = [], [], []
    for i, (t0, bl, meter) in enumerate(red):
        t1 = red[i + 1][0] if i + 1 < len(red) else end_ms
        if t1 <= t0 or bl <= 0:
            continue
        step = bl / GRID
        slots = int(max(1, min(16, meter))) * GRID
        n = int(math.floor((t1 - t0) / step + 1e-6))
        k = np.arange(n)
        times.append(t0 + k * step)
        met.append(k % slots)
        bls.append(np.full(n, bl))
    if not times:
        z = np.zeros(0)
        return TickSeq(z, z.astype(int), z.astype(int), z)
    times = np.concatenate(times)
    met = np.concatenate(met).astype(np.int64)
    beat_ms = np.concatenate(bls)
    labels = np.zeros(len(times), dtype=np.int64)
    if objects is None or len(objects) == 0:
        return TickSeq(times, met, labels, beat_ms)

    spacing = np.diff(times, append=times[-1] + (times[-1] - times[-2] if len(times) > 1 else 100))
    for t, typ, end, _nc in objects:
        i = int(np.searchsorted(times, t))
        cands = [j for j in (i - 1, i) if 0 <= j < len(times)]
        if not cands:
            continue
        j = min(cands, key=lambda j: abs(times[j] - t))
        if abs(times[j] - t) > 0.35 * spacing[j]:
            continue                                   # off-grid (1/3, 1/6 ...): skipped in v0
        typ = int(typ)
        if typ == OBJ_CIRCLE:
            if labels[j] == 0:
                labels[j] = 1
        elif typ == OBJ_SLIDER:
            e = int(np.searchsorted(times, end))
            ecands = [k for k in (e - 1, e) if 0 <= k < len(times)]
            k = min(ecands, key=lambda k: abs(times[k] - end)) if ecands else j + 1
            k = max(k, j + 1)
            if k >= len(times):
                continue
            labels[j] = 2
            body = labels[j + 1:k]
            body[body == 0] = 3
            if labels[k] == 0:
                labels[k] = 4
        elif typ == OBJ_SPINNER:
            e = int(np.searchsorted(times, end))
            k = min(max(j + 1, e), len(times))
            seg = labels[j:k]
            seg[seg == 0] = 5
    return TickSeq(times, met, labels, beat_ms)


def object_starts(labels: np.ndarray) -> np.ndarray:
    """1 where an object starts (circle, slider head, first tick of a spinner)."""
    starts = np.isin(labels, START_CLASSES).astype(np.float64)
    spin = (labels == 5).astype(np.int8)
    starts += (np.diff(np.concatenate([[0], spin])) == 1)
    return starts


def local_density(labels: np.ndarray, metrical: np.ndarray, window_measures: int = DENSITY_WINDOW_MEASURES) -> np.ndarray:
    """Objects per measure in a +- window around each tick (uses the local measure length in ticks)."""
    n = len(labels)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    slots = int(metrical.max()) + 1 if n else 4 * GRID
    slots = max(slots, GRID)
    radius = window_measures * slots
    c = np.concatenate([[0.0], np.cumsum(object_starts(labels))])
    idx = np.arange(n)
    lo = np.clip(idx - radius, 0, n)
    hi = np.clip(idx + radius, 0, n)
    counts = c[hi] - c[lo]
    measures = np.maximum(hi - lo, 1) / slots
    return (counts / measures).astype(np.float32)


def cond_vector(row: dict) -> np.ndarray:
    """Normalised global conditioning vector from an index row (or a dict with the same keys)."""
    return np.array([
        (float(row.get("sr", 4.0)) - 4.0) / 3.0,
        (float(row.get("cs", 4.0)) - 4.0) / 2.0,
        (float(row.get("ar", 8.0)) - 7.0) / 3.0,
        (float(row.get("od", 6.0)) - 6.0) / 3.0,
        (float(row.get("hp", 5.0)) - 5.0) / 2.0,
        (float(row.get("year", 2018)) - 2018.0) / 6.0,
    ], dtype=np.float32)


def local_loudness(mel_u8: np.ndarray, frames: np.ndarray, radius_frames: int = 200) -> np.ndarray:
    """Mean log-mel energy in a +-2 s window around each tick, z-scored over the track."""
    e = mel_u8.mean(axis=1).astype(np.float32)
    c = np.concatenate([[0.0], np.cumsum(e)])
    lo = np.clip(frames - radius_frames, 0, len(e))
    hi = np.clip(frames + radius_frames, 0, len(e))
    m = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return ((m - e.mean()) / (e.std() + 1e-6)).astype(np.float32)


def gather_patches(mel_u8: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """(M, PATCH, 64) uint8 audio patches centred on each tick frame (zero padded at the edges)."""
    T = len(mel_u8)
    idx = frames[:, None] + np.arange(-PATCH // 2, PATCH // 2)[None, :]
    valid = (idx >= 0) & (idx < T)
    out = np.asarray(mel_u8)[np.clip(idx, 0, T - 1)]
    out[~valid] = 0
    return out


def extra_features(loud: np.ndarray, beat_ms: np.ndarray, density: Optional[np.ndarray]) -> np.ndarray:
    """(M, EXTRA_DIM) float32: loudness z, log2(beat/500ms), density/4, density-known flag."""
    n = len(loud)
    out = np.zeros((n, EXTRA_DIM), dtype=np.float32)
    out[:, 0] = loud
    out[:, 1] = np.log2(np.maximum(beat_ms, 50.0) / 500.0)
    if density is not None:
        out[:, 2] = density / DENSITY_SCALE
        out[:, 3] = 1.0
    return out


class TickDataset(Dataset):
    def __init__(self, rows: List[dict], prep_dir: str | Path, seq_len: int = 512, train: bool = True,
                 density_dropout: float = 0.2, skill_names=(), skill_dropout: float = 0.2,
                 evaluation_crop: str = "start", density_mode: str = "reference", attributes_dir=None):
        self.rows, self.prep, self.seq_len, self.train = rows, Path(prep_dir), seq_len, train
        self.density_dropout = density_dropout
        self.skill_names, self.skill_dropout = tuple(skill_names), skill_dropout
        if evaluation_crop not in ("start", "center") or density_mode not in ("reference", "unknown"):
            raise ValueError("Invalid crop or density condition mode")
        self.evaluation_crop, self.density_mode = evaluation_crop, density_mode
        self.attributes_dir = Path(attributes_dir) if attributes_dir else None
        self._mel_cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _mel(self, track: str) -> np.ndarray:
        m = self._mel_cache.get(track)
        if m is None:
            m = np.load(self.prep / "tracks" / f"{track}.npy", mmap_mode="r")
            if len(self._mel_cache) > 64:
                self._mel_cache.clear()
            self._mel_cache[track] = m
        return m

    def __getitem__(self, i: int) -> Dict[str, "torch.Tensor"]:
        row = self.rows[i]
        mel = self._mel(row["track"])
        z = np.load(self.prep / "maps" / f"{row['beatmap_id']}.npz")
        ticks = build_ticks(z["red_lines"], z["objects"], end_ms=len(mel) * FRAME_MS)
        M, L = len(ticks.times), self.seq_len
        if M == 0:
            return self.__getitem__((i + 1) % len(self.rows))
        if self.train:
            first = int(np.searchsorted(ticks.times, z["objects"][0, 0]))
            last = int(np.searchsorted(ticks.times, z["objects"][-1, 0]))
            lo, hi = max(0, first - L // 2), max(0, min(M - 1, last) - L // 4)
            start = int(np.random.randint(lo, max(lo + 1, hi + 1)))
        else:
            if self.evaluation_crop == "center":
                first = int(np.searchsorted(ticks.times, z["objects"][0, 0]))
                last = int(np.searchsorted(ticks.times, z["objects"][-1, 0]))
                start = max(0, min(M - L, (first + last - L) // 2))
            else:
                start = 0
        density_all = local_density(ticks.labels, ticks.metrical)
        sl = slice(start, start + L)
        times, met, labels, beat_ms = ticks.times[sl], ticks.metrical[sl], ticks.labels[sl], ticks.beat_ms[sl]
        density = density_all[sl]
        if self.density_mode == "unknown" or (self.train and np.random.random() < self.density_dropout):
            density = None
        n = len(times)
        frames = np.round(times / FRAME_MS).astype(np.int64)
        audio = gather_patches(mel, frames)
        extra = extra_features(local_loudness(np.asarray(mel), frames), beat_ms, density)
        prev_label = BOS if start == 0 else int(ticks.labels[start - 1])
        skills = row.get("skill_labels", {}) if self.skill_names else {}
        if self.train and self.skill_names and np.random.random() < self.skill_dropout:
            skills = {}

        def pad(a: np.ndarray, value=0):
            if n >= L:
                return a[:L]
            padw = [(0, L - n)] + [(0, 0)] * (a.ndim - 1)
            return np.pad(a, padw, constant_values=value)

        result = {
            "audio": torch.from_numpy(pad(audio).astype(np.float32) / 255.0),
            "metrical": torch.from_numpy(pad(met)).long(),
            "extra": torch.from_numpy(pad(extra)).float(),
            "labels": torch.from_numpy(pad(labels, -100)).long(),      # -100 = padding, ignored
            "cond": torch.from_numpy(conditioned_vector(cond_vector(row), self.skill_names, skills)),
            "prev_label": torch.tensor(prev_label, dtype=torch.long),
        }
        if self.attributes_dir is not None:
            from .attributes import tick_attributes
            attributes = np.load(self.attributes_dir / "maps" / f"{row['beatmap_id']}.npz")["attributes"]
            if len(attributes) != len(z["objects"]):
                raise ValueError("Attribute/object alignment changed")
            result["attributes"] = torch.from_numpy(pad(tick_attributes(ticks, z["objects"], attributes)[sl], -100)).long()
        return result


def load_index(prep_dir: str | Path, val_frac: float = 0.03, min_objects: int = 50) -> Tuple[List[dict], List[dict]]:
    """Split by beatmapset so all difficulties of one set land on the same side."""
    with (Path(prep_dir) / "index.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    rows = [r for r in rows if r["n_objects"] >= min_objects]
    train, val = [], []
    for r in rows:
        h = (r["beatmapset_id"] * 2654435761) % 1000 / 1000.0
        (val if h < val_frac else train).append(r)
    return train, val


def read_splits(root):
    """Read explicit frozen manifests and reject song leakage before training."""
    splits = {name: [json.loads(line) for line in (Path(root) / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
              for name in ("train", "validation", "test")}
    groups = {name: {r["song_group"] for r in rows} for name, rows in splits.items()}
    for name, other in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[name] & groups[other]:
            raise ValueError(f"Song group overlap: {name}/{other}")
    for name, rows in splits.items():
        if not rows or any(r["split"] != name for r in rows):
            raise ValueError(f"Empty or incorrectly assigned {name} split")
    return splits
