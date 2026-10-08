"""Tick transformer with two decoding paradigms sharing one trunk:

* mode "ar":     causal over labels; input token at tick i is the label of tick i-1 (BOS first);
                 sampled tick by tick.
* mode "masked": bidirectional; input tokens are the labels with a random subset replaced by MASK;
                 decoded in a few parallel MaskGIT-style rounds.

Optional *audio context* layers (`audio_ctx_layers > 0`) run bidirectionally over the label-free
per-tick features first, so the causal decoder still gets to "hear" the audio that comes after the
current tick, the way a mapper listens to the whole phrase. Audio is known in advance, so this is
legal for autoregressive sampling.

Both predict the class of every tick from the audio patch, the metrical position, the scalar
features (loudness, beat length, local density) and the global conditioning.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import BOS, COND_DIM, EXTRA_DIM, MASK, MAX_METER_SLOTS, N_CLASSES, N_INPUT_TOKENS, PATCH
from .prepare_data import N_MELS


@dataclass
class ModelConfig:
    d_model: int = 384
    n_layers: int = 6            # label-aware layers (causal in "ar", bidirectional in "masked")
    n_heads: int = 6
    dropout: float = 0.1
    max_len: int = 2048
    mode: str = "ar"             # "ar" | "masked"
    audio_ctx_layers: int = 0    # bidirectional label-free layers over the audio features first
    audio_frontend: str = "linear"  # linear v0 patch projection or learned convolutional spectrum encoder
    skill_names: tuple[str, ...] = ()  # positive-only skill conditions appended to global features
    attribute_vocab: dict = field(default_factory=dict)  # source-supervised object attributes


class SpectralEncoder(nn.Module):
    """Learn local time/frequency patterns before the shared beat-context encoder."""
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(32 * (PATCH // 4) * (N_MELS // 4), width), nn.GELU())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, length = audio.shape[:2]
        return self.net(audio.reshape(batch * length, 1, PATCH, N_MELS)).reshape(batch, length, -1)


class TickTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        if cfg.audio_frontend == "linear":
            self.audio_proj = nn.Sequential(nn.Flatten(-2), nn.Linear(PATCH * N_MELS, d), nn.GELU(), nn.Linear(d, d))
        elif cfg.audio_frontend == "conv":
            self.audio_proj = SpectralEncoder(d)
        else:
            raise ValueError(f"Unknown audio frontend: {cfg.audio_frontend}")
        self.token_emb = nn.Embedding(N_INPUT_TOKENS, d)
        self.met_emb = nn.Embedding(MAX_METER_SLOTS, d)
        self.extra_proj = nn.Sequential(nn.Linear(EXTRA_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.cond_proj = nn.Sequential(nn.Linear(COND_DIM + len(cfg.skill_names), d), nn.GELU(), nn.Linear(d, d))
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        self.dec_pos_emb = nn.Embedding(cfg.max_len, d)

        def layer():
            return nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                              batch_first=True, norm_first=True)

        self.audio_ctx = (nn.TransformerEncoder(layer(), cfg.audio_ctx_layers, enable_nested_tensor=False)
                          if cfg.audio_ctx_layers > 0 else None)
        self.blocks = nn.TransformerEncoder(layer(), cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_CLASSES)
        self.attribute_heads = nn.ModuleDict({name: nn.Linear(d, len(values))
                                             for name, values in cfg.attribute_vocab.items()})
        self.drop = nn.Dropout(cfg.dropout)

    @property
    def causal(self) -> bool:
        return self.cfg.mode == "ar"

    def encode_audio(self, audio: torch.Tensor, metrical: torch.Tensor, extra: torch.Tensor, cond: torch.Tensor,
                     pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Label-free per-tick representation (B,L,d); bidirectional over the window if ctx layers exist."""
        L = metrical.shape[1]
        h = (self.audio_proj(audio) + self.met_emb(metrical) + self.extra_proj(extra)
             + self.cond_proj(cond).unsqueeze(1) + self.pos_emb(torch.arange(L, device=metrical.device)).unsqueeze(0))
        if self.audio_ctx is not None:
            h = self.audio_ctx(self.drop(h), src_key_padding_mask=pad_mask)
        return h

    def decode(self, h: torch.Tensor, tokens: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h (B,L,d) from encode_audio + label tokens (B,L) -> logits (B,L,C)."""
        return self.head(self.decode_features(h, tokens, pad_mask))

    def decode_features(self, h: torch.Tensor, tokens: torch.Tensor,
                        pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        L = tokens.shape[1]
        x = h + self.token_emb(tokens) + self.dec_pos_emb(torch.arange(L, device=tokens.device)).unsqueeze(0)
        x = self.drop(x)
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=tokens.device)
            x = self.blocks(x, mask=mask, src_key_padding_mask=pad_mask, is_causal=True)
        else:
            x = self.blocks(x, src_key_padding_mask=pad_mask)
        return self.norm(x)

    def forward(self, audio: torch.Tensor, tokens: torch.Tensor, metrical: torch.Tensor, extra: torch.Tensor,
                cond: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                attribute_targets: Optional[torch.Tensor] = None):
        """audio (B,L,PATCH,64) tokens (B,L) metrical (B,L) extra (B,L,EXTRA_DIM) cond (B,COND_DIM) -> (B,L,C)"""
        features = self.decode_features(self.encode_audio(audio, metrical, extra, cond, pad_mask), tokens, pad_mask)
        logits = self.head(features)
        if attribute_targets is None:
            return logits
        from .attributes import ATTRIBUTE_NAMES
        # Only applicable source objects allocate attribute logits. Empty heads remain in autograd.
        predictions = {name: head(features[attribute_targets[..., ATTRIBUTE_NAMES.index(name)] != -100])
                       for name, head in self.attribute_heads.items()}
        return logits, predictions

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **(extra or {})}, path)

    def with_skills(self, names) -> "TickTransformer":
        """Initialize new condition columns at zero, preserving the pretrained function."""
        if self.cfg.skill_names:
            raise ValueError("Model already has a skill schema")
        model = TickTransformer(replace(self.cfg, skill_names=tuple(names)))
        state = dict(self.state_dict())
        weight = state["cond_proj.0.weight"]
        state["cond_proj.0.weight"] = F.pad(weight, (0, len(names)))
        model.load_state_dict(state)
        return model

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TickTransformer":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items() if k in ModelConfig.__dataclass_fields__})
        model = cls(cfg)
        state = dict(ckpt["state_dict"])
        if "dec_pos_emb.weight" not in state:
            # v0 checkpoints added one positional embedding before the label-aware blocks; with a zero
            # decoder positional table the new two-stage forward is numerically identical.
            state["dec_pos_emb.weight"] = torch.zeros_like(model.dec_pos_emb.weight)
        model.load_state_dict(state)
        return model.to(device).eval()


# --------------------------------------------------------------------------- training inputs

def ar_inputs(labels: torch.Tensor, prev_label: torch.Tensor) -> torch.Tensor:
    """Shift labels right by one; padding (-100) becomes 0."""
    prev = torch.cat([prev_label[:, None], labels[:, :-1]], dim=1)
    return prev.clamp(min=0)


def masked_inputs(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask, torch.full_like(labels, MASK), labels.clamp(min=0))


def random_mask(labels: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """MaskGIT-style mask: ratio r = cos(pi/2 * u), u ~ U(0,1), applied to the valid positions."""
    B, L = labels.shape
    valid = labels != -100
    u = torch.rand(B, 1, device=labels.device, generator=generator)
    ratio = torch.cos(math.pi / 2 * u).clamp(0.02, 1.0)
    score = torch.rand(B, L, device=labels.device, generator=generator)
    score[~valid] = 2.0                                   # never mask padding
    n_valid = valid.sum(1, keepdim=True)
    n_mask = (ratio * n_valid).ceil().long().clamp(min=1)
    ranks = score.argsort(1).argsort(1)
    return (ranks < n_mask) & valid


# --------------------------------------------------------------------------- sampling

# rows = previous class, cols = allowed next class     none circ head body end spin
_ALLOWED = torch.tensor([
    [1, 1, 1, 0, 0, 1],   # after none
    [1, 1, 1, 0, 0, 1],   # after circle
    [0, 0, 0, 1, 1, 0],   # after slider head: body or end only
    [0, 0, 0, 1, 1, 0],   # after body: body or end
    [1, 1, 1, 0, 0, 1],   # after slider end
    [1, 1, 1, 0, 0, 1],   # after spinner tick
], dtype=torch.bool)


def constrained_sample(logits: torch.Tensor, prev: int, temperature: float = 1.0, none_bias: float = 0.0,
                       generator: Optional[torch.Generator] = None) -> int:
    l = logits.float().clone()
    l[0] += none_bias
    allowed = _ALLOWED[prev if 0 <= prev < N_CLASSES else 0].to(l.device)
    l[~allowed] = -1e9
    if temperature <= 1e-6:
        return int(l.argmax())
    return int(torch.multinomial(F.softmax(l / temperature, dim=-1), 1, generator=generator))


def repair_structure(labels: np.ndarray) -> np.ndarray:
    """Make a label sequence structurally valid: bodies/ends need a head, heads need an end."""
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        c = out[i]
        if c in (3, 4) and (i == 0 or out[i - 1] not in (2, 3)):
            out[i] = 0                                    # stray body / end
        elif c == 2:
            j = i + 1
            while j < n and out[j] == 3:
                j += 1
            if j >= n:
                out[i] = 1                                # head at the very end -> circle
                out[i + 1:j] = 0                          # and drop its dangling body
            elif out[j] != 4:
                if j == i + 1:
                    out[i] = 1                            # nothing follows -> circle
                else:
                    out[j - 1] = 4                        # close the slider on its last body tick
            i = j
            continue
        i += 1
    return out


@torch.no_grad()
def encode_audio_long(model: TickTransformer, audio: torch.Tensor, metrical: torch.Tensor, extra: torch.Tensor,
                      cond: torch.Tensor, chunk: int = 2048, overlap: int = 256) -> torch.Tensor:
    """encode_audio over an arbitrarily long sequence: overlapping chunks, centre parts kept."""
    n = len(metrical)
    dev = metrical.device
    use_amp = dev.type == "cuda"
    out = torch.zeros(n, model.cfg.d_model, device=dev)
    start = 0
    while start < n:
        end = min(n, start + chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            h = model.encode_audio(audio[start:end].unsqueeze(0), metrical[start:end].unsqueeze(0),
                                   extra[start:end].unsqueeze(0), cond.unsqueeze(0))[0].float()
        keep_from = 0 if start == 0 else overlap // 2
        out[start + keep_from:end] = h[keep_from:]
        if end >= n:
            break
        start = end - overlap
    return out


@torch.no_grad()
def sample_ar(model: TickTransformer, audio: torch.Tensor, metrical: torch.Tensor, extra: torch.Tensor,
              cond: torch.Tensor, *, temperature: float = 0.9, none_bias: float = 0.0, context: int = 512,
              generator: Optional[torch.Generator] = None, prev0: int = BOS) -> np.ndarray:
    """audio (L,PATCH,64) metrical (L,) extra (L,EXTRA_DIM) cond (COND_DIM,) on the model device."""
    n = len(metrical)
    dev = metrical.device
    use_amp = dev.type == "cuda"
    h_all = encode_audio_long(model, audio, metrical, extra, cond)
    tokens = np.full(n + 1, prev0, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        lo = max(0, i - context + 1)
        tok = torch.from_numpy(tokens[lo:i + 1]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model.decode(h_all[lo:i + 1].unsqueeze(0), tok.unsqueeze(0))[0, -1]
        labels[i] = constrained_sample(logits, int(tokens[i]), temperature, none_bias, generator)
        tokens[i + 1] = labels[i]
    return repair_structure(labels)      # only the sequence end can be invalid (open slider)


@torch.no_grad()
def sample_masked(model: TickTransformer, audio: torch.Tensor, metrical: torch.Tensor, extra: torch.Tensor,
                  cond: torch.Tensor, *, steps: int = 12, temperature: float = 0.9, none_bias: float = 0.0,
                  generator: Optional[torch.Generator] = None, chunk: int = 1024, overlap: int = 256) -> np.ndarray:
    """Iterative parallel decoding: start fully masked, each round commit the most confident predictions.
    Long songs are decoded in chunks; the first `overlap` ticks of a chunk are fixed from the previous one."""
    n = len(metrical)
    dev = metrical.device
    use_amp = dev.type == "cuda"
    final = np.zeros(n, dtype=np.int64)
    start = 0
    while start < n:
        end = min(n, start + chunk)
        L = end - start
        known = torch.zeros(L, dtype=torch.bool, device=dev)
        tokens = torch.full((L,), MASK, dtype=torch.long, device=dev)
        if start > 0:
            k = min(overlap, L)
            tokens[:k] = torch.from_numpy(final[start:start + k]).to(dev)
            known[:k] = True
        n_free = int((~known).sum())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            h = model.encode_audio(audio[start:end].unsqueeze(0), metrical[start:end].unsqueeze(0),
                                   extra[start:end].unsqueeze(0), cond.unsqueeze(0))
        for s in range(steps):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.decode(h, tokens.unsqueeze(0))[0].float()
            logits[:, 0] += none_bias
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            conf = probs.gather(1, sampled[:, None]).squeeze(1)
            conf[known] = float("inf")
            ratio = math.cos(math.pi / 2 * (s + 1) / steps)          # fraction still masked after this round
            n_mask = int(math.floor(ratio * n_free))
            noise = -torch.log(-torch.log(torch.rand(L, device=dev, generator=generator).clamp_min(1e-9)))
            score = conf.log().clamp_min(-30) + noise * 0.5 * (1 - (s + 1) / steps)
            score[known] = float("inf")
            order = score.argsort(descending=True)
            commit = torch.zeros(L, dtype=torch.bool, device=dev)
            commit[order[: L - n_mask]] = True            # known positions have infinite score: always kept
            new_tokens = torch.where(commit, sampled, torch.full_like(tokens, MASK))
            new_tokens[known] = tokens[known]             # previously committed values stay as they are
            tokens, known = new_tokens, commit
        final[start:end] = tokens.cpu().numpy()
        start = end - overlap if end < n else n
    return repair_structure(final)


@torch.no_grad()
def sample_attributes(model, audio, metrical, extra, cond, labels, *, generator=None, temperature=.9, chunk=1024):
    """Predict object attributes after rhythm decoding; no attribute is supplied as input."""
    result = {name: np.full(len(labels), -100, dtype=np.int64) for name in model.attribute_heads}
    for start in range(0, len(labels), chunk):
        end = min(start + chunk, len(labels))
        tokens = torch.as_tensor(labels[start:end], dtype=torch.long, device=audio.device)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=audio.device.type == "cuda"):
            h = model.encode_audio(audio[start:end][None], metrical[start:end][None], extra[start:end][None], cond[None])
            features = model.decode_features(h, tokens)[0]
            for name, head in model.attribute_heads.items():
                valid = np.isin(labels[start:end], [1, 2]) if name in ("new_combo", "hitsound") else labels[start:end] == 2
                indices = np.flatnonzero(valid)
                if not len(indices):
                    continue
                logits = head(features[torch.as_tensor(indices, device=audio.device)]).float()
                if temperature <= 0:
                    selected = logits.argmax(-1)
                else:
                    selected = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)[:, 0]
                result[name][start + indices] = selected.cpu().numpy()
    return result
