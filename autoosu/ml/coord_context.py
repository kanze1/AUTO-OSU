"""Experimental audio context for coord v0; not a released inference checkpoint format.

Local acoustic patches and whole-song beat tokens are available at both training and
inference. No clean coordinates, reference distances or mapper identities enter this
branch. A zero-initialized residual preserves the pretrained spatial model initially.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .coord.models import timestep_embedding
from .dataset import FRAME_MS, gather_patches
from .model import SpectralEncoder
from .coord_objects import ObjectContextEncoder
from .coord_plan import SpacingPlanner


@dataclass(frozen=True)
class MusicContextConfig:
    width: int = 128
    heads: int = 4
    song_layers: int = 2
    whole_song: bool = True
    object_context: bool = False
    spacing_plan: bool = False

    def __post_init__(self):
        if self.heads < 1 or self.width < 4 or self.width % self.heads or self.width % 2 or self.song_layers < 1:
            raise ValueError("Music width must be positive/even and divisible by heads; layers must be positive")


def music_features(mel: np.ndarray, times_ms: np.ndarray, beat_ms: float, offset_ms: float):
    """Build unbatched features on one shared audio timeline, including beat-zero offset.

    Whole-song memory is pooled on beat intervals, not hard-coded phrase boundaries.
    Sequence crops select local queries; they retain the same whole-song memory.
    """
    times = np.asarray(times_ms, dtype=np.float64)
    if mel.ndim != 2 or mel.shape[1] != 64 or len(mel) == 0 or mel.dtype != np.uint8:
        raise ValueError("Expected nonempty uint8 mel with 64 channels")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Expected finite nonempty token times")
    if not np.isfinite([beat_ms, offset_ms]).all() or beat_ms <= 0:
        raise ValueError("Expected positive finite beat length and finite offset")
    frames = np.rint(times / FRAME_MS).astype(np.int64)
    local = gather_patches(mel, frames).astype(np.float32) / 255.
    frame_beats = (np.arange(len(mel)) * FRAME_MS - offset_ms) / beat_ms
    beat_ids = np.floor(frame_beats).astype(np.int64)
    unique, inverse = np.unique(beat_ids, return_inverse=True)
    sums = np.zeros((len(unique), 64), dtype=np.float64)
    np.add.at(sums, inverse, mel.astype(np.float32) / 255.)
    memory = (sums / np.bincount(inverse)[:, None]).astype(np.float32)
    return dict(local=torch.from_numpy(local), song=torch.from_numpy(memory),
                song_beats=torch.from_numpy(unique.astype(np.float32) + .5),
                query_beats=torch.from_numpy(((times - offset_ms) / beat_ms).astype(np.float32)))


class MusicContextEncoder(nn.Module):
    def __init__(self, context_size: int, cfg: MusicContextConfig):
        super().__init__()
        self.cfg = cfg
        self.local = SpectralEncoder(cfg.width)
        self.rhythm = nn.Linear(context_size, cfg.width)
        if cfg.whole_song:
            self.song_projection = nn.Linear(64, cfg.width)
            layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, 4 * cfg.width,
                dropout=0., batch_first=True, norm_first=True, activation="gelu")
            self.song_encoder = nn.TransformerEncoder(layer, cfg.song_layers, enable_nested_tensor=False)
            self.retrieve = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(cfg.width)

    def _position(self, beats):
        return timestep_embedding(beats.reshape(-1), self.cfg.width).reshape(*beats.shape, self.cfg.width)

    def forward(self, c, local, song, song_beats, query_beats):
        query = self.local(local) + self.rhythm(c.transpose(1, 2)) + self._position(query_beats)
        if self.cfg.whole_song:
            memory = self.song_encoder(self.song_projection(song) + self._position(song_beats))
            query = query + self.retrieve(query, memory, memory, need_weights=False)[0]
        return self.norm(query)


class ContextResidual(nn.Module):
    """Keep the original embedding operation, adding a learnable acoustic residual."""
    def __init__(self, original: nn.Module, context_size: int, width: int, hidden_size: int):
        super().__init__()
        self.original = original
        self.context_size = context_size
        self.audio_projection = nn.Linear(width, hidden_size, bias=False)
        nn.init.zeros_(self.audio_projection.weight)

    def forward(self, x, c):
        return (self.original(x, c[..., :self.context_size])
                + self.audio_projection(c[..., self.context_size:]))


class AudioConditionedCoord(nn.Module):
    """Owns the supplied base model; train only the new context branch in the pilot.

    This prototype takes unpadded per-song batches. It deliberately does not expose
    the base model's currently unsupported padding argument as a working feature.
    """
    def __init__(self, base: nn.Module, cfg: MusicContextConfig):
        super().__init__()
        self.cfg = cfg
        self.context_size = base.context_size
        for p in base.parameters():
            p.requires_grad_(False)
        hidden_size = base.context_embedder.mlp[0].out_features
        base.context_embedder = ContextResidual(base.context_embedder, self.context_size, cfg.width, hidden_size)
        self.base = base
        self.music = MusicContextEncoder(self.context_size, cfg)
        if cfg.object_context:
            self.objects = ObjectContextEncoder(cfg.width, cfg.heads)
        if cfg.spacing_plan:
            self.planner = SpacingPlanner(cfg.width, cfg.heads)

    def encode_condition(self, c, music, objects):
        context = self.music(c, **music)
        parameters = None
        if self.cfg.spacing_plan:
            if objects is None:
                raise ValueError('Spacing planner requires complete object layout')
            plan, parameters = self.planner(context, objects)
            context = context + plan
        return context, parameters

    def condition(self, c, music, x, t, objects):
        context, _ = self.encode_condition(c, music, objects)
        if self.cfg.object_context:
            if objects is None:
                raise ValueError('Object-conditioned model requires complete object layout')
            context = context + self.objects(x, t, context, objects)
        return torch.cat((c, context.transpose(1, 2)), dim=1)

    def forward(self, x, t, c, y, music, attn_mask=None, objects=None):
        return self.base(x, t, self.condition(c, music, x, t, objects), y, attn_mask=attn_mask)

    def save_adapter(self, path: str | Path, base_sha256: str):
        if any(p.requires_grad for p in self.base.context_embedder.original.parameters()) or any(
                p.requires_grad for p in self.base.blocks.parameters()):
            raise ValueError('A jointly trained model requires a full checkpoint, not an adapter')
        torch.save(dict(format="autoosu-coordinate-audio-prototype/2", config=asdict(self.cfg),
            base_sha256=base_sha256, music=self.music.state_dict(),
            objects=self.objects.state_dict() if self.cfg.object_context else None,
            planner=self.planner.state_dict() if self.cfg.spacing_plan else None,
            projection=self.base.context_embedder.audio_projection.state_dict()), path)

    @classmethod
    def load_adapter(cls, base, path, base_sha256):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["format"] not in ("autoosu-coordinate-audio-prototype/1", "autoosu-coordinate-audio-prototype/2") or record["base_sha256"] != base_sha256:
            raise ValueError("Adapter format or base checkpoint identity mismatch")
        model = cls(base, MusicContextConfig(**record["config"]))
        model.music.load_state_dict(record["music"])
        if model.cfg.object_context:
            model.objects.load_state_dict(record['objects'])
        if model.cfg.spacing_plan:
            model.planner.load_state_dict(record['planner'])
        model.base.context_embedder.audio_projection.load_state_dict(record["projection"])
        return model
