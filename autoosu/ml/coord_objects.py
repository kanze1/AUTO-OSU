"""Object ownership for coordinate tokens, independent of target coordinates.

The legacy slider end token stores the canonical path endpoint even for repeats.
Physical cursor exit is the head for even repeats and the end token for odd repeats.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .coord.models import timestep_embedding


def object_layout(times_ms, types):
    """Parse complete objects. Truncated/orphan tokens are errors, not fake circles.

    Returned indices carry no clean geometry; they also apply to generated rhythms.
    Timing features keep signed inter-object gaps, including deliberate overlaps.
    """
    times, kinds = np.asarray(times_ms, dtype=np.float64), np.asarray(types)
    if times.ndim != 1 or kinds.shape != times.shape or not len(times) or not np.isfinite(times).all():
        raise ValueError('Expected finite, nonempty matching time/type vectors')
    if not np.issubdtype(kinds.dtype, np.integer) or np.any((kinds < 0) | (kinds > 15)):
        raise ValueError('Invalid coordinate token type')
    owners = np.empty(len(times), dtype=np.int64)
    heads, ends, exits, features = [], [], [], []
    i = 0
    while i < len(times):
        start, kind = i, int(kinds[i])
        if kind in (0, 1):
            end, exit_index, category, repeat_class = i, i, 0, 0
        elif kind == 2:
            end, exit_index, category, repeat_class = i+1, i+1, 2, 0
            if end >= len(times) or kinds[end] != 3:
                raise ValueError('Incomplete spinner')
        elif kind in (4, 5):
            j = i+1
            while j < len(times) and kinds[j] in (6, 7, 8, 9):
                j += 1
            if j+1 >= len(times) or kinds[j] != 10 or kinds[j+1] not in (11, 12, 13, 14, 15):
                raise ValueError('Incomplete slider')
            end, category = j+1, 1
            repeat_class = int(kinds[end])-10
            exit_index = start if kinds[end] in (12, 14) else end
        else:
            raise ValueError('Orphan coordinate token; crop must start on an object boundary')
        if np.any(np.diff(times[start:end+1]) < 0) or (heads and times[start] < times[heads[-1]]):
            raise ValueError('Reversed object timing')
        gap = times[start]-times[ends[-1]] if ends else 0.
        duration = times[end]-times[start]
        feature = [np.log1p(duration/1000.), np.sign(gap)*np.log1p(abs(gap)/1000.), float(bool(heads)),
                   float(kind in (1, 5)), *[float(category == c) for c in range(3)],
                   *[float(repeat_class == r) for r in range(1, 6)]]
        owners[start:end+1] = len(heads)
        heads.append(start)
        ends.append(end)
        exits.append(exit_index)
        features.append(feature)
        i = end+1
    return dict(owners=torch.from_numpy(owners), heads=torch.tensor(heads), ends=torch.tensor(ends),
                exits=torch.tensor(exits), features=torch.tensor(features, dtype=torch.float32))


def complete_object_crops(times, types, budget, count=4):
    """At most `budget` tokens, preserving whole objects without padded fake nodes."""
    if budget < 2 or count < 1:
        raise ValueError('Positive crop count and token budget >= 2 required')
    layout = object_layout(times, types)
    heads, ends = layout['heads'].numpy(), layout['ends'].numpy()
    if np.any(ends-heads+1 > budget):
        raise ValueError('A complete object exceeds the crop token budget')
    # Avoid tail-only crops while covering the source at evenly spaced object boundaries.
    last = max(0, int(np.searchsorted(heads, max(0, len(times)-budget), side='right'))-1)
    starts = np.unique(np.linspace(0, last, count, dtype=int))
    crops = []
    for obj in starts:
        start = int(heads[obj])
        stop_obj = int(np.searchsorted(ends, start+budget, side='left'))-1
        stop = int(ends[stop_obj])+1
        crops.append((start, stop))
    return crops


class ObjectContextEncoder(nn.Module):
    """One state per object, derived from current noisy positions plus known rhythm.

    Object states receive equal sequence slots regardless of slider anchor count.
    The first object's predecessor is unknown at a crop boundary, not assumed central.
    """
    def __init__(self, width, heads, layers=2):
        super().__init__()
        self.motion = nn.Linear(6+12, width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4*width, dropout=0., batch_first=True,
                                          norm_first=True, activation='gelu')
        self.objects = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.relative_anchor = nn.Linear(2, width)
        self.norm = nn.LayerNorm(width)
        self.width = width

    def forward(self, x, t, context, layout):
        # Unpadded per-song batches share the same object structure.
        positions = x.transpose(1, 2)
        head = positions[:, layout['heads']]
        exit_pos = positions[:, layout['exits']]
        previous = torch.cat((head[:, :1], exit_pos[:, :-1]), dim=1)
        features = layout['features'][None].expand(len(x), -1, -1)
        geometry = torch.cat((head, exit_pos-head, head-previous, features), dim=-1)
        state = context[:, layout['heads']] + self.motion(geometry)
        state = state + self.time(timestep_embedding(t, self.width))[:, None]
        state = self.objects(state)
        relative = positions-head[:, layout['owners']]
        return self.norm(state[:, layout['owners']] + self.relative_anchor(relative))
