"""Positive-only community skill conditions; zero means unspecified, never negative."""
from __future__ import annotations

import numpy as np

SKILL_TAGS = {
    "jumps": "skillset/jumps",
    "bursts": "streams/bursts",
    "streams": "skillset/streams",
    "slider_tech": "tech/slider tech",
}
SKILL_NAMES = tuple(SKILL_TAGS)


def skill_vector(names, labels=None):
    labels = labels or {}
    unknown = set(labels) - set(names)
    if unknown:
        raise ValueError(f"Unsupported model skill labels: {sorted(unknown)}")
    if any(value != 1 for value in labels.values()):
        raise ValueError("This pilot supports positive skill conditions only; omitted means unknown")
    return np.array([float(name in labels) for name in names], dtype=np.float32)


def conditioned_vector(base, names, labels=None):
    return np.concatenate([base, skill_vector(names, labels)])
