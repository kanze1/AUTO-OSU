"""Source-supervised object attributes, separate from subjective skill labels.

Sound targets are standard addition flags, not custom audio assets or sample banks.
Slider topology encodes coordinate-model anchor types, including Bezier segment boundaries.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from .osu_parse import OBJ_CIRCLE, OBJ_SLIDER, OBJ_SPINNER, _sections

ATTRIBUTE_NAMES = ("new_combo", "hitsound", "tail_hitsound", "repeats", "topology")


def slider_topology(curve):
    parts = curve.split("|")
    kind = parts[0]
    if kind not in ("L", "B", "P", "C") or len(parts) < 2:
        raise ValueError("Invalid source slider curve")
    points = [tuple(float(v) for v in p.split(":")) for p in parts[1:]]
    if any(len(p) != 2 or not np.isfinite(p).all() for p in points):
        raise ValueError("Invalid source anchors")
    # P with more/less than two post-head points is interpreted as Bezier by osu!.
    if kind == "P" and len(points) != 2:
        kind = "B"
    types = []
    for i in range(len(points) - 1):
        if kind == "B":
            if points[i] == points[i + 1]:
                types.append(9)
            elif i == 0 or points[i] != points[i - 1]:
                types.append(6)
        else:
            types.append({"L": 9, "P": 7, "C": 8}[kind])
    return kind + ":" + ".".join(map(str, types))


def source_attributes(content):
    """Return time/type identities plus attributes; ambiguous duplicates stay unknown."""
    records = []
    for line in _sections(content).get("HitObjects", []):
        p = line.split(",")
        if len(p) < 5:
            raise ValueError("Source hit object lacks required fields")
        time, flags, sound = float(p[2]), int(p[3]), int(p[4])
        kind = OBJ_SPINNER if flags & 8 else OBJ_SLIDER if flags & 2 else OBJ_CIRCLE if flags & 1 else None
        if kind is None:
            continue
        values = dict(new_combo=int(bool(flags & 4)), hitsound=(sound & 14) >> 1)
        if kind == OBJ_SLIDER:
            if len(p) < 8 or int(p[6]) < 1:
                raise ValueError("Invalid source slider")
            repeats = int(p[6])
            # Omitted edge sounds inherit the object flags; explicit zero is normal sound.
            edges = [int(s) for s in p[8].split("|")] if len(p) > 8 and p[8] else [sound] * (repeats + 1)
            if len(edges) != repeats + 1:
                raise ValueError("Source slider edge count differs from repeat count")
            values.update(hitsound=(edges[0] & 14) >> 1, tail_hitsound=(edges[-1] & 14) >> 1,
                          repeats=repeats, topology=slider_topology(p[5]))
        records.append(((time, kind), values))
    counts = Counter(key for key, _ in records)
    return {key: values for key, values in records if counts[key] == 1}


def align_attributes(objects, records, vocabulary):
    """Align to existing prepared objects without inventing labels for unmatched sources."""
    encoded = np.full((len(objects), len(ATTRIBUTE_NAMES)), -100, dtype=np.int32)
    indices = {name: {value: i for i, value in enumerate(vocabulary[name])} for name in ATTRIBUTE_NAMES}
    counts = Counter((float(o[0]), int(o[1])) for o in objects)
    unknown = Counter()
    for i, obj in enumerate(objects):
        key = float(obj[0]), int(obj[1])
        if counts[key] != 1 or key not in records:
            unknown["unmatched_or_ambiguous_object"] += 1
            continue
        for j, name in enumerate(ATTRIBUTE_NAMES):
            if name in records[key]:
                value = records[key][name]
                encoded[i, j] = indices[name].get(value, -100)
                if encoded[i, j] == -100:
                    unknown[name] += 1
    return encoded, unknown


def tick_attributes(ticks, objects, attributes):
    """Only unique, type-matching source onsets supervise attributes on the tick grid."""
    targets = np.full((len(ticks.times), len(ATTRIBUTE_NAMES)), -100, dtype=np.int64)
    assigned = set()
    ambiguous = set()
    for obj, values in zip(objects, attributes):
        insertion = int(np.searchsorted(ticks.times, obj[0]))
        candidates = [j for j in (insertion - 1, insertion) if 0 <= j < len(ticks.times)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(ticks.times[k] - obj[0]))
        expected = {OBJ_CIRCLE: 1, OBJ_SLIDER: 2, OBJ_SPINNER: 5}[int(obj[1])]
        if abs(ticks.times[j] - obj[0]) > .35 * ticks.beat_ms[j] / 4 or ticks.labels[j] != expected:
            continue
        if j in assigned:
            ambiguous.add(j)
        else:
            targets[j] = values
            assigned.add(j)
    for j in ambiguous:
        targets[j] = -100
    return targets
