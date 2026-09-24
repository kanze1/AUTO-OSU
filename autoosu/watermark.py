"""Public content marker for newly exported maps; not an authenticity signature.

The fixed codec uses circle-coordinate parity at musical ticks. Never use a
negative result to infer human authorship. No secret, network or model is used.
"""
from __future__ import annotations

import hashlib
import math

from .beatmap import Beatmap, Circle
from .provenance import fingerprint, parse_sections

SCHEMA = "autoosu-circle-parity/1"
MIN_TICKS = 64
MAX_TICKS = 8192
THRESHOLD_PERCENT = 85
FAMILIES = ("rules", "rhythm", "coord", "model")


def _code(family: str, tick: int) -> bytes:
    return hashlib.sha256(f"{SCHEMA}:{family}:{tick}".encode("ascii")).digest()


def _tick(time: float, offset: float, beat_length: float) -> int:
    value = (time - offset) * 4 / beat_length
    if not math.isfinite(value) or abs(value) > 1e9:
        raise ValueError("Musical tick outside supported range")
    return math.floor(value + .5)


def _family(engine: dict) -> str:
    rhythm = engine.get("rhythm", {}).get("engine") == "model"
    coord = engine.get("coord", {}).get("engine") == "model"
    return "model" if rhythm and coord else "rhythm" if rhythm else "coord" if coord else "rules"


def _coordinate(value: int, bit: int, direction: int, extent: int, radius: float) -> int:
    if value % 2 == bit:
        return value
    # Preserve the full-circle margin when the original head already has it.
    lo, hi = (math.ceil(radius), math.floor(extent - radius)) if radius <= value <= extent - radius else (0, extent)
    for candidate in (value + direction, value - direction):
        if lo <= candidate <= hi:
            return candidate
    raise ValueError("Circle coordinate cannot carry a parity bit")


def embed_watermark(beatmap: Beatmap, engine: dict) -> dict:
    """Mark eligible circle heads in place, moving each axis by at most one pixel."""
    family = _family(engine)
    result = dict(schema=SCHEMA, status="insufficient", engine_claim=family, ticks=0,
                  marked_circles=0, moved_circles=0, max_displacement_px=0.)
    red = sorted((tp for tp in beatmap.timing_points if tp.uninherited), key=lambda tp: tp.time)
    if not red or not 10 <= red[0].beat_length <= 60000:
        return dict(result, status="unsupported")
    # Match the serialized clock exactly, including its six decimal places.
    offset, beat_length = red[0].time, float(f"{red[0].beat_length:.6f}")
    circles = [o for o in beatmap.hit_objects if isinstance(o, Circle)]
    groups = {}
    for obj in circles:
        groups.setdefault(_tick(obj.time, offset, beat_length), []).append(obj)
    ticks = sorted(groups)[:MAX_TICKS]
    result["ticks"] = len(ticks)
    if len(ticks) < MIN_TICKS:
        return result
    radius = max(0., min(96., 54.4 - 4.48 * beatmap.cs))
    changes = []
    for tick in ticks:
        code = _code(family, tick)
        for obj in groups[tick]:
            x = _coordinate(obj.x, code[0] & 1, 1 if code[2] & 1 else -1, 512, radius)
            y = _coordinate(obj.y, code[1] & 1, 1 if code[3] & 1 else -1, 384, radius)
            changes.append((obj, x, y))
    # Apply only after every coordinate is validated.
    for obj, x, y in changes:
        displacement = math.hypot(x - obj.x, y - obj.y)
        result["max_displacement_px"] = max(result["max_displacement_px"], displacement)
        result["moved_circles"] += int(displacement > 0)
        obj.x, obj.y = x, y
    result.update(status="embedded", marked_circles=len(changes))
    return result


def detect_watermark(raw: bytes) -> dict:
    """Bounded, metadata-independent check against a frozen public codebook.

    Repeated circles at one tick count once; conflicting bits are erasures and
    stay in the denominator. Four whole-map parity shifts are tested per family.
    The match fraction is a score, not a probability of model authorship.
    """
    result = dict(schema=SCHEMA, status="unsupported", ticks=0, bits=0,
                  threshold_percent=THRESHOLD_PERCENT, minimum_ticks=MIN_TICKS,
                  trials=len(FAMILIES) * 4)
    try:
        fingerprint(raw)  # Reject malformed / nonstandard maps before interpreting bits.
        _, sections = parse_sections(raw)
        red = []
        for line in sections.get("TimingPoints", []):
            fields = line.split(",")
            if len(fields) < 7 or int(fields[6]) == 1:
                time, length = float(fields[0]), float(fields[1])
                if length > 0:
                    red.append((time, length))
        if not red:
            raise ValueError("No red timing point")
        offset, beat_length = min(red, key=lambda row: row[0])
        if not 10 <= beat_length <= 60000:
            raise ValueError("Unsupported beat length")
        groups = {}
        for line in sections.get("HitObjects", []):
            fields = line.split(",")
            if not int(fields[3]) & 1:
                continue
            x, y, time = map(float, fields[:3])
            if not (x.is_integer() and y.is_integer() and 0 <= x <= 512 and 0 <= y <= 384):
                raise ValueError("Circle coordinates outside supported integer playfield")
            tick = _tick(time, offset, beat_length)
            votes = groups.setdefault(tick, [set(), set()])
            votes[0].add(int(x) & 1)
            votes[1].add(int(y) & 1)
        ticks = sorted(groups)[:MAX_TICKS]
        result.update(ticks=len(ticks), bits=2 * len(ticks))
        if len(ticks) < MIN_TICKS:
            return dict(result, status="insufficient")
        candidates = []
        for family in FAMILIES:
            matches = [0, 0, 0, 0]
            for tick in ticks:
                code = _code(family, tick)
                for shift in range(4):
                    matches[shift] += sum(votes == {(code[axis] & 1) ^ ((shift >> axis) & 1)}
                                          for axis, votes in enumerate(groups[tick]))
            score = max(matches)
            candidates.append(dict(engine_claim=family, matched_bits=score, parity_shift=matches.index(score)))
        candidates.sort(key=lambda row: row["matched_bits"], reverse=True)
        accepted = [row for row in candidates if row["matched_bits"] * 100 >= THRESHOLD_PERCENT * result["bits"]]
        result.update(best_match_fraction=candidates[0]["matched_bits"] / result["bits"],
                      matched_bits=candidates[0]["matched_bits"])
        if len(accepted) == 1:
            return dict(result, status="detected", **accepted[0])
        return dict(result, status="ambiguous" if accepted else "not_detected")
    except (ValueError, UnicodeError, ArithmeticError, IndexError) as exc:
        return dict(result, reason=str(exc))
