"""Pinned NM/stable difficulty measurement of serialized osu!standard outputs.

Model conditions are inputs, never measured results. Calculator failures remain
explicit and must not prevent an otherwise valid map from being saved.
"""
from __future__ import annotations

import math
from importlib import metadata

import numpy as np

from .beatmap import Beatmap, Slider, Spinner
from .ml.osu_parse import parse_osu

CALCULATOR_VERSION = "4.0.2"
EVALUATION_NAME = "autoosu-evaluation.json"


def calculator_info() -> dict:
    return dict(name="rosu-pp-py", version=CALCULATOR_VERSION, mode="osu", mods="NM", lazer=False)


def measure_difficulty(text: str) -> dict:
    result = dict(status="unavailable", calculator=calculator_info(), stars=None)
    try:
        installed = metadata.version("rosu-pp-py")
        if installed != CALCULATOR_VERSION:
            result["reason"] = f"Expected rosu-pp-py {CALCULATOR_VERSION}, found {installed}"
            return result
        import rosu_pp_py as rosu
    except (ImportError, metadata.PackageNotFoundError, OSError) as exc:
        result["reason"] = f"Calculator unavailable (requires Python 3.11+): {exc}"
        return result
    try:
        beatmap = rosu.Beatmap(content=text)
        if beatmap.mode != rosu.GameMode.Osu or beatmap.n_objects < 2:
            raise ValueError("Measurement needs at least two osu!standard objects")
        if beatmap.is_suspicious():
            raise ValueError("Calculator rejected an unusually complex map")
        calc = rosu.Difficulty(mods=0, lazer=False)
        attrs, strains = calc.calculate(beatmap), calc.strains(beatmap)
        values = {k: float(getattr(attrs, k)) for k in ("stars", "aim", "speed", "slider_factor")}
        series = {k: [float(v) for v in getattr(strains, k)] for k in ("aim", "aim_no_sliders", "speed")}
        if not all(math.isfinite(v) for v in [*values.values(), *(v for s in series.values() for v in s)]):
            raise ValueError("Calculator returned a non-finite result")
        # rosu-pp 4.0.1's strain skill starts its first section at the second
        # object's ceil(time / 400) boundary. Only map this for our v14 format;
        # old formats have their own offset conversions in the calculator.
        parsed = parse_osu(text)
        first_end = None
        if text.lstrip("\ufeff\r\n ").startswith("osu file format v14") and parsed.n_objects == beatmap.n_objects:
            first_end = math.ceil(float(parsed.objects[1, 0]) / strains.section_length) * strains.section_length
        result.update(status="ok", **values, max_combo=int(attrs.max_combo), objects=int(beatmap.n_objects),
                      strains=dict(section_ms=float(strains.section_length), first_section_end_ms=first_end,
                                   **series),
                      peaks={k: dict(max=max(v, default=0.0), p95=float(np.quantile(v, .95)) if v else 0.0)
                             for k, v in series.items()})
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}: {exc}")
    return result


def inspect_structure(beatmap: Beatmap, beat_length: float, shortened_sliders: int = 0) -> dict:
    """Diagnostics on the exported representation; not a playability verdict.

    A run counts consecutive circles separated by a quarter beat (+/- 3 ms).
    Overlap uses serialized slider length/SV, including the export's rounding.
    Bounds count heads and anchors, not the whole rendered slider curve.
    """
    text = beatmap.to_osu()
    parsed = parse_osu(text)
    objects = sorted(beatmap.hit_objects, key=lambda o: o.time)
    starts = np.array([o.time for o in objects], dtype=float)
    gaps = np.diff(starts)
    overlap = sum(parsed.objects[i, 0] < parsed.objects[i - 1, 2] - 3 for i in range(1, parsed.n_objects))
    heads = sum(not isinstance(o, Spinner) and not (0 <= o.x <= 512 and 0 <= o.y <= 384) for o in objects)
    anchors = sum(not (0 <= x <= 512 and 0 <= y <= 384)
                  for o in objects if isinstance(o, Slider) for x, y in o.points)
    invalid_sliders = sum(isinstance(o, Slider) and (o.length <= 0 or o.duration <= 0 or o.repeats < 1)
                          for o in objects)
    distances = [math.dist(a.end_pos, (b.x, b.y)) for a, b in zip(objects, objects[1:])
                 if not isinstance(a, Spinner) and not isinstance(b, Spinner)]
    longest, run = 0, 0
    for i, o in enumerate(objects):
        if isinstance(o, (Slider, Spinner)):
            run = 0
        elif i and run and abs(gaps[i - 1] - beat_length / 4) <= 3:
            run += 1
        else:
            run = 1
        longest = max(longest, run)
    # Maximum count in a half-open 2-second window, including recovery gaps.
    counts = np.searchsorted(starts, starts + 2000, side="left") - np.arange(len(starts))
    return dict(objects=len(objects), simultaneous_or_reversed_starts=int(np.sum(gaps <= 0)),
                overlapping_objects=int(overlap), heads_outside_playfield=int(heads),
                anchors_outside_playfield=int(anchors), invalid_sliders=int(invalid_sliders),
                shortened_sliders=shortened_sliders, longest_quarter_beat_circle_run=longest,
                peak_nps_2s=float(max(counts, default=0) / 2),
                spacing_p50_px=float(np.median(distances)) if distances else 0.0,
                spacing_p95_px=float(np.quantile(distances, .95)) if distances else 0.0)


def compact_measurement(measurement: dict) -> dict:
    return {k: v for k, v in measurement.items() if k not in ("strains", "peaks")}
