"""Optional generation controls and audio-time highlight plans.

All persisted settings are plain JSON. A plan uses original audio seconds;
the osu export offset is applied only when writing objects and timing points.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(target_stars=None, candidates=3, density=None, density_curve=[], spacing_scale=None,
                highlight_mode="auto", highlights=[], highlight_sv=False,
                highlight_density=True, highlight_spacing=True)


def load_control_file(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("Control plan exceeds 64 KiB")
    value = json.loads(raw.decode("utf-8-sig"))
    if isinstance(value, dict) and "schema" in value:
        if value.get("schema") != "autoosu.control-plan/1" or value.keys() != {"schema", "controls"}:
            raise ValueError("Unsupported control-plan schema")
        value = value["controls"]
    return validate_controls(value)


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in {low}..{high}")
    return float(value)


def validate_controls(value=None, duration_s=None):
    if value is None:
        value = {}
    if not isinstance(value, dict) or value.keys() - DEFAULTS.keys():
        raise ValueError("Unknown generation control fields")
    out = dict(DEFAULTS, **value)
    for key, low, high in (("target_stars", 1, 12), ("density", 0, 16), ("spacing_scale", .5, 1.5)):
        if out[key] is not None:
            out[key] = _number(out[key], key, low, high)
    if type(out["candidates"]) is not int or not 1 <= out["candidates"] <= 3:
        raise ValueError("candidates must be an integer in 1..3")
    if out["highlight_mode"] not in ("legacy", "off", "manual", "auto") or any(type(out[k]) is not bool for k in ("highlight_sv", "highlight_density", "highlight_spacing")):
        raise ValueError("Invalid highlight mode or SV setting")
    curve = out["density_curve"]
    if not isinstance(curve, list) or len(curve) > 256:
        raise ValueError("density_curve must contain at most 256 [seconds, objects-per-measure] points")
    points = []
    for point in curve:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("density_curve points are [seconds, objects-per-measure]")
        t = _number(point[0], "curve time", 0, duration_s if duration_s is not None else 86400)
        d = _number(point[1], "curve density", 0, 16)
        if points and t <= points[-1][0]:
            raise ValueError("density_curve times must increase strictly")
        points.append([t, d])
    if points and out["density"] is not None:
        raise ValueError("Use a constant density or a density curve, not both")
    out["density_curve"] = points
    if not isinstance(out["highlights"], list) or len(out["highlights"]) > 8:
        raise ValueError("Use at most eight highlight regions")
    regions = []
    for region in out["highlights"]:
        if not isinstance(region, dict) or region.keys() - {"start_s", "end_s", "strength", "attack_s", "release_s"}:
            raise ValueError("Invalid highlight region fields")
        r = dict(dict(strength=1., attack_s=2., release_s=4.), **region)
        for key in ("start_s", "end_s"):
            r[key] = _number(r.get(key), key, 0, duration_s if duration_s is not None else 86400)
        for key, high in (("strength", 1), ("attack_s", 30), ("release_s", 30)):
            r[key] = _number(r[key], key, 0, high)
        if r["end_s"] <= r["start_s"] or r["strength"] == 0:
            raise ValueError("Highlight needs a positive length and strength")
        regions.append(r)
    regions.sort(key=lambda r: r["start_s"])
    if any(a["end_s"] > b["start_s"] for a, b in zip(regions, regions[1:])):
        raise ValueError("Highlight regions must not overlap")
    if out["highlight_mode"] == "manual" and not regions:
        raise ValueError("Manual highlight mode needs at least one region")
    if out["highlight_mode"] != "manual" and regions:
        raise ValueError("Manual regions require manual highlight mode")
    out["highlights"] = regions
    return out


def envelope(times_ms, regions):
    t = np.asarray(times_ms, dtype=float) / 1000
    result = np.zeros_like(t)
    for r in regions:
        start, end = r["start_s"], r["end_s"]
        weight = ((t >= start) & (t < end)).astype(float)
        if r["attack_s"]:
            ramp = np.clip((t - start + r["attack_s"]) / r["attack_s"], 0, 1)
            weight = np.where(t < start, ramp, weight)
        if r["release_s"]:
            ramp = np.clip((end + r["release_s"] - t) / r["release_s"], 0, 1)
            weight = np.where(t >= end, ramp, weight)
        result = np.maximum(result, weight * r["strength"])
    return result


def smooth_density(values, slots=16, window_measures=8):
    """Training's half-open +/-8-measure window, normalized at song edges."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    radius = window_measures * slots
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - radius), np.minimum(n, idx + radius)
    cumulative = np.r_[0., np.cumsum(values)]
    return ((cumulative[hi] - cumulative[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def density_profile(times, options, regions, *, baseline_events=None):
    """Return the local density condition, or None for the original unknown condition."""
    regions = regions if options["highlight_density"] else []
    if options["density_curve"]:
        points = np.asarray(options["density_curve"])
        values = np.interp(np.asarray(times) / 1000, points[:, 0], points[:, 1])
    elif options["density"] is not None:
        values = np.full(len(times), options["density"], dtype=float)
    elif regions and baseline_events is not None:
        values = np.zeros(len(times))
        if len(times) > 1:
            for ev in baseline_events:
                index = int(round((ev.time - times[0]) / (times[1] - times[0])))
                if 0 <= index < len(times):
                    values[index] += 16  # object starts per 16-tick measure, before smoothing
    else:
        return None
    values *= 1 + .35 * envelope(times, regions)
    return np.clip(smooth_density(values), 0, 16).astype(np.float32)


def automatic_highlights(analysis, timing):
    """Conservative bar-aligned energy/onset proposal; confidence is a heuristic."""
    measure_s = timing.beat_length * 4 / 1000
    offset_s = timing.offset_ms / 1000
    n = max(0, int((analysis.duration - offset_s) / measure_s))
    if n < 12:
        return [], dict(status="abstained", reason="too_short", confidence=0.)
    features = []
    for values in (analysis.rms, analysis.onset_env_perc, analysis.onset_env_harm):
        bars = []
        for i in range(n):
            a = analysis.frame_at(offset_s + i * measure_s)
            b = max(a + 1, analysis.frame_at(offset_s + (i + 1) * measure_s))
            bars.append(float(np.mean(values[a:b])))
        bars = np.asarray(bars)
        lo, hi = np.quantile(bars, [.1, .9])
        # Avoid amplifying tiny variations in silence or a flat signal.
        features.append(np.clip((bars-lo)/max(hi-lo, .1, abs(hi)*.2), 0, 1))
    score = .4*features[0] + .35*features[1] + .25*features[2]
    score = np.convolve(np.pad(score, (1, 1), mode="edge"), np.ones(3)/3, mode="valid")
    flags = score >= max(.6, float(np.quantile(score, .7)))
    runs, start = [], None
    for i, on in enumerate(np.r_[flags, False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            outside = np.r_[score[:start], score[i:]]
            contrast = float(score[start:i].mean() - outside.mean()) if len(outside) else 0.
            if 4 <= i-start <= max(4, n*.45) and contrast >= .18:
                runs.append((contrast, start, i))
            start = None
    chosen = sorted(sorted(runs, reverse=True)[:2], key=lambda r: r[1])
    regions = [dict(start_s=max(0., offset_s+a*measure_s), end_s=min(analysis.duration, offset_s+b*measure_s),
                    strength=1., attack_s=min(2., measure_s), release_s=min(4., measure_s*2)) for _, a, b in chosen]
    return regions, dict(status="proposed" if regions else "abstained", reason="energy_and_onset_contrast" if regions else "low_contrast",
                         confidence=min(1., max((c for c, _, _ in chosen), default=0.)/.6),
                         bar_seconds=measure_s, scores=score.tolist())


def resolve_highlights(options, analysis, timing):
    mode = options["highlight_mode"]
    if mode == "auto":
        regions, evidence = automatic_highlights(analysis, timing)
    else:
        regions, evidence = options["highlights"], dict(status="manual" if mode == "manual" else mode)
    return dict(mode=mode, regions=regions, evidence=evidence, time_basis="original_audio_seconds")


def controlled_sections(sections, plan, timing, density_enabled=True):
    from .rhythm import Sections
    if plan["mode"] == "legacy":
        return sections
    times = timing.offset_ms + (np.arange(len(sections.intensity)) * 4 + 2) * timing.beat_length
    strength = envelope(times, plan["regions"])
    intensity = np.clip(sections.intensity + (.3 if density_enabled else 0.) * strength, 0, 1)
    kiai = [(round(r["start_s"]*1000), round(r["end_s"]*1000)) for r in plan["regions"]]
    return Sections(intensity, kiai)


def accent_events(events, regions):
    from .beatmap import HS_FINISH
    for r in regions:
        selected = [e for e in events if r["start_s"]*1000 <= e.time < r["end_s"]*1000]
        if selected:
            selected[0].new_combo = True
        for ev in selected:
            ev.intensity = max(ev.intensity, .7 * r["strength"])
            if abs(ev.beat/4 - round(ev.beat/4)) < .01 and ev.strength >= .5:
                ev.hitsound |= HS_FINISH


def candidate_rejections(measurement, diagnostics, baseline=None):
    reasons = [key for key in ("overlapping_objects", "heads_outside_playfield", "invalid_sliders", "simultaneous_or_reversed_starts")
               if diagnostics.get(key, 0)]
    if measurement.get("status") != "ok":
        reasons.append("measurement_unavailable")
    for kind in ("aim", "speed"):
        peak = measurement.get("peaks", {}).get(kind, {})
        ratio = peak.get("max", 0) / max(peak.get("p95", 0), 1e-6)
        ref = (baseline or {}).get("peaks", {}).get(kind, {})
        limit = max(5., 1.25 * ref.get("max", 0) / max(ref.get("p95", 0), 1e-6))
        if ratio > limit:
            reasons.append(kind + "_isolated_peak")
    return reasons


def region_metrics(beatmap, measurement, regions, shift_ms):
    """Observed section contrasts in original-audio time, not requested strengths."""
    starts = np.array([o.time + shift_ms for o in beatmap.hit_objects], dtype=float)
    strains = measurement.get("strains", {})
    first = strains.get("first_section_end_ms")
    result = []
    for region in regions:
        start, end = region["start_s"]*1000, region["end_s"]*1000
        mask = (starts >= start) & (starts < end)
        selected = [o for o, keep in zip(beatmap.hit_objects, mask) if keep]
        gaps = [math.dist(a.end_pos, (b.x,b.y)) for a,b in zip(selected,selected[1:])]
        row = dict(start_s=region["start_s"],end_s=region["end_s"],objects=len(selected),
                   nps=len(selected)/((end-start)/1000),spacing_p50_px=float(np.median(gaps)) if gaps else 0.)
        for name in ("aim", "speed"):
            series = np.asarray(strains.get(name, []))
            if first is not None and len(series):
                times = first + np.arange(len(series))*strains["section_ms"] + shift_ms
                values = series[(times >= start) & (times < end)]
                row[name+"_strain_mean"] = float(values.mean()) if len(values) else None
        result.append(row)
    return result
