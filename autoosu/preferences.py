"""Measured skill tendencies using existing v0 density/distance conditions.

These descriptors are response checks, not a trained skill classifier. Compare
with the same-song balanced reference at a similar measured star rating.
"""
from __future__ import annotations

import math

import numpy as np

from .beatmap import Circle
from .controls import smooth_density


CONDITIONS = {
    "jumps": dict(density_scale=1., spacing_scale=1.20),
    "streams": dict(density_scale=1.5, spacing_scale=.80),
}
STAR_TOLERANCE = .5


def preference_options(options):
    """Fill only unset controls; explicit values and curves remain authoritative."""
    result = dict(options)
    if result["spacing_scale"] is None:
        result["spacing_scale"] = CONDITIONS[options["skill_preference"]]["spacing_scale"]
    return result


def preference_density(times, events, preference):
    """Scale the reference's observed density, retaining its section shape.

The reference already contains any highlight emphasis, so do not apply that
boost a second time. Use the same +/-8-measure smoothing as training.
"""
    values = np.zeros(len(times), dtype=float)
    if len(times) > 1:
        step = times[1] - times[0]
        for event in events:
            index = int(round((event.time - times[0]) / step))
            if 0 <= index < len(values):
                values[index] += 16
    return np.clip(smooth_density(values) * CONDITIONS[preference]["density_scale"], 0, 16).astype(np.float32)


def pattern_metrics(beatmap, beat_length):
    """Count quarter-beat circle runs and separated half/whole-beat jumps.

Run lengths >=5 include both bursts and streams; this is a continuous-tapping
tendency, not a guarantee of long streams. Sliders, spinners and gaps break runs.
The jump threshold is four circle radii, never applied to rapid stream spacing.
"""
    objects = beatmap.hit_objects
    radius = max(1., 54.4 - 4.48 * beatmap.cs)
    jumps, jump_distances, runs, run = 0, [], [], 0
    for index, obj in enumerate(objects):
        previous = objects[index - 1] if index else None
        if not isinstance(obj, Circle):
            if run:
                runs.append(run)
            run = 0
            continue
        gap = obj.time - previous.time if previous is not None else 0
        if run and abs(gap - beat_length / 4) <= 3:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 1
        if isinstance(previous, Circle) and .375 * beat_length <= gap <= 1.25 * beat_length:
            distance = math.dist((previous.x, previous.y), (obj.x, obj.y))
            jump_distances.append(distance)
            jumps += distance >= 4 * radius
    if run:
        runs.append(run)
    tapping = [length for length in runs if length >= 5]
    return dict(objects=len(objects), jump_pairs=int(jumps),
                jump_fraction=jumps / max(1, len(objects) - 1),
                jump_spacing_p50_px=float(np.median(jump_distances)) if jump_distances else 0.,
                tapping_runs=len(tapping), tapping_objects=sum(tapping),
                tapping_fraction=sum(tapping) / max(1, len(objects)),
                longest_run=max(runs, default=0))


def preference_shift(preference, observed, baseline):
    """Require a visible pattern-fraction change, not just a higher NPS/SR."""
    if preference == "streams":
        return observed["tapping_fraction"] >= baseline["tapping_fraction"] + .05
    fraction_increased = observed["jump_fraction"] >= baseline["jump_fraction"] + .05
    spacing_increased = (min(observed.get("jump_pairs", 0), baseline.get("jump_pairs", 0)) >= 8
                         and observed["jump_fraction"] >= baseline["jump_fraction"] - .01
                         and observed.get("jump_spacing_p50_px", 0) >= max(
                             baseline.get("jump_spacing_p50_px", 0) * 1.15,
                             baseline.get("jump_spacing_p50_px", 0) + 16))
    return fraction_increased or spacing_increased


def select_preference(attempts, preference, target):
    """Safety first, then same-star response; retain a usable reference on misses."""
    baseline = attempts[0]["patterns"]
    reference_stars = attempts[0]["measured"].get("stars")
    goal = target if target is not None else attempts[0]["measured"].get("stars")
    eligible = [i for i, attempt in enumerate(attempts) if not attempt["rejection_reasons"]]
    close = [i for i in eligible if goal is not None
             and abs(attempts[i]["measured"]["stars"] - goal) <= STAR_TOLERANCE]
    matched = [i for i in close if i and reference_stars is not None
               and abs(attempts[i]["measured"]["stars"] - reference_stars) <= STAR_TOLERANCE
               and preference_shift(preference, attempts[i]["patterns"], baseline)]
    if matched:
        chosen = min(matched, key=lambda i: abs(attempts[i]["measured"]["stars"] - goal))
    elif target is None and 0 in eligible:
        chosen = 0
    elif eligible and goal is not None:
        chosen = min(eligible, key=lambda i: abs(attempts[i]["measured"]["stars"] - goal))
    else:
        chosen = eligible[0] if eligible else 0
    return chosen, dict(requested=preference, status="observed" if chosen in matched else "not_observed",
                        reference_stars=attempts[0]["measured"].get("stars"), comparison_target=goal,
                        star_tolerance=STAR_TOLERANCE, baseline=baseline, observed=attempts[chosen]["patterns"],
                        interpretation="Pattern response relative to this balanced reference; not a human skill/style verdict")
