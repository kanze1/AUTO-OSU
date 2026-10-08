"""Rhythm layer: decide *when* objects happen and what kind (circle / slider / spinner).

Instead of thresholding raw onsets, every grid tick gets drum features (kick / snare / hat),
a melody-onset feature and a sustain estimate. Notes are then chosen measure by measure:
the number of notes per measure follows the loudness of the section (quiet intro -> sparse,
chorus -> dense) and inside a measure the strongest drum hits win, with a bias towards strong
metrical positions and towards repeating the previous measure's rhythm.
"""
from __future__ import annotations

import math
from bisect import bisect_left, insort
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio import AudioAnalysis
from .beatmap import HS_CLAP, HS_FINISH, HS_WHISTLE
from .difficulty import DifficultyPreset
from .timing import Timing


@dataclass
class RhythmEvent:
    time: int                 # ms
    beat: float               # beats since the red line
    kind: str = "circle"      # circle | slider | spinner
    strength: float = 0.0     # overall salience 0..1
    percussive: float = 0.5   # share of drum vs melody
    kick: float = 0.0
    snare: float = 0.0
    intensity: float = 0.5    # section loudness 0..1 (drives spacing / emphasis)
    end_time: int = 0
    end_beat: float = 0.0
    repeats: int = 1
    new_combo: bool = False
    hitsound: int = 0
    slider_topology: Optional[str] = None  # learned coordinate-token structure; None for legacy models
    tail_hitsound: Optional[int] = None

    @property
    def last_beat(self) -> float:
        return self.end_beat if self.kind != "circle" else self.beat

    @property
    def emphasis(self) -> float:
        return max(self.kick, self.snare)


@dataclass
class Tick:
    beat: float
    time: int
    kick: float
    snare: float
    hat: float
    melody: float
    sustain_beats: float
    loud: float

    @property
    def drum(self) -> float:
        return max(self.kick, self.snare)

    @property
    def score(self) -> float:
        return max(self.drum, 0.8 * self.melody)

    @property
    def percussive(self) -> float:
        return self.drum / (self.drum + self.melody + 1e-9)


@dataclass
class Grid:
    """Rhythm grid actually used for one difficulty (depends on BPM as well as the preset)."""
    div: int            # snapping divisor (3/6 for triplet songs)
    base: int           # 1, 2 or 4: finest straight subdivision allowed
    min_gap: float      # minimum gap between object starts, beats
    min_slider: float   # shortest slider, beats
    quant: float        # slider length quantum, beats

    @property
    def step(self) -> float:
        return 1.0 / self.div


@dataclass
class Sections:
    """Per-measure loudness (0..1) and the kiai ranges derived from it."""
    intensity: np.ndarray
    kiai: List[Tuple[int, int]]

    def at(self, beat: float) -> float:
        if len(self.intensity) == 0:
            return 0.5
        m = int(np.clip(math.floor(beat / 4.0), 0, len(self.intensity) - 1))
        return float(self.intensity[m])


def _is_on(beat: float, div: float) -> bool:
    return abs(beat * div - round(beat * div)) < 1e-6


# --------------------------------------------------------------------------- grid

def _snap_fraction(analysis: AudioAnalysis, timing: Timing, div: int, min_strength: float = 0.3) -> float:
    bl = timing.beat_length
    tol = float(np.clip(0.3 * bl / div, 25.0, 70.0))
    ok = total = 0
    for o in analysis.onsets:
        if o.strength < min_strength:
            continue
        b = timing.beat_at(o.time * 1000.0)
        total += 1
        ok += abs(b - round(b * div) / div) * bl <= tol
    return ok / total if total else 0.0


def effective_grid(analysis: AudioAnalysis, timing: Timing, preset: DifficultyPreset) -> Grid:
    """Coarsen the preset grid until one tick is at least min_gap_ms long, so a 200 BPM song
    gets 1/2 where a 100 BPM song gets 1/4. Switch to the 1/3 family for triplet songs."""
    base = preset.max_divisor
    while base > 1 and timing.beat_length / base < preset.min_gap_ms:
        base //= 2
    triplet = base > 1 and _snap_fraction(analysis, timing, 3) > _snap_fraction(analysis, timing, 4) + 0.05
    div = {1: 1, 2: 3, 4: 6}[base] if triplet else base
    min_gap = max(preset.min_gap_beats, 1.0 / base)
    quant = 1.0 if base == 1 else max(0.5, min_gap)
    return Grid(div=div, base=base, min_gap=min_gap, min_slider=max(preset.min_slider_beats, quant), quant=quant)


# --------------------------------------------------------------------------- features

def tick_features(analysis: AudioAnalysis, timing: Timing, grid: Grid, window_s: float = 0.025) -> List[Tick]:
    bl = timing.beat_length
    step = grid.step
    first = math.ceil(timing.beat_at(0.0) / step) * step
    last = timing.beat_at(analysis.duration * 1000.0 - 60.0)
    if last <= first:
        return []
    fps, ffps = analysis.fps, analysis.sr / analysis.fine_hop
    sub = analysis.fine_env_low
    lag = max(1, int(round(0.03 * ffps)))
    kick_env = np.clip(np.roll(sub, -lag) - np.roll(sub, lag), 0.0, None)
    onset_t = np.array([o.time for o in analysis.onsets])
    onset_sus = np.array([o.sustain for o in analysis.onsets])

    def win(env: np.ndarray, ts: float, f: float) -> float:
        a = max(0, int((ts - window_s) * f))
        b = min(len(env), int((ts + window_s) * f) + 1)
        return float(env[a:b].max()) if b > a else 0.0

    ticks: List[Tick] = []
    for b in np.arange(first, last, step):
        t = timing.ms_at(b)
        ts = t / 1000.0
        sustain = 0.0
        if len(onset_t):
            j = int(np.searchsorted(onset_t, ts))
            for k in (j - 1, j):
                if 0 <= k < len(onset_t) and abs(onset_t[k] - ts) <= 0.035:
                    sustain = max(sustain, float(onset_sus[k]))
        ticks.append(Tick(
            beat=round(float(b), 6), time=t,
            kick=win(kick_env, ts, ffps), snare=win(analysis.band_mid, ts, fps),
            hat=win(analysis.band_high, ts, fps), melody=win(analysis.onset_env_harm, ts, fps),
            sustain_beats=sustain * 1000.0 / bl, loud=float(analysis.rms[analysis.frame_at(ts)]),
        ))

    # normalise each feature relative to this song (95th percentile -> 1)
    for name in ("kick", "snare", "hat", "melody"):
        vals = np.array([getattr(t, name) for t in ticks])
        ref = float(np.percentile(vals[vals > 0], 95)) if (vals > 0).any() else 1.0
        for t in ticks:
            setattr(t, name, min(1.0, getattr(t, name) / max(ref, 1e-9)))
    return ticks


def analyse_sections(analysis: AudioAnalysis, timing: Timing) -> Sections:
    measure_ms = 4 * timing.beat_length
    n = int((analysis.duration * 1000.0 - timing.offset_ms) / measure_ms)
    if n < 2:
        return Sections(np.array([0.5] * max(n, 1)), [])
    loud = np.array([
        analysis.rms_between((timing.offset_ms + m * measure_ms) / 1000.0,
                             (timing.offset_ms + (m + 1) * measure_ms) / 1000.0)
        for m in range(n)
    ])
    sm = np.convolve(loud, np.ones(3) / 3, mode="same") if n >= 3 else loud
    lo, hi = np.percentile(sm, 10), np.percentile(sm, 90)
    inten = np.clip((sm - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    flags = inten >= 0.7
    # fill holes of up to two measures so a chorus is one kiai section, not a strobe
    i = 0
    while i < n:
        if not flags[i]:
            j = i
            while j < n and not flags[j]:
                j += 1
            if 0 < i and j < n and j - i <= 2:
                flags[i:j] = True
            i = j
        else:
            i += 1
    kiai: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i in range(n + 1):
        on = i < n and bool(flags[i])
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= 4:
                kiai.append((timing.ms_at(start * 4), timing.ms_at(i * 4)))
            start = None
    return Sections(inten, kiai)


# --------------------------------------------------------------------------- selection

def _gap_ok(chosen: List[float], beat: float, min_gap: float) -> bool:
    i = bisect_left(chosen, beat)
    if i > 0 and beat - chosen[i - 1] < min_gap - 1e-6:
        return False
    if i < len(chosen) and chosen[i] - beat < min_gap - 1e-6:
        return False
    return True


def select_ticks(ticks: List[Tick], preset: DifficultyPreset, grid: Grid, timing: Timing,
                 sections: Sections, floor: float = 0.15) -> List[Tick]:
    """Measure-by-measure top-K selection; K scales with section intensity."""
    k_max = preset.max_nps * 4 * timing.beat_length / 1000.0
    by_measure: Dict[int, List[Tick]] = {}
    for t in ticks:
        by_measure.setdefault(math.floor(t.beat / 4 + 1e-9), []).append(t)

    chosen_beats: List[float] = []
    chosen: List[Tick] = []
    prev_slots: set = set()
    for m in sorted(by_measure):
        inten = sections.at(4 * m)
        k = int(round(k_max * (0.25 + 0.75 * inten)))
        cands = []
        for t in by_measure[m]:
            slot = round(t.beat - 4 * m, 4)
            if slot == 0:
                w = 1.0
            elif _is_on(slot, 1):
                w = 0.95
            elif _is_on(slot, 2):
                w = 0.85
            else:
                w = 0.75
            s = t.score * w
            if slot in prev_slots:
                s += 0.12                       # keep the rhythm consistent across measures
            if slot == 0 and t.drum >= 0.35:
                s += 0.3                        # downbeat anchor
            cands.append((s, t))
        cands.sort(key=lambda c: -c[0])
        picked = []
        for s, t in cands:
            if len(picked) >= k or s < floor:
                break
            if _gap_ok(chosen_beats, t.beat, grid.min_gap):
                insort(chosen_beats, t.beat)
                picked.append(t)
        prev_slots = {round(t.beat - 4 * m, 4) for t in picked}
        chosen.extend(picked)
    chosen.sort(key=lambda t: t.beat)

    if preset.stream_max_len > 0 and grid.base >= 4:
        run, i = 1, 1
        while i < len(chosen):
            run = run + 1 if abs((chosen[i].beat - chosen[i - 1].beat) - 0.25) < 1e-6 else 1
            if run > preset.stream_max_len:
                del chosen[i]
                run = 1
            else:
                i += 1
    return chosen


# --------------------------------------------------------------------------- events

def _make_events(sel: List[Tick], preset: DifficultyPreset, grid: Grid, timing: Timing,
                 sections: Sections, rng: np.random.Generator) -> List[RhythmEvent]:
    quant, min_gap, min_slider = grid.quant, grid.min_gap, grid.min_slider
    max_slider = min(4.0, math.floor(400.0 / (preset.slider_multiplier * 100.0) / quant) * quant)
    easy = preset.max_divisor <= 2
    events: List[RhythmEvent] = []
    i = 0
    while i < len(sel):
        t = sel[i]
        nxt = sel[i + 1] if i + 1 < len(sel) else None
        inten = sections.at(t.beat)
        ev = RhythmEvent(time=t.time, beat=t.beat, strength=t.score, percussive=t.percussive,
                         kick=t.kick, snare=t.snare, intensity=inten)
        gap = (nxt.beat - t.beat) if nxt else 2.0 + min_gap
        p_slider = preset.slider_bias * (1.35 - 0.7 * inten)   # calmer sections: more sliders
        consumed = 1

        # (a) merge with the next note when it is close and not stronger: "kick -> snare" slider
        if (nxt is not None and gap <= 1.0 + 1e-6 and gap >= min_slider - 1e-6 and gap <= max_slider
                and nxt.drum <= t.drum * 1.05 + 0.05 and rng.random() < p_slider * (1.3 if easy else 1.0)):
            ev.kind, ev.end_beat = "slider", nxt.beat
            consumed = 2
        else:
            max_len = gap - min_gap
            if max_len >= min_slider - 1e-6:
                sustained = t.sustain_beats >= min_slider * 0.75
                if sustained and rng.random() < 0.8:
                    length = math.floor(min(t.sustain_beats, max_len, max_slider) / quant + 1e-6) * quant
                    if length >= min_slider - 1e-6:
                        ev.kind, ev.end_beat = "slider", t.beat + length
                elif rng.random() < p_slider * 0.25:
                    ev.kind, ev.end_beat = "slider", t.beat + min_slider
        if ev.kind == "slider":
            ev.end_time = timing.ms_at(ev.end_beat)
        events.append(ev)
        i += consumed
    return events


def _add_spinners(events: List[RhythmEvent], analysis: AudioAnalysis, preset: DifficultyPreset,
                  timing: Timing) -> List[RhythmEvent]:
    bl = timing.beat_length
    pad = 2.0 if preset.max_divisor == 1 else 1.0
    out: List[RhythmEvent] = []
    for i, ev in enumerate(events):
        out.append(ev)
        end_beat = ev.last_beat
        end_ms = timing.ms_at(end_beat)
        nxt = events[i + 1] if i + 1 < len(events) else None
        if nxt is None:
            tail_s = analysis.duration - end_ms / 1000.0
            if tail_s > 3.0 and analysis.rms_between(end_ms / 1000.0 + 0.5, end_ms / 1000.0 + 3.0) > 0.2:
                start = end_beat + pad
                end = min(start + 8.0, timing.beat_at(analysis.duration * 1000.0 - 300.0))
                if (end - start) * bl >= preset.min_spinner_ms:
                    out.append(RhythmEvent(time=timing.ms_at(start), beat=start, kind="spinner",
                                           end_time=timing.ms_at(end), end_beat=end, new_combo=True))
            continue
        gap_s = (nxt.beat - end_beat) * bl / 1000.0
        if gap_s >= 3.0:
            loud = analysis.rms_between(end_ms / 1000.0 + 0.3, nxt.time / 1000.0 - 0.3)
            if loud >= 0.2:
                start, end = end_beat + pad, nxt.beat - pad
                if (end - start) * bl >= preset.min_spinner_ms:
                    out.append(RhythmEvent(time=timing.ms_at(start), beat=start, kind="spinner",
                                           end_time=timing.ms_at(end), end_beat=end, new_combo=True))
    return out


def _combos_and_hitsounds(events: List[RhythmEvent], preset: DifficultyPreset) -> None:
    count = 0
    last_measure: Optional[int] = None
    prev: Optional[RhythmEvent] = None
    max_combo = 8 if preset.max_divisor == 1 else 12
    for ev in events:
        if ev.kind == "spinner":
            ev.new_combo = True
            count, last_measure, prev = 0, None, ev
            continue
        measure = math.floor(ev.beat / 4 + 1e-6)
        nc = (prev is None or prev.kind == "spinner"
              or (measure != last_measure and count >= preset.combo_min)
              or count >= max_combo
              or ev.beat - prev.last_beat >= 4.0)
        if nc:
            count, last_measure = 0, measure
        ev.new_combo = nc
        count += 1

        if ev.snare >= 0.5 and ev.snare > ev.kick:
            ev.hitsound |= HS_CLAP
        if _is_on(ev.beat, 1) and round(ev.beat) % 4 == 0 and ev.kick >= 0.6 and ev.intensity >= 0.6:
            ev.hitsound |= HS_FINISH
        if ev.emphasis < 0.25 and ev.strength >= 0.4:
            ev.hitsound |= HS_WHISTLE
        prev = ev


def build_events(analysis: AudioAnalysis, timing: Timing, preset: DifficultyPreset,
                 rng: np.random.Generator, sections: Optional[Sections] = None) -> List[RhythmEvent]:
    sections = sections or analyse_sections(analysis, timing)
    grid = effective_grid(analysis, timing, preset)
    ticks = tick_features(analysis, timing, grid)
    sel = select_ticks(ticks, preset, grid, timing, sections)
    events = _make_events(sel, preset, grid, timing, sections, rng)
    events = _add_spinners(events, analysis, preset, timing)
    _combos_and_hitsounds(events, preset)
    return events


def detect_kiai(analysis: AudioAnalysis, timing: Timing) -> List[Tuple[int, int]]:
    return analyse_sections(analysis, timing).kiai
