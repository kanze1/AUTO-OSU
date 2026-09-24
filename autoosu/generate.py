"""End-to-end pipeline: audio file -> .osz"""
from __future__ import annotations

import dataclasses
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import AudioAnalysis, analyze, load_audio
from .audio_io import VIDEO_EXTS
from .beatmap import Beatmap, Break, HitObject, Slider, Spinner, TimingPoint
from .difficulty import DifficultyPreset, get_preset
from .controls import (accent_events, candidate_rejections, controlled_sections, density_profile,
                       region_metrics, resolve_highlights, validate_controls)
from .metrics import compact_measurement, inspect_structure, measure_difficulty
from .package import extract_cover, prepare_audio, prepare_background, read_metadata, sanitize, write_osz
from .provenance import atomic_json, build_manifest, engine_info, fingerprint, model_identity, source_tags, store_record
from .placement import place
from .rhythm import RhythmEvent, Sections, analyse_sections, build_events
from .timing import Timing, estimate_timing
from .watermark import embed_watermark

# osu! convention: ranked beatmaps place hit objects ~26 ms *before* the audio transient as
# decoded by ffmpeg/librosa (community measurement, and what Mapperatorinator reproduces).
# Players calibrate their offsets against ranked maps, so we follow the same convention.
OSU_TIMING_SHIFT_MS = 26

ProgressFn = Callable[[float, str], None]
SvOverride = Tuple[int, int, float]          # start ms, end ms, slider velocity multiplier


@dataclass
class DiffResult:
    preset: DifficultyPreset
    events: List[RhythmEvent]
    beatmap: Beatmap
    star_condition: Optional[float] = None
    measurement: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    watermark: dict = field(default_factory=dict)
    control_report: dict = field(default_factory=dict)

    def summary(self) -> dict:
        objs = self.beatmap.hit_objects
        n = len(objs)
        sliders = sum(isinstance(o, Slider) for o in objs)
        spinners = sum(isinstance(o, Spinner) for o in objs)
        span = (objs[-1].end_time - objs[0].time) / 1000.0 if n > 1 else 1.0
        return {"objects": n, "circles": n - sliders - spinners, "sliders": sliders,
                "spinners": spinners, "nps": round(n / max(span, 1e-6), 2),
                "length_s": round(span, 1), "star_condition": self.star_condition,
                "measured_stars": self.measurement.get("stars"),
                "measurement_status": self.measurement.get("status", "unavailable"),
                "measurement": compact_measurement(self.measurement), "watermark": self.watermark,
                "controls": self.control_report}


@dataclass
class GenerateResult:
    osz: Path
    audio_file: Path
    timing: Timing
    analysis: AudioAnalysis
    diffs: List[DiffResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    osu_shift_ms: int = OSU_TIMING_SHIFT_MS
    device: str = "cpu"
    provenance: Optional[dict] = None
    provenance_recorded: bool = False
    warnings: List[str] = field(default_factory=list)
    evaluation_path: Optional[Path] = None


def approach_ms(ar: float) -> float:
    return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5


def compute_breaks(objects: List[HitObject], ar: float, min_gap_ms: int = 5000) -> List[Break]:
    breaks: List[Break] = []
    pre = int(approach_ms(ar))
    for prev, nxt in zip(objects, objects[1:]):
        if nxt.time - prev.end_time >= min_gap_ms:
            breaks.append(Break(prev.end_time + 200, nxt.time - pre))
    return breaks


def apply_shift(objects: List[HitObject], shift_ms: int) -> None:
    """Move every object earlier by shift_ms (in place)."""
    for o in objects:
        o.time -= shift_ms
        if isinstance(o, Spinner):
            o.end -= shift_ms


def effective_preset(preset: DifficultyPreset, timing: Timing) -> DifficultyPreset:
    """Scale the slider velocity down on fast songs so sliders keep a sane pixel length."""
    factor = float(np.clip(180.0 / timing.bpm, 0.75, 1.0))
    sm = round(preset.slider_multiplier * factor * 20) / 20
    return dataclasses.replace(preset, slider_multiplier=sm) if sm != preset.slider_multiplier else preset


def build_beatmap(preset: DifficultyPreset, timing: Timing, objects: List[HitObject],
                  kiai: List[tuple], audio_filename: str, title: str, artist: str,
                  creator: str, shift_ms: int = OSU_TIMING_SHIFT_MS,
                  sv_overrides: Sequence[SvOverride] = (), kiai_sv: Optional[float] = None) -> Beatmap:
    kiai_sv = preset.kiai_sv if kiai_sv is None else kiai_sv
    apply_shift(objects, shift_ms)
    tps = [TimingPoint(int(round(timing.offset_ms)) - shift_ms, timing.beat_length, uninherited=True)]
    for start, end in kiai:
        tps.append(TimingPoint(start - shift_ms, -100.0 / kiai_sv, uninherited=False, kiai=True))
        tps.append(TimingPoint(end - shift_ms, -100.0, uninherited=False, kiai=False))

    def state_at(t: int) -> Tuple[float, bool]:
        sv, k = 1.0, False
        for a, b in kiai:
            if a <= t < b:
                sv, k = kiai_sv, True
        return sv, k

    for start, end, scale in sv_overrides:      # sliders that had to be shortened to fit the screen
        sv, k = state_at(start)
        tps.append(TimingPoint(start - shift_ms, -100.0 / (sv * scale), uninherited=False, kiai=k))
        sv2, k2 = state_at(end)
        tps.append(TimingPoint(end - shift_ms, -100.0 / sv2, uninherited=False, kiai=k2))
    preview = (kiai[0][0] - shift_ms) if kiai else (objects[len(objects) * 2 // 5].time if objects else -1)
    return Beatmap(
        audio_filename=audio_filename, title=title, artist=artist, version=preset.name, creator=creator,
        hp=preset.hp, cs=preset.cs, od=preset.od, ar=preset.ar,
        slider_multiplier=preset.slider_multiplier, distance_spacing=preset.spacing,
        preview_time=preview, timing_points=tps,
        breaks=compute_breaks(objects, preset.ar), hit_objects=objects,
    )


def pick_device(device: Optional[str] = None) -> str:
    from .devices import resolve_device

    return resolve_device(device)


def cached_model(cache: Dict, kind: str, path: str, device: str, loader):
    """A batch owns this cache; never retain models globally after it finishes."""
    source = Path(path).resolve()
    stat = source.stat()
    key = (kind, str(source), stat.st_mtime_ns, stat.st_size, device)
    if key not in cache:
        cache[key] = loader(path, device)
    return cache[key]


def generate(audio_path: str | Path, difficulties: List[str], out_dir: str | Path = "out",
             seed: int = 0, bpm: Optional[float] = None, offset_ms: Optional[float] = None,
             title: Optional[str] = None, artist: Optional[str] = None,
             creator: str = "AUTO-OSU", osu_shift_ms: int = OSU_TIMING_SHIFT_MS, log=print,
             rhythm_model: Optional[str] = None, temperature: float = 0.9, density: Optional[float] = None,
             density_bias: float = 0.0, star_rating: Optional[float] = None, decode_steps: int = 12,
             coord_model: Optional[str] = None, coord_steps: int = 100, cfg_scale: float = 1.0,
             device: Optional[str] = None, progress: Optional[ProgressFn] = None,
             model_cache: Optional[Dict] = None, records_dir: Optional[Path] = None,
             controls: Optional[dict] = None) -> GenerateResult:
    """Analyse a song and write one .osz with the requested difficulties.

    rhythm_model / coord_model: paths to the trained models; without them the rule-based layers run.
    progress(fraction, message) is called as work advances (for GUIs); log(text) gets the human summary.
    """
    t0 = _time.perf_counter()
    if star_rating is not None and (not np.isfinite(star_rating) or not 0 < star_rating <= 12):
        raise ValueError("Star condition must be finite and between 0 and 12")
    options = validate_controls(controls)
    if density is not None:
        if options["density"] is not None or options["density_curve"]:
            raise ValueError("Density is specified both as an argument and in generation controls")
        options = validate_controls(dict(options, density=density))
    if options["target_stars"] is not None and not (rhythm_model and coord_model):
        raise ValueError("Target-star candidate selection requires both models")
    if (options["density"] is not None or options["density_curve"]) and not rhythm_model:
        raise ValueError("Density conditioning requires the rhythm model")
    audio_path, out_dir = Path(audio_path), Path(out_dir)
    presets = [get_preset(d) for d in difficulties]
    if not presets:
        raise ValueError("Choose at least one difficulty")
    dev = pick_device(device) if (rhythm_model or coord_model) else "cpu"
    cache = model_cache if model_cache is not None else {}
    engine = engine_info(model_identity("rhythm", rhythm_model, cache), model_identity("coord", coord_model, cache))
    report = progress or (lambda f, m: None)

    report(0.0, "load")
    log(f"[1/4] loading {audio_path.name}")
    y, sr = load_audio(audio_path)
    report(0.05, "analyse")
    log(f"[2/4] analysing audio ({len(y) / sr:.1f} s)")
    analysis = analyze(y, sr)
    options = validate_controls(options, analysis.duration)
    log(f"      {len(analysis.onsets)} onsets detected")

    report(0.18, "timing")
    log("[3/4] estimating timing")
    timing = estimate_timing(analysis, bpm_override=bpm, offset_override_ms=offset_ms)
    log(f"      BPM {timing.bpm:g}  offset {timing.offset_ms:.0f} ms (written as {timing.offset_ms - osu_shift_ms:.0f} ms, osu! convention)")

    meta_title, meta_artist = read_metadata(audio_path)
    title, artist = title or meta_title, artist or meta_artist
    workdir = out_dir / ".work"
    audio_file = prepare_audio(audio_path, workdir)
    background = None
    cover = extract_cover(audio_path)
    if cover:
        background = prepare_background(cover[0], cover[1], workdir)
    elif audio_path.suffix.lower() in VIDEO_EXTS:
        from .audio_io import extract_video_frame

        background = extract_video_frame(audio_path, workdir / "bg.jpg")
    if background:
        log(f"      background: {background.name} ({'cover art' if cover else 'video frame'})")
    plan = resolve_highlights(options, analysis, timing)
    sections = controlled_sections(analyse_sections(analysis, timing), plan, timing, options["highlight_density"])
    kiai = sections.kiai
    log(f"      {len(kiai)} kiai section(s): " + ", ".join(f"{a / 1000:.0f}-{b / 1000:.0f}s" for a, b in kiai))

    model = mel = cm = None
    if rhythm_model:
        from .ml.model import TickTransformer
        from .ml.sample import song_mel_uint8

        report(0.22, "load rhythm model")
        model = cached_model(cache, "rhythm", rhythm_model, dev, TickTransformer.load)
        mel = song_mel_uint8(audio_path)
        log(f"      rhythm model {Path(rhythm_model).name} on {dev}")
    if coord_model:
        from .ml.coord_infer import load_coord_model

        report(0.25, "load coordinate model")
        cm = cached_model(cache, "coord", coord_model, dev, load_coord_model)
        log(f"      coordinate model {Path(coord_model).name} on {dev}")
    if engine != engine_info(model_identity("rhythm", rhythm_model, cache), model_identity("coord", coord_model, cache)):
        raise RuntimeError("Model file changed while loading; retry generation")

    log("[4/4] generating difficulties")
    diffs: List[DiffResult] = []
    map_records = []
    evaluations, warnings = [], []
    span = 0.68 / max(1, len(presets))
    for i, preset in enumerate(presets):
        base = 0.28 + i * span
        preset = effective_preset(preset, timing)
        star = star_rating if star_rating is not None else (options["target_stars"] or preset.star)
        count = options["candidates"] if options["target_stars"] is not None else 1
        candidates, attempts = [], []
        for attempt in range(count):
            attempt_options = dict(options)
            condition = star
            reference_candidate = None
            if attempt:
                reference_candidate = 0
                measured = candidates[0].measurement.get("stars")
                correction = options["target_stars"] - measured if measured is not None else 0.
                condition = float(np.clip(star + correction, 1, 12))
                if attempt == 2:
                    usable = [j for j, a in enumerate(attempts) if not a["rejection_reasons"]] or [0]
                    reference_candidate = min(usable, key=lambda j: abs((candidates[j].measurement.get("stars") or 0)-options["target_stars"]))
                    reference = candidates[reference_candidate]
                    measured = reference.measurement.get("stars") or options["target_stars"]
                    condition = reference.star_condition
                    if options["spacing_scale"] is None:
                        attempt_options["spacing_scale"] = float(np.clip((options["target_stars"]/max(measured, 1.))**2, .5, 1.5))
                    # Retain the reference rhythm while testing distance. Changing
                    # density as well made the last candidate overshoot abruptly.
            rng = np.random.default_rng(seed * 1000 + i)
            report(base + span * attempt / count, f"{preset.name}: candidate {attempt+1}/{count} rhythm")
            if model is not None:
                from .ml.sample import generate_rhythm, rhythm_grid
                times = rhythm_grid(mel, timing)[1]
                common = dict(star_rating=condition, temperature=temperature, none_bias=density_bias,
                              decode_steps=decode_steps, seed=seed * 1000 + i, device=dev)
                reference = None
                if plan["regions"] and options["highlight_density"] and attempt_options["density"] is None and not attempt_options["density_curve"]:
                    reference = generate_rhythm(model, mel, timing, preset, analysis, sections, **common)
                profile = density_profile(times, attempt_options, plan["regions"], baseline_events=reference)
                events = generate_rhythm(model, mel, timing, preset, analysis, sections, density=profile, **common)
            else:
                events = build_events(analysis, timing, preset, rng, sections)
            accent_events(events, plan["regions"])
            kiai_sv = preset.kiai_sv if options["highlight_mode"] == "legacy" or options["highlight_sv"] else 1.
            sv_sections = [(a, b, kiai_sv) for a, b in kiai]
            overrides: List[SvOverride] = []
            if cm is not None:
                from .ml.coord_infer import place_with_model

                def coord_progress(f: float, msg: str, _b=base, _n=preset.name, _a=attempt) -> None:
                    report(_b + span * (_a + .05 + .95 * f) / count, f"{_n}: placing ({msg})")

                placed = place_with_model(events, preset, timing, cm, sv_sections, star=condition, seed=seed * 1000 + i,
                                          steps=coord_steps, cfg_scale=cfg_scale, progress=coord_progress,
                                          spacing_scale=attempt_options["spacing_scale"],
                                          highlight_regions=plan["regions"] if options["highlight_spacing"] else ())
                objects, overrides = placed.objects, placed.sv_overrides
            else:
                placement_preset = dataclasses.replace(preset, spacing=preset.spacing * (attempt_options["spacing_scale"] or 1.))
                objects = place(events, placement_preset, timing, rng, sv_sections)
            bm = build_beatmap(preset, timing, objects, kiai, audio_file.name, title, artist, creator, osu_shift_ms,
                               sv_overrides=overrides, kiai_sv=kiai_sv)
            bm.tags = source_tags(engine)
            if background:
                bm.background = background.name
            watermark = embed_watermark(bm, engine)
            measurement = measure_difficulty(bm.to_osu())
            diagnostics = inspect_structure(bm, timing.beat_length, len(overrides))
            used_star = condition if rhythm_model or coord_model else None
            candidate = DiffResult(preset, events, bm, used_star, measurement, diagnostics, watermark)
            reasons = candidate_rejections(measurement, diagnostics, candidates[0].measurement if candidates else None)
            attempts.append(dict(index=attempt, reference_candidate=reference_candidate, star_condition=used_star, density=attempt_options["density"],
                                 spacing_scale=attempt_options["spacing_scale"], measured=compact_measurement(measurement),
                                 diagnostics=diagnostics, peaks=measurement.get("peaks"), rejection_reasons=reasons,
                                 raw_sha256=fingerprint(bm.to_osu().encode())["raw_sha256"]))
            candidates.append(candidate)
            if options["target_stars"] is not None and not reasons and abs(measurement["stars"]-options["target_stars"]) <= .5:
                break
        eligible = [j for j, a in enumerate(attempts) if not a["rejection_reasons"]]
        if options["target_stars"] is not None and eligible:
            chosen = min(eligible, key=lambda j: abs(candidates[j].measurement["stars"]-options["target_stars"]))
        else:
            chosen = 0
        res = candidates[chosen]
        bm, measurement, diagnostics, watermark, used_star = res.beatmap, res.measurement, res.diagnostics, res.watermark, res.star_condition
        target = options["target_stars"]
        achieved = target is not None and chosen in eligible and abs(measurement["stars"]-target) <= .5
        res.control_report = dict(options=options, highlight_plan=plan, candidates=attempts, selected=chosen,
                                  target_stars=target, target_met=achieved if target is not None else None,
                                  selection_status="accepted" if chosen in eligible else "fallback_unverified",
                                  regions=region_metrics(bm, measurement, plan["regions"], osu_shift_ms))
        if target is not None:
            log(f"      target {target:g}: {'within tolerance' if achieved else 'not reached'}; selected candidate {chosen+1}/{len(attempts)}")
            if not achieved:
                warnings.append(f"{preset.name}: target {target:g} stars not reached within valid-candidate tolerance; see actual measured rating")
        diffs.append(res)
        map_records.append(dict(filename=sanitize(bm.osu_filename()), preset=dataclasses.asdict(preset),
                                seed=seed * 1000 + i, star_condition=used_star,
                                measured_difficulty=compact_measurement(measurement), watermark=watermark, controls=res.control_report,
                                **fingerprint(bm.to_osu().encode("utf-8"))))
        evaluations.append(dict(filename=map_records[-1]["filename"], raw_sha256=map_records[-1]["raw_sha256"],
                                star_condition=used_star, measurement=measurement, diagnostics=diagnostics,
                                watermark=watermark, controls=res.control_report))
        s = res.summary()
        log(f"      {preset.name:<7} {s['objects']:4d} objects "
            f"({s['circles']} circles, {s['sliders']} sliders, {s['spinners']} spinners) "
            f"{s['nps']:.2f} obj/s" + (f", {diagnostics['shortened_sliders']} slider(s) shortened" if diagnostics['shortened_sliders'] else ""))
        if measurement["status"] == "ok":
            condition = f", model condition {used_star:g}" if used_star is not None else ""
            log(f"      measured {measurement['stars']:.2f} stars (NM stable){condition}")
        else:
            warning = f"{preset.name}: difficulty measurement unavailable: {measurement.get('reason', 'unknown error')}"
            warnings.append(warning)
            log(warning)
        log(f"      content watermark: {watermark['status']} ({watermark['ticks']} musical ticks)")

    report(0.97, "package")
    manifest = build_manifest(map_records, engine,
                              dict(seed=seed, bpm=timing.bpm, offset_ms=timing.offset_ms,
                                   bpm_override=bpm, offset_override_ms=offset_ms, osu_shift_ms=osu_shift_ms,
                                   temperature=temperature, density=density, density_bias=density_bias,
                                   decode_steps=decode_steps, coord_steps=coord_steps, cfg_scale=cfg_scale, device=dev, controls=options),
                              audio_path, audio_file)
    evaluation = dict(schema="autoosu.evaluation/1", generation_id=manifest["generation_id"], maps=evaluations)
    osz = write_osz([d.beatmap for d in diffs], audio_file, out_dir, extra_files=[background] if background else (),
                    manifest=manifest, evaluation=evaluation)
    evaluation_path = osz.with_suffix(".evaluation.json")
    try:
        atomic_json(evaluation_path, evaluation)
    except OSError as exc:
        evaluation_path = None
        warnings.append(f"Evaluation is in the .osz, but its separate report could not be saved: {exc}")
        log(warnings[-1])
    recorded = False
    try:
        store_record(manifest, records_dir)
        recorded = True
    except OSError as exc:
        warnings.append(f"Beatmap saved, but the local generation record could not be saved: {exc}")
        log(warnings[-1])
    report(1.0, "done")
    return GenerateResult(osz=osz, audio_file=audio_file, timing=timing, analysis=analysis, diffs=diffs,
                          elapsed_s=_time.perf_counter() - t0, osu_shift_ms=osu_shift_ms, device=dev,
                          provenance=manifest, provenance_recorded=recorded, warnings=warnings,
                          evaluation_path=evaluation_path)
