"""Command line entry point: python -m autoosu song.mp3 -d Normal Hard"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .difficulty import PRESETS
from .generate import generate
from .models import MODELS, ensure_model, find_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoosu", description="Generate an osu!standard beatmap from a song.")
    p.add_argument("audio", nargs="?", help="audio/video file, or a folder for batch generation")
    p.add_argument("--recursive", action="store_true", help="also scan subfolders for batch generation")
    p.add_argument("--check-cuda", action="store_true", help="check CUDA in this runtime and exit")
    p.add_argument("--check-source", type=Path, metavar="PATH", help="check new-map watermarks, declarations and local records in .osu, .osz or a folder")
    p.add_argument("--source-report", type=Path, metavar="JSON", help="save --check-source results as JSON")
    p.add_argument("--records-dir", type=Path, help="local generation record directory (default: ~/.autoosu/provenance)")
    p.add_argument("--setup-runtime", action="store_true", help="use uv to install and verify an app-managed GPU runtime")
    p.add_argument("-d", "--difficulty", nargs="+", default=["Hard", "Insane"],
                   metavar="NAME", help=f"difficulties to generate: {', '.join(PRESETS)}")
    p.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    p.add_argument("--seed", type=int, default=0, help="random seed (same seed = same map)")
    p.add_argument("--bpm", type=float, help="override detected BPM")
    p.add_argument("--offset", type=float, help="override detected offset (ms)")
    p.add_argument("--title", help="override song title")
    p.add_argument("--artist", help="override artist")
    p.add_argument("--creator", default="AUTO-OSU")
    p.add_argument("--osu-shift", type=int, default=26,
                   help="ms to place objects before the audio transient (osu! ranked convention, default 26)")
    g = p.add_argument_group("models")
    g.add_argument("--rules", action="store_true", help="use the rule-based rhythm and placement only (no models)")
    g.add_argument("--rhythm-model", help="path to the rhythm model (.pt); default: models/rhythm_v0.pt")
    g.add_argument("--coord-model", help="path to the coordinate model (.pt); default: models/coord_v0.pt")
    g.add_argument("--no-coord-model", action="store_true", help="rule-based placement after the ML rhythm")
    g.add_argument("--download", action="store_true", help="download missing models from the GitHub release")
    g.add_argument("--device", default="auto", help="auto | cuda | cpu")
    g.add_argument("--temperature", type=float, default=0.9, help="sampling temperature for the rhythm model")
    g.add_argument("--density", type=float, help="rhythm model: target objects per measure (conditioning)")
    g.add_argument("--density-bias", type=float, default=0.0,
                   help="rhythm model: extra logit bias on 'no note' (negative = denser)")
    g.add_argument("--decode-steps", type=int, default=12, help="rhythm model: number of decoding rounds")
    g.add_argument("--coord-steps", type=int, default=100, help="coordinate model: diffusion steps (fewer = faster)")
    g.add_argument("--cfg-scale", type=float, default=1.0, help="coordinate model: classifier-free guidance scale")
    g.add_argument("--star", type=float, help="star rating to condition the models on (default per difficulty)")
    c = p.add_argument_group("experimental difficulty and highlight controls")
    c.add_argument("--control-plan", type=Path, help="JSON generation control plan")
    c.add_argument("--target-star", type=float, help="measured star target; up to three candidates, not a guarantee")
    c.add_argument("--candidates", type=int, choices=[1, 2, 3], help="maximum candidates; skill preferences need at least two")
    c.add_argument("--skill-preference", choices=["balanced", "jumps", "streams"],
                   help="experimental pattern preference at a similar measured difficulty")
    c.add_argument("--spacing-scale", type=float, help="optional incoming-head distance scale, 0.5..1.5; adds a coordinate pass")
    c.add_argument("--highlight-mode", choices=["legacy", "off", "manual", "auto"])
    c.add_argument("--highlight", action="append", default=[], metavar="START:END[:STRENGTH]",
                   help="manual region in original audio seconds; repeat for multiple regions")
    c.add_argument("--highlight-sv", action="store_true", default=None, help="also apply the preset's kiai slider velocity")
    p.add_argument("--debug-plot", action="store_true", help="save a PNG showing onsets, grid and chosen notes")
    p.add_argument("--preview", action="store_true",
                   help="also write an mp3 per difficulty with click sounds on every object, to check by ear")
    p.add_argument("--dump-events", action="store_true", help="print every generated event")
    p.add_argument("--gui", action="store_true", help="open the graphical interface")
    return p


def generation_controls(args):
    from .controls import load_control_file, validate_controls
    values = load_control_file(args.control_plan) if args.control_plan else {}
    for key, value in (("target_stars", args.target_star), ("candidates", args.candidates),
                       ("spacing_scale", args.spacing_scale), ("highlight_mode", args.highlight_mode),
                       ("highlight_sv", args.highlight_sv), ("skill_preference", args.skill_preference)):
        if value is not None:
            values[key] = value
    if args.highlight:
        regions = []
        for token in args.highlight:
            fields = token.split(":")
            if len(fields) not in (2, 3):
                raise ValueError("Use --highlight START:END[:STRENGTH], in audio seconds")
            regions.append(dict(start_s=float(fields[0]), end_s=float(fields[1]),
                                strength=float(fields[2]) if len(fields) == 3 else 1.))
        values["highlights"] = regions
        if args.highlight_mode is None:
            values["highlight_mode"] = "manual"
    return validate_controls(values)


def resolve_models(args) -> tuple:
    """Pick the model files from the arguments / models folder. Returns (rhythm, coord) paths or None."""
    if args.rules:
        return None, None

    def pick(name: str, explicit):
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise SystemExit(f"error: model file {path} not found")
            return str(path)
        found = find_model(name)
        if found:
            return str(found)
        if args.download:
            print(f"downloading {MODELS[name].filename} ...")
            return str(ensure_model(name, lambda f, m: print(f"\r  {m}", end="", flush=True)))
        return None

    rhythm = pick("rhythm", args.rhythm_model)
    coord = None if args.no_coord_model else pick("coord", args.coord_model)
    return rhythm, coord


def main(argv=None) -> int:
    # Windows consoles default to a legacy code page; song titles are often Japanese
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    if args.check_source is not None:
        from .source_check import check_source, format_report
        from .provenance import atomic_json
        try:
            if args.source_report and args.source_report.suffix.lower() != ".json":
                raise ValueError("--source-report must be a .json file")
            report = check_source(args.check_source, recursive=args.recursive, records_dir=args.records_dir)
            print(format_report(report))
            if args.source_report:
                atomic_json(args.source_report, report)
                print(f"Report: {args.source_report}")
            return 2 if any(r["status"] == "error" for r in report["results"]) else 0
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.source_report:
        print("error: --source-report requires --check-source", file=sys.stderr)
        return 2
    if args.setup_runtime:
        from .runtime import install_runtime
        from .devices import describe_cuda
        try:
            runtime = install_runtime(progress=lambda f, m: print(f"{f:.0%} {m}", flush=True))
            print(describe_cuda(runtime.cuda))
            print(f"Managed Python: {runtime.python}")
            return 0
        except Exception as exc:
            print(f"GPU setup failed: {exc}", file=sys.stderr)
            return 1
    if args.check_cuda:
        from .devices import describe_cuda
        from .runtime import detect_runtime

        runtime = detect_runtime()
        print(describe_cuda(runtime.cuda))
        if runtime.python:
            print(f"Managed Python: {runtime.python}")
        elif runtime.managed_error:
            print(f"Managed runtime needs repair: {runtime.managed_error}")
        return 0
    if args.gui or not args.audio:
        from .gui import run_gui

        return run_gui()
    audio = Path(args.audio)
    if not audio.exists():
        print(f"error: {audio} not found", file=sys.stderr)
        return 2
    try:
        controls = generation_controls(args)
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rhythm, coord = resolve_models(args)
    if not args.rules and args.device != "cpu" and not os.environ.get("AUTOOSU_MANAGED_WORKER"):
        from .runtime import active_python, popen, probe_python
        python = active_python()
        if python and python.resolve() != Path(sys.executable).resolve():
            try:
                ready = probe_python(python).available
            except Exception:
                ready = False
            if ready:
                forwarded = list(sys.argv[1:] if argv is None else argv)
                if rhythm:
                    forwarded += ["--rhythm-model", str(Path(rhythm).resolve())]
                if coord:
                    forwarded += ["--coord-model", str(Path(coord).resolve())]
                return popen([python, "-I", "-m", "autoosu.cli", *forwarded],
                             stdout=sys.stdout, stderr=sys.stderr).wait()
    if not args.rules:
        missing = [n for n, p in (("rhythm", rhythm), ("coord", coord)) if p is None and not (n == "coord" and args.no_coord_model)]
        if missing:
            print(f"note: no {' / '.join(missing)} model found in the models folder; add --download to fetch "
                  f"them or --rules for the rule-based generator", file=sys.stderr)

    if audio.is_dir():
        from .batch import generate_batch

        try:
            batch = generate_batch(
                audio, args.difficulty, Path(args.out), recursive=args.recursive, preview=args.preview,
                debug_plot=args.debug_plot, seed=args.seed, bpm=args.bpm, offset_ms=args.offset,
                title=args.title, artist=args.artist, creator=args.creator, osu_shift_ms=args.osu_shift,
                rhythm_model=rhythm, temperature=args.temperature, density=args.density,
                density_bias=args.density_bias, star_rating=args.star, decode_steps=args.decode_steps,
                coord_model=coord, coord_steps=args.coord_steps, cfg_scale=args.cfg_scale, device=args.device,
                records_dir=args.records_dir, controls=controls,
                on_result=_dump_events if args.dump_events else None,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Batch complete: {batch.succeeded} succeeded, {batch.failed} failed ({batch.elapsed_s:.1f} s)")
        print(f"Report: {batch.report}")
        return 1 if batch.failed else 0
    if args.recursive:
        print("error: --recursive requires a folder input", file=sys.stderr)
        return 2

    try:
        res = generate(audio, args.difficulty, args.out, seed=args.seed, bpm=args.bpm, offset_ms=args.offset,
                       title=args.title, artist=args.artist, creator=args.creator, osu_shift_ms=args.osu_shift,
                       rhythm_model=rhythm, temperature=args.temperature, density=args.density,
                       density_bias=args.density_bias, star_rating=args.star, decode_steps=args.decode_steps,
                       coord_model=coord, coord_steps=args.coord_steps, cfg_scale=args.cfg_scale, device=args.device,
                       records_dir=args.records_dir, controls=controls)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dump_events:
        _dump_events(res)

    if args.debug_plot:
        from .debug import plot_debug

        png = Path(args.out) / (audio.stem + "_debug.png")
        plot_debug(res, png)
        print(f"debug plot: {png}")

    if args.preview:
        from .preview import render_preview

        for d in res.diffs:
            out = Path(args.out) / f"{audio.stem} [{d.preset.name}]_preview.mp3"
            print(f"preview: {render_preview(res.audio_file, d.beatmap, out, click_shift_ms=res.osu_shift_ms)}")

    print(f"done in {res.elapsed_s:.1f} s ({res.device}) -> {res.osz}")
    return 0


def _dump_events(res) -> None:
    for d in res.diffs:
        print(f"--- {d.preset.name}")
        for ev in d.events:
            extra = f" -> {ev.end_time}" if ev.kind != "circle" else ""
            print(f"{ev.time:7d}{extra:>10}  {ev.kind:7s} beat {ev.beat:8.2f} str {ev.strength:.2f}"
                  f"{'  NC' if ev.new_combo else ''}")


if __name__ == "__main__":
    sys.exit(main())
