"""Frozen song-grouped difficulty experiments. Songs and outputs stay out of Git.

Prepare both development and reserved songs before generating anything. Reserved
songs are not a human-annotated benchmark or proof of exclusion from v0 training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoosu.provenance import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    from autoosu.ml.osu_parse import parse_osu

    if (args.out / "corpus.json").exists():
        raise ValueError("Corpus already frozen; choose a new --out to prepare another")
    excluded = set()
    if args.exclude_corpus:
        excluded = {r["group"] for r in json.loads(args.exclude_corpus.read_text(encoding="utf-8"))["songs"]}
    candidates = {}
    for shard in sorted(args.shards.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".json") or member.size > 32 * 1024 * 1024:
                    continue
                data = json.load(archive.extractfile(member))
                group = data["audio_hash"]
                if group in excluded or group in candidates or not 60 <= float(data.get("audio_length", 0)) <= 150:
                    continue
                maps = [m for m in data["beatmaps"] if m.get("mode") == 0 and m.get("approved") in (1, 2, 4)
                        and 4 <= float(m.get("difficultyrating") or 0) <= 8]
                maps.sort(key=lambda m: (abs(float(m["difficultyrating"]) - 6), int(m["beatmap_id"])))
                for m in maps:
                    parsed = parse_osu(m["content"])
                    # One red line only: no hidden phase resets or BPM changes.
                    if len(parsed.red_lines) != 1 or parsed.red_lines[0].meter != 4 or parsed.n_objects < 100:
                        continue
                    red = parsed.red_lines[0]
                    bpm = 60000 / red.beat_length
                    if not 100 <= bpm <= 240:
                        continue
                    candidates[group] = dict(group=group, shard=shard.name, member=member.name.replace(".json", ".opus"),
                                             reference=m["content"], beatmap_id=m["beatmap_id"], creator_id=m["creator_id"],
                                             duration_s=float(data["audio_length"]), bpm=bpm, offset_ms=red.time,
                                             tempo_group="fast" if bpm >= 180 else "moderate")
                    break
    chosen = []
    for tempo in ("moderate", "fast"):
        eligible = sorted((r for r in candidates.values() if r["tempo_group"] == tempo),
                          key=lambda r: hashlib.sha256(("q1-v1:" + r["group"]).encode()).hexdigest())
        needed = (args.development + args.heldout) // 2
        if len(eligible) < needed:
            raise ValueError(f"Not enough {tempo} candidates: {len(eligible)} < {needed}")
        for i, row in enumerate(eligible[:needed]):
            row["partition"] = "development" if i < args.development // 2 else "heldout"
            chosen.append(row)
    for row in chosen:
        audio = args.out / "audio" / f"{row['group']}.opus"
        audio.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.shards / row["shard"]) as archive:
            audio.write_bytes(archive.extractfile(row["member"]).read())
        reference = args.out / "reference" / f"{row['group']}.osu"
        reference.parent.mkdir(exist_ok=True)
        reference.write_text(row.pop("reference"), encoding="utf-8")
        row.update(audio=audio.relative_to(args.out).as_posix(), audio_sha256=digest(audio),
                   reference=reference.relative_to(args.out).as_posix(), reference_sha256=digest(reference),
                   human_highlights=None, human_playtest=None)
    atomic_json(args.out / "corpus.json", dict(schema="autoosu.difficulty-corpus/1", songs=chosen,
                exclusions=sorted(excluded), selection="hash order, balanced by reference BPM; constant timing, 4/4, 60-150 s",
                limits="Exact audio hashes only. Alternate recordings and v0 training overlap unverified. Human annotations pending."))
    print(json.dumps(dict(songs=len(chosen), development=args.development, heldout=args.heldout)), flush=True)


def run(args):
    from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
    from autoosu.metrics import CALCULATOR_VERSION

    corpus = json.loads((args.out / "corpus.json").read_text(encoding="utf-8"))
    songs = [s for s in corpus["songs"] if s["partition"] == args.partition]
    settings = dict(seed=240924, decode_steps=12, coord_steps=100, temperature=.9, cfg_scale=1., device=args.device)
    recipe = dict(corpus_sha256=digest(args.out / "corpus.json"), calculator_version=CALCULATOR_VERSION,
                  models={k: digest(args.models / f"{k}_v0.pt") for k in ("rhythm", "coord")},
                  code={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT / "autoosu").rglob("*.py"))},
                  base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  settings=settings, conditions=args.conditions, timing_modes=["reference", "automatic"])
    recipe_path = args.out / f"{args.partition}-recipe.json"
    result_path = args.out / f"{args.partition}-generated.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text(encoding="utf-8")) != recipe:
        raise ValueError("Recipe changed; do not mix comparisons or silently resume different code")
    atomic_json(recipe_path, recipe)
    rows = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else []
    completed = {(r["group"], r["timing_mode"], r["condition"]) for r in rows}
    cache = {}
    total = len(songs) * len(args.conditions) * 2
    for song in songs:
        audio = args.out / song["audio"]
        if digest(audio) != song["audio_sha256"] or digest(args.out / song["reference"]) != song["reference_sha256"]:
            raise ValueError("Frozen input changed")
        for mode in recipe["timing_modes"]:
            for star in args.conditions:
                if (song["group"], mode, star) in completed:
                    continue
                destination = args.out / "generated" / song["group"] / mode / str(star)
                started = time.perf_counter()
                # Generation subtracts 26 ms on export: add it to reference redline input.
                result = generate(audio, ["Insane"], destination, title="Difficulty evaluation", artist=song["group"][:12],
                                  rhythm_model=str(args.models / "rhythm_v0.pt"), coord_model=str(args.models / "coord_v0.pt"),
                                  star_rating=star, bpm=song["bpm"] if mode == "reference" else None,
                                  offset_ms=song["offset_ms"] + OSU_TIMING_SHIFT_MS if mode == "reference" else None,
                                  model_cache=cache, records_dir=args.out / "records", log=lambda *_: None, **settings)
                with zipfile.ZipFile(result.osz) as z:
                    raw = z.read(next(n for n in z.namelist() if n.endswith(".osu")))
                path = destination / "map.osu"
                path.write_bytes(raw)
                diff = result.diffs[0]
                rows.append(dict(group=song["group"], partition=song["partition"], timing_mode=mode,
                                 tempo_group=song["tempo_group"], condition=star, bpm=result.timing.bpm,
                                 offset_ms=result.timing.offset_ms - OSU_TIMING_SHIFT_MS, summary=diff.summary(),
                                 diagnostics=diff.diagnostics, measurement=diff.measurement,
                                 path=path.relative_to(args.out).as_posix(), sha256=digest(path),
                                 elapsed_s=time.perf_counter() - started))
                atomic_json(result_path, rows)
                print(f"{len(rows)}/{total} {song['group'][:8]} {mode} {star:g} -> {diff.measurement['stars']} stars", flush=True)


def report(args):
    import numpy as np
    import rosu_pp_py as rosu
    from importlib.metadata import version
    from autoosu.metrics import CALCULATOR_VERSION

    if version("rosu-pp-py") != CALCULATOR_VERSION:
        raise ValueError("Use the pinned calculator")
    rows = json.loads((args.out / f"{args.partition}-generated.json").read_text(encoding="utf-8"))
    recipe = json.loads((args.out / f"{args.partition}-recipe.json").read_text(encoding="utf-8"))
    corpus = json.loads((args.out / "corpus.json").read_text(encoding="utf-8"))
    expected = sum(s["partition"] == args.partition for s in corpus["songs"]) * len(recipe["conditions"]) * 2
    if len(rows) != expected:
        raise ValueError(f"Incomplete experiment: {len(rows)}/{expected}")
    for row in rows:
        path = args.out / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Output file changed")
        bm = rosu.Beatmap(path=str(path))
        if bm.is_suspicious():
            raise ValueError("Suspicious output")
        actual = rosu.Difficulty(mods=0, lazer=False).calculate(bm)
        if row["measurement"]["status"] != "ok" or abs(actual.stars - row["measurement"]["stars"]) > 1e-9:
            raise ValueError("Independent parse/measurement mismatch")
    aggregates = []
    for mode in recipe["timing_modes"]:
        for star in recipe["conditions"]:
            subset = [r for r in rows if r["timing_mode"] == mode and r["condition"] == star]
            stars = [r["measurement"]["stars"] for r in subset]
            aggregates.append(dict(timing_mode=mode, condition=star, n=len(subset),
                                   median_stars=float(np.median(stars)), min_stars=min(stars), max_stars=max(stars),
                                   within_half_star=sum(abs(s - star) <= .5 for s in stars),
                                   median_nps=float(np.median([r["summary"]["nps"] for r in subset])),
                                   median_aim=float(np.median([r["measurement"]["aim"] for r in subset])),
                                   median_speed=float(np.median([r["measurement"]["speed"] for r in subset]))))
    invalid = [r for r in rows if any(r["diagnostics"][key] for key in (
        "simultaneous_or_reversed_starts", "overlapping_objects", "heads_outside_playfield", "invalid_sliders"))]
    anchors_flagged = sum(r["diagnostics"]["anchors_outside_playfield"] > 0 for r in rows)
    compact_rows = [{k: v for k, v in r.items() if k not in ("measurement", "path", "summary")}
                    | dict(stars=r["measurement"]["stars"], aim=r["measurement"]["aim"], speed=r["measurement"]["speed"],
                           peaks=r["measurement"]["peaks"], nps=r["summary"]["nps"]) for r in rows]
    result = dict(schema="autoosu.difficulty-screen/1", partition=args.partition, recipe=recipe,
                  maps=len(rows), independent_parse_matches=len(rows), structural_error_maps=len(invalid),
                  anchor_flagged_maps=anchors_flagged,
                  elapsed_s=sum(r["elapsed_s"] for r in rows), aggregates=aggregates, rows=compact_rows,
                  limits=corpus["limits"] + " No client import/playtest; structural diagnostics do not prove playability.")
    atomic_json(args.out / f"{args.partition}-report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("recipe", "rows")}, indent=2), flush=True)


def curves(args):
    """Inspect informational anchor flags using a second parser (optional slider package)."""
    from importlib.metadata import version
    import numpy as np
    from slider import Beatmap
    from slider.beatmap import Slider

    rows = json.loads((args.out / f"{args.partition}-generated.json").read_text(encoding="utf-8"))
    reports = []
    for row in rows:
        if not row["diagnostics"]["anchors_outside_playfield"]:
            continue
        path = args.out / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Output changed")
        bm = Beatmap.parse(path.read_text(encoding="utf-8-sig"))
        sliders = [o for o in bm.hit_objects(stacking=False) if isinstance(o, Slider)
                   and any(not (0 <= p.x <= 512 and 0 <= p.y <= 384) for p in o.curve.points)]
        bad = []
        for sl in sliders:
            xy = np.asarray([tuple(sl.curve(float(t))) for t in np.linspace(0, 1, 1001)])
            if np.any(xy < [-.01, -.01]) or np.any(xy > [512.01, 384.01]):
                bad.append(dict(time_ms=sl.time.total_seconds() * 1000,
                                minimum=xy.min(0).tolist(), maximum=xy.max(0).tolist()))
        reports.append(dict(group=row["group"], condition=row["condition"], timing_mode=row["timing_mode"],
                            sha256=row["sha256"], sliders=len(sliders), sampled_paths_outside=bad))
    result = dict(parser="slider", version=version("slider"), samples_per_curve=1001,
                  stacking=False, tolerance_px=.01, maps=len(reports), sliders=sum(r["sliders"] for r in reports),
                  paths_outside=sum(len(r["sampled_paths_outside"]) for r in reports), rows=reports,
                  limits="Only sliders flagged for out-of-field anchors. Sampled center paths, not exact extrema or client playtesting.")
    atomic_json(args.out / f"{args.partition}-curves.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "report", "curves"])
    parser.add_argument("--out", type=Path, default=ROOT / "out/difficulty-evaluation")
    parser.add_argument("--shards", type=Path, default=ROOT / "data/hf/compressed")
    parser.add_argument("--exclude-corpus", type=Path)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--partition", choices=["development", "heldout"], default="development")
    parser.add_argument("--development", type=int, default=8)
    parser.add_argument("--heldout", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", nargs="+", type=float, default=[5.5, 6., 6.5, 7., 7.5, 8.])
    args = parser.parse_args()
    if any(n < 2 or n % 2 for n in (args.development, args.heldout)):
        parser.error("Song counts must be positive even numbers")
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
