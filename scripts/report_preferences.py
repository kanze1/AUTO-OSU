"""Verify frozen preference outputs and write a compact, song-free report.

First run --curves with an independent environment containing slider 0.8.2;
then run without that flag in the app environment for source-record checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(directories, output):
    from autoosu.provenance import atomic_json
    from autoosu.source_check import check_source

    reports = []
    structural = ("overlapping_objects", "heads_outside_playfield", "invalid_sliders", "simultaneous_or_reversed_starts")
    for directory in directories:
        recipe = read(directory / "recipe.json")
        rows = read(directory / "results.json")
        assert len(rows) == len(recipe["groups"]) * len(recipe["modes"])
        assert {(r["group"], r["mode"]) for r in rows} == {
            (group, mode) for group in recipe["groups"] for mode in recipe["modes"]}
        parser = read(directory / "heldout-curves.json")
        assert parser["maps"] == len(rows) and parser["paths_outside"] == 0
        assert {(r["group"], r["variant"], r["sha256"]) for r in parser["rows"]} == {
            (r["group"], r["mode"], r["sha256"]) for r in rows}
        compact = []
        for row in rows:
            assert sha(directory / row["path"]) == row["sha256"]
            control = row["summary"]["controls"]
            sources = check_source(directory / row["osz"], records_dir=directory / "records")["results"]
            assert sources and all(r["status"] == "local_record_match" for r in sources)
            compact.append(dict(group=row["group"], mode=row["mode"], sha256=row["sha256"],
                measured_stars=row["measurement"]["stars"], patterns=row["patterns"],
                selection_status=control["selection_status"], preference=control.get("preference"),
                selected=control["selected"], diagnostics=row["diagnostics"],
                candidates=[{k:c.get(k) for k in ("star_condition", "rhythm_star_condition", "measured", "patterns",
                    "density_source", "spacing_scale", "rejection_reasons", "raw_sha256")} for c in control["candidates"]],
                watermark=row["summary"].get("watermark"), source_status=[r["status"] for r in sources]))
        summary = {}
        for mode in recipe["modes"]:
            selected = [r for r in rows if r["mode"] == mode]
            summary[mode] = dict(maps=len(selected), observed=sum(
                r["summary"]["controls"].get("preference", {}).get("status") == "observed" for r in selected),
                structural_errors=sum(any(r["diagnostics"][k] for k in structural) for r in selected),
                unverified_fallbacks=sum(r["summary"]["controls"]["selection_status"] != "accepted" for r in selected),
                median_elapsed_s=statistics.median(r["elapsed_s"] for r in selected),
                star_range=[min(r["measurement"]["stars"] for r in selected), max(r["measurement"]["stars"] for r in selected)])
        reports.append(dict(recipe=recipe, results_sha256=sha(directory / "results.json"),
            summary=summary, independent_parser={k:v for k,v in parser.items() if k != "rows"}, rows=compact))
        print(directory.name, json.dumps(summary), flush=True)
    atomic_json(output, dict(schema="autoosu.preference-evaluation/1", evaluations=reports,
        limits=["Small policy-held-out screen, not training-held-out evidence or a human skill verdict.",
                "Reference and automatic timing share the same songs; do not count them as independent song groups.",
                "No explicit star target was used in this screen; high-star joint targeting remains experimental.",
                "Independent parser checks do not establish manual gameplay or editor-resave acceptance."]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--curves", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/skill-preferences-evaluation.json")
    args = parser.parse_args()
    if args.curves:
        from evaluate_controls import curves
        for directory in args.directories:
            rows = [dict(row, variant=row["mode"]) for row in read(directory / "results.json")]
            (directory / "heldout-results.json").write_text(json.dumps(rows), encoding="utf-8")
            curves(SimpleNamespace(out=directory, partition="heldout"))
    else:
        report(args.directories, args.output)
