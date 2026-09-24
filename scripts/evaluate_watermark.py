"""Frozen watermark evaluation. Raw maps remain in ignored out/, never in Git.

prepare freezes the codec and negatives before run observes any match scores.
Historical maps are only known-unmarked controls, never AI/human labels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoosu.beatmap import Beatmap, Circle, TimingPoint
from autoosu.provenance import atomic_json, engine_info, parse_sections
from autoosu.watermark import MAX_TICKS, MIN_TICKS, SCHEMA, THRESHOLD_PERCENT, detect_watermark, embed_watermark


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def codec():
    return dict(schema=SCHEMA, min_ticks=MIN_TICKS, max_ticks=MAX_TICKS, threshold_percent=THRESHOLD_PERCENT,
                source_sha256=sha((ROOT / "autoosu/watermark.py").read_bytes()))


def prepare(args):
    if (args.out / "recipe.json").exists():
        raise ValueError("Recipe exists; use a new output directory")
    excluded = {s["group"] for s in read(args.difficulty / "corpus.json")["songs"]}
    old = read(args.baseline / "corpus.json")
    # Exclude all earlier classifier references and generation songs, not just held-out songs.
    excluded.update(r["group"] for key in ("references", "songs") for r in old[key])
    args.out.mkdir(parents=True, exist_ok=True)
    frozen_codec = codec()
    rows, seen = [], set(excluded)
    for shard in sorted(args.shards.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".json") or member.size > 32 * 1024 * 1024:
                    continue
                data = json.load(archive.extractfile(member))
                group = data["audio_hash"]
                if group in seen:
                    continue
                candidates = [m for m in data["beatmaps"] if m.get("mode") == 0
                              and "1900" < str(m.get("last_update") or "")[:4] <= "2025"]
                if not candidates:
                    continue
                m = min(candidates, key=lambda m: sha(("watermark-v1:" + str(m["beatmap_id"])).encode()))
                raw = m["content"].encode("utf-8")
                path = args.out / "negative" / f"{group}.osu"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(raw)
                seen.add(group)
                rows.append(dict(group=group, path=path.relative_to(args.out).as_posix(), sha256=sha(raw),
                                 shard=shard.name, member=member.name, beatmap_id=m["beatmap_id"]))
        print(f"prepared {shard.name}: {len(rows)} independent unmarked songs", flush=True)
    atomic_json(args.out / "recipe.json", dict(codec=frozen_codec, negatives=rows, excluded_groups=sorted(excluded),
                selection="One pre-2026 standard map per audio hash by beatmap-ID hash. All 8 local shards; no watermark score used.",
                negative_label="known unmarked before this codec existed; no human-authorship assertion",
                positive_recipe=read(args.baseline / "recipe.json"),
                positive_manifest_sha256=sha((args.baseline / "generated.json").read_bytes()),
                base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))


def mark_text(raw, family):
    """Apply the production embedder to circle heads without rewriting other fields."""
    _, sections = parse_sections(raw)
    difficulty = dict(line.split(":", 1) for line in sections["Difficulty"])
    red = [line.split(",") for line in sections["TimingPoints"] if line.split(",")[6] == "1"]
    circles = []
    for line in sections["HitObjects"]:
        fields = line.split(",")
        if int(fields[3]) & 1:
            circles.append(Circle(int(fields[0]), int(fields[1]), int(fields[2])))
    bm = Beatmap("audio.mp3", "", "", "", cs=float(difficulty["CircleSize"]), hit_objects=circles,
                 timing_points=[TimingPoint(int(r[0]), float(r[1])) for r in red])
    eng = engine_info({"engine": "model" if family in ("rhythm", "model") else "rules"},
                      {"engine": "model" if family in ("coord", "model") else "rules"})
    embedding = embed_watermark(bm, eng)
    lines, section, index = [], None, 0
    for line in raw.decode("utf-8-sig").splitlines():
        if line.startswith("["):
            section = line
        if section == "[HitObjects]" and line and not line.startswith("[") and not line.startswith("//"):
            fields = line.split(",")
            if int(fields[3]) & 1:
                obj = circles[index]
                fields[0:2] = [str(obj.x), str(obj.y)]
                line = ",".join(fields)
                index += 1
        lines.append(line)
    return ("\r\n".join(lines) + "\r\n").encode(), embedding


def edit(raw, kind):
    lines, section, circle = [], "", 0
    for line in raw.decode("utf-8-sig").splitlines():
        if line.startswith("["):
            section = line
        if section == "[Metadata]" and ":" in line:
            line = line.split(":", 1)[0] + ":"
        if section == "[HitObjects]" and "," in line and not line.startswith("//"):
            f = line.split(",")
            if int(f[3]) & 1:
                if kind == "edit10" and circle % 10 == 0:
                    f[0] = str(max(0, min(512, int(f[0]) + 7)))
                    f[1] = str(max(0, min(384, int(f[1]) - 5)))
                    f[2] = str(int(f[2]) + 3)
                if kind == "delete10" and circle % 10 == 0:
                    circle += 1
                    continue
                circle += 1
                line = ",".join(f)
        lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8-sig")


def run(args):
    from autoosu.metrics import measure_difficulty

    recipe = read(args.out / "recipe.json")
    if recipe["codec"] != codec():
        raise ValueError("Codec changed after freezing; do not tune on this test set")
    if recipe["positive_manifest_sha256"] != sha((args.baseline / "generated.json").read_bytes()):
        raise ValueError("Positive source manifest changed")
    negative = []
    for row in recipe["negatives"]:
        raw = (args.out / row["path"]).read_bytes()
        if sha(raw) != row["sha256"]:
            raise ValueError("Frozen negative changed")
        negative.append(dict(group=row["group"], **detect_watermark(raw)))
    print("negative results:", dict(collections.Counter(r["status"] for r in negative)), flush=True)
    positives = []
    for i, row in enumerate(read(args.baseline / "generated.json")):
        raw = (args.baseline / row["path"]).read_bytes()
        if sha(raw) != row["sha256"]:
            raise ValueError("Frozen baseline map changed")
        family = dict(both_models="model", rhythm_only="rhythm", coord_only="coord", rules="rules")[row["label"]]
        marked, embedding = mark_text(raw, family)
        before, after = measure_difficulty(raw.decode()), measure_difficulty(marked.decode())
        if before["status"] != "ok" or after["status"] != "ok":
            raise ValueError("Measured difficulty unavailable")
        _, original_sections = parse_sections(raw)
        _, marked_sections = parse_sections(marked)
        assert all(original_sections[k] == marked_sections[k] for k in original_sections if k != "HitObjects")
        assert len(original_sections["HitObjects"]) == len(marked_sections["HitObjects"])
        for a, b in zip(original_sections["HitObjects"], marked_sections["HitObjects"]):
            af, bf = a.split(","), b.split(",")
            if int(af[3]) & 1:
                assert af[2:] == bf[2:]
                assert abs(int(af[0])-int(bf[0])) <= 1 and abs(int(af[1])-int(bf[1])) <= 1
            else:
                assert a == b
        path = args.out / "marked" / row["group"] / (family + ".osu")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(marked)
        results = {kind: detect_watermark(edit(marked, kind)) for kind in ("metadata", "edit10", "delete10")}
        results["original"] = detect_watermark(marked)
        positives.append(dict(group=row["group"], family=family, raw_sha256=row["sha256"],
                              marked_sha256=sha(marked), embedding=embedding, unmarked=detect_watermark(raw), checks=results,
                              before_stars=before["stars"], after_stars=after["stars"],
                              star_delta=after["stars"] - before["stars"]))
        if (i + 1) % 20 == 0:
            print(f"positive comparisons: {i + 1}", flush=True)
    eligible = [r for r in negative if r["status"] in ("detected", "not_detected", "ambiguous")]
    false_positive = sum(r["status"] == "detected" for r in eligible)
    summary = dict(negative_total=len(negative), negative_eligible=len(eligible), false_positives=false_positive,
                   negative_status=dict(collections.Counter(r["status"] for r in negative)),
                   zero_fp_one_sided_95_upper=1-math.pow(.05, 1/len(eligible)) if eligible and not false_positive else None,
                   positive_total=len(positives), positive_songs=len({r["group"] for r in positives}),
                   embedding_status=dict(collections.Counter(r["embedding"]["status"] for r in positives)),
                   checks={kind: dict(collections.Counter(r["checks"][kind]["status"] for r in positives))
                           for kind in ("original", "metadata", "edit10", "delete10")},
                   wrong_family=sum(c.get("engine_claim", r["family"]) != r["family"] for r in positives for c in r["checks"].values()),
                   max_abs_star_delta=max(abs(r["star_delta"]) for r in positives),
                   mean_abs_star_delta=sum(abs(r["star_delta"]) for r in positives)/len(positives),
                   unmarked_baseline_status=dict(collections.Counter(r["unmarked"]["status"] for r in positives)))
    atomic_json(args.out / "report.json", dict(schema="autoosu.watermark-evaluation/1", codec=recipe["codec"],
                recipe_sha256=sha((args.out / "recipe.json").read_bytes()), summary=summary,
                negatives=negative, positives=positives))
    print(json.dumps(summary, indent=2), flush=True)


def roundtrip(args):
    """Optional independent parser test; this is not an osu! editor round-trip."""
    from importlib.metadata import version
    from slider import Beatmap as ParsedBeatmap

    if read(args.out / "recipe.json")["codec"] != codec():
        raise ValueError("Frozen codec changed")
    rows = []
    for path in sorted((args.out / "marked").rglob("*.osu")):
        original = path.read_bytes()
        rewritten = ParsedBeatmap.parse(original.decode("utf-8-sig")).pack().encode("utf-8")
        row = dict(path=path.relative_to(args.out).as_posix(), input_sha256=sha(original),
                   resaved_sha256=sha(rewritten), watermark=detect_watermark(rewritten))
        destination = args.out / "resaved" / path.relative_to(args.out / "marked")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rewritten)
        rows.append(row)
    summary = dict(maps=len(rows), results=dict(collections.Counter(r["watermark"]["status"] for r in rows)))
    atomic_json(args.out / "roundtrip.json", dict(parser="slider", version=version("slider"), summary=summary, rows=rows,
                limit="Independent parser serialization only; not an osu! client import, editor save, or playtest."))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "roundtrip"])
    parser.add_argument("--out", type=Path, default=Path("out/watermark-evaluation"))
    parser.add_argument("--baseline", type=Path, default=Path("out/d1-v0-probe"))
    parser.add_argument("--difficulty", type=Path, default=Path("out/difficulty-evaluation"))
    parser.add_argument("--shards", type=Path, default=Path("data/hf/compressed"))
    args = parser.parse_args()
    globals()[args.action](args)
