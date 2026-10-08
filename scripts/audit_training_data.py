"""Inventory M1 inputs without treating free-text tags as reviewed skill labels.

Reads existing uncompressed HF shards and the prepared index; writes only to a new
output directory. No model training or changes to the source corpus are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import tarfile


SKILLS = ("jumps", "bursts", "streams", "stamina", "slider_tech", "reading")
TAG_TERMS = frozenset((*SKILLS, "jump", "burst", "stream", "tech"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def legacy_split(set_id: int) -> str:
    # Matches autoosu.ml.dataset.load_index's default validation fraction.
    return "validation" if (set_id * 2654435761) % 1000 / 1000.0 < 0.03 else "train"


def tag_candidates(tags: str) -> list[str]:
    """Literal search aids for human review, never training targets."""
    words = set(re.findall(r"[\w/]+", tags.casefold()))
    return sorted(w for w in words if w in TAG_TERMS or w.startswith("skillset/"))


def cross_split_groups(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row)
    mixed = [rs for rs in groups.values() if len({r["legacy_split"] for r in rs}) > 1]
    return {"groups": len(groups), "mixed_groups": len(mixed),
            "maps_in_mixed_groups": sum(map(len, mixed)),
            "validation_maps_in_mixed_groups": sum(
                r["legacy_split"] == "validation" for rs in mixed for r in rs)}


def audit(shards: Path, index: Path, out: Path) -> dict:
    paths = sorted(shards.glob("*.tar"))
    if not paths:
        raise ValueError(f"No tar shards in {shards}")
    index_bytes = index.read_bytes()
    prepared = [json.loads(line) for line in index_bytes.splitlines() if line.strip()]
    keys = [(r["beatmap_id"], r["track"]) for r in prepared]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate beatmap/track keys in prepared index")
    lookup = dict(zip(keys, prepared))
    # Refuse to mix evidence from different runs or overwrite review annotations.
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    found = set()
    sources = []
    map_fields = Counter()
    meta_fields = Counter()
    skill_terms = Counter()
    tag_presence = Counter()
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for shard in paths:
            meta_digest = hashlib.sha256()
            n_metadata = n_maps = 0
            # Local uncompressed tar permits seeking past audio without decoding it.
            with tarfile.open(shard, "r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".json"):
                        continue
                    with archive.extractfile(member) as handle:
                        raw = handle.read()
                    meta_digest.update(member.name.encode() + b"\0" + raw + b"\0")
                    meta = json.loads(raw)
                    n_metadata += 1
                    meta_fields.update(meta.keys())
                    track = f"{shard.stem}_{member.name.rsplit('.', 1)[0]}"
                    for beatmap in meta["beatmaps"]:
                        key = (int(beatmap["beatmap_id"]), track)
                        if key not in lookup:
                            continue
                        if key in found:
                            raise ValueError(f"Duplicate source record: {key}")
                        found.add(key)
                        map_fields.update(beatmap.keys())
                        tags = beatmap.get("tags")
                        if tags is not None and not isinstance(tags, str):
                            raise ValueError(f"Non-text tags in source record: {key}")
                        tag_presence["nonempty" if tags else "missing_or_empty"] += 1
                        terms = tag_candidates(tags or "")
                        skill_terms.update(terms)
                        prep = lookup[key]
                        row = {**prep, "shard": shard.name, "metadata_member": member.name,
                               "source_audio_hash": meta.get("audio_hash"),
                               "beatmap_content_sha256": sha256(beatmap["content"].encode()),
                               "source_tags": tags, "candidate_terms": terms,
                               "legacy_split": legacy_split(prep["beatmapset_id"]),
                               "skill_labels": {skill: None for skill in SKILLS},
                               "label_status": "unreviewed", "label_source": None}
                        manifest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        rows.append(row)
                        n_maps += 1
            sources.append({"name": shard.name, "bytes": shard.stat().st_size,
                            "metadata_sha256": meta_digest.hexdigest(),
                            "metadata_records": n_metadata, "matched_maps": n_maps})
            print(f"{shard.name}: {n_maps} matched; {len(rows)}/{len(prepared)} total", flush=True)
    missing = sorted(set(keys) - found)
    (out / "missing.json").write_text(json.dumps(missing), encoding="utf-8")
    # Deterministic stratified queue. Unknown labels remain null until human review.
    buckets = defaultdict(list)
    for row in rows:
        band = str(int(row["sr"]))
        buckets[(band, bool(row["candidate_terms"]))].append(row)
    review = []
    selected_audio = set()
    for bucket in sorted(buckets):
        count = 0
        for row in sorted(buckets[bucket], key=lambda r: sha256(
                f"m1-review-v1:{r['beatmap_id']}:{r['track']}".encode())):
            audio = row["source_audio_hash"] or row["track"]
            if audio in selected_audio:
                continue
            selected_audio.add(audio)
            review.append({**row, "reviewer": None, "review_notes": None,
                           "review_bucket": {"star_floor": bucket[0], "tag_candidate": bucket[1]}})
            count += 1
            if count == 5:
                break
    (out / "review_queue.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in review), encoding="utf-8")
    legacy_rows = [r for r in rows if r["n_objects"] >= 50]
    report = {
        "schema_version": 1, "status": "inventory_complete" if not missing else "incomplete_source_join",
        "training_started": False, "training_ready": False,
        "source_dataset": "project-riz/osu-beatmaps (existing local snapshot)",
        "index_sha256": sha256(index_bytes), "script_sha256": sha256(Path(__file__).read_bytes()),
        "shards": sources, "prepared_maps": len(prepared), "matched_maps": len(rows),
        "missing_source_records": len(missing), "prepared_tracks": len({r["track"] for r in prepared}),
        "prepared_sets": len({r["beatmapset_id"] for r in prepared}),
        "star_floor_counts": dict(sorted(Counter(int(r["sr"]) for r in prepared).items())),
        "source_metadata_field_counts": dict(meta_fields), "matched_map_field_counts": dict(map_fields),
        "free_text_tag_presence": dict(tag_presence), "literal_skill_term_counts": dict(skill_terms),
        "maps_with_literal_skill_terms": sum(bool(r["candidate_terms"]) for r in rows),
        "reviewed_skill_label_maps": 0, "review_queue_maps": len(review),
        "legacy_eligible_maps": len(legacy_rows),
        "legacy_track_split": cross_split_groups(legacy_rows, "track"),
        "legacy_source_audio_hash_split": cross_split_groups(legacy_rows, "source_audio_hash"),
        "evidence_limits": [
            "Audio hashes are source declarations; audio bytes and near duplicates were not verified.",
            "Metadata digests cover JSON member names and bytes, not complete tar archives.",
            "Free-text tags and literal skill terms are not reviewed per-difficulty skill labels.",
            "No independent test split was frozen; prior evaluation songs still need exclusion.",
            "Snapshot usage conditions and per-asset provenance need review before training.",
            "Human label agreement and conditioned model inputs remain unimplemented.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {
        "shards", "matched_map_field_counts", "source_metadata_field_counts"}}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.shards, args.index, args.out)
    if report["missing_source_records"]:
        raise SystemExit("Source join incomplete; see missing.json")


if __name__ == "__main__":
    main()
