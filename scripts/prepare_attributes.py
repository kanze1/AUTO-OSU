"""Recover source object attributes with train-only vocabularies and explicit exclusions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoosu.ml.attributes import ATTRIBUTE_NAMES, align_attributes, source_attributes
from autoosu.ml.dataset import read_splits


def source_maps(rows, shards):
    lookup = {r["beatmap_id"]: r for r in rows}
    for shard in sorted({r["shard"] for r in rows}):
        members = {r["metadata_member"] for r in rows if r["shard"] == shard}
        with tarfile.open(shards / shard) as archive:
            for member in archive:
                if member.name not in members:
                    continue
                for bm in json.load(archive.extractfile(member))["beatmaps"]:
                    bid = int(bm["beatmap_id"])
                    if bid not in lookup:
                        continue
                    row = lookup[bid]
                    if hashlib.sha256(bm["content"].encode()).hexdigest() != row["beatmap_content_sha256"]:
                        raise ValueError(f"Source hash mismatch: {bid}")
                    yield row, bm["content"]
        print(json.dumps(dict(shard=shard, stage="source_scan")), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("data", "prep", "shards", "out"):
        ap.add_argument("--" + arg, type=Path, required=True)
    args = ap.parse_args()
    splits = read_splits(args.data)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "maps").mkdir()
    vocabulary = {"new_combo": [0, 1], "hitsound": list(range(8)), "tail_hitsound": list(range(8))}
    counts = {name: Counter() for name in ATTRIBUTE_NAMES}
    excluded = []
    # First pass freezes the vocabulary from training sources only.
    for row, content in source_maps(splits["train"], args.shards):
        try:
            records = source_attributes(content)
        except (ValueError, IndexError) as error:
            excluded.append(dict(beatmap_id=row["beatmap_id"], reason=str(error), stage="vocabulary"))
            continue
        for values in records.values():
            for name, value in values.items():
                counts[name][value] += 1
    for name in ("repeats", "topology"):
        vocabulary[name] = sorted(counts[name])
    (args.out / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2))
    stats = {}
    # Unknown attributes retain -100; every rhythm map remains in its original split.
    for split, rows in splits.items():
        unknown, covered = Counter(), Counter()
        processed = set()
        for row, content in source_maps(rows, args.shards):
            bid = row["beatmap_id"]
            objects = np.load(args.prep / "maps" / f"{bid}.npz")["objects"]
            try:
                records = source_attributes(content)
            except (ValueError, IndexError) as error:
                excluded.append(dict(beatmap_id=bid, reason=str(error), stage=split))
                records = {}
            targets, missing = align_attributes(objects, records, vocabulary)
            np.savez_compressed(args.out / "maps" / f"{bid}.npz", attributes=targets)
            unknown.update(missing)
            covered.update({name: int((targets[:, i] != -100).sum()) for i, name in enumerate(ATTRIBUTE_NAMES)})
            processed.add(bid)
        if processed != {r["beatmap_id"] for r in rows}:
            raise ValueError(f"Incomplete attribute extraction: {split}")
        stats[split] = dict(maps=len(processed), covered=dict(covered), unknown=dict(unknown))
    report = dict(schema="autoosu.object-attributes/1", vocabulary_source="train only", splits=stats,
        vocabulary_sha256=hashlib.sha256((args.out / "vocabulary.json").read_bytes()).hexdigest(),
        classes={k: len(v) for k, v in vocabulary.items()}, exclusions=excluded,
        limits="Quarter-beat rhythm grid retained. Standard head/tail sound flags only; sample banks, custom audio and internal repeat sounds are not learned by this version.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "exclusions"}), flush=True)


if __name__ == "__main__":
    main()
