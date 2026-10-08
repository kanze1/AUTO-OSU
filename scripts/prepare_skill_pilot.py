"""Join pinned community votes, verify map versions, cache coordinate tokens and group songs.

Experimental fine-tuning split only: v0 pretraining exposure is not undone by this split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import re
import sys
import tarfile
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coord"))
from autoosu.ml.skills import SKILL_NAMES, SKILL_TAGS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def song_key(meta):
    def normalize(value):
        return re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", value).casefold())
    return normalize(meta["artist"]), normalize(meta["title"])


def osu_file_md5(content):
    """HF text uses LF; osu API file hashes refer to the original CRLF serialization."""
    return hashlib.md5(content.replace("\r\n", "\n").replace("\n", "\r\n").encode()).hexdigest()


def cache_map(job):
    import numpy as np
    import torch
    from slider import Beatmap
    from osu_diffusion.utils.data_loading import beatmap_to_sequence
    torch.set_num_threads(1)
    row, content, expected_md5, out = job
    actual_md5 = osu_file_md5(content)
    if actual_md5 != expected_md5:
        return None, {"beatmap_id": row["beatmap_id"], "reason": "metadata_version_md5_mismatch",
                      "expected": expected_md5, "actual": actual_md5}
    beatmap = Beatmap.parse(content)
    seq = beatmap_to_sequence(beatmap).numpy()
    if seq.shape[1] < 128 or not np.isfinite(seq).all():
        return None, {"beatmap_id": row["beatmap_id"], "reason": "short_or_nonfinite_sequence"}
    target = Path(out) / "coords" / f"{row['beatmap_id']}.npy"
    np.save(target, seq)
    return {**row, "coord_path": str(target.resolve()), "file_md5": actual_md5}, None


def freeze_groups(rows, audio_hashes):
    """Union exact audio bytes, source audio identities, beatmapsets and artist/title."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for row in rows:
        track = "track:" + row["track"]
        union(track, "audio:" + audio_hashes[row["track"]])
        union(track, "declared:" + row["source_audio_hash"])
        union(track, "set:" + str(row["beatmapset_id"]))
        union(track, "song:" + json.dumps(row["song_key"]))
    for row in rows:
        group = find("track:" + row["track"])
        row["song_group"] = digest(group.encode())
        row["audio_bytes_sha256"] = audio_hashes[row["track"]]
        bucket = int(digest(("m1-pilot-20261008:" + group).encode())[:8], 16) % 100
        row["split"] = "test" if bucket < 10 else "validation" if bucket < 20 else "train"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--tags", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    meta = {int(r["beatmap_id"]): r for r in csv.DictReader((args.tags / "metadata.csv").open())}
    votes = {}
    # Upstream calls its first column beatmapset_id, but it contains beatmap IDs.
    # Join through the explicit metadata beatmap_id and verify file MD5 below.
    for row in csv.DictReader((args.tags / "tags.csv").open()):
        bid = int(row["beatmapset_id"])
        selected = {k: 1 for k, field in SKILL_TAGS.items() if float(row[field] or 0) >= args.min_votes}
        if selected:
            votes[bid] = (selected, {k: int(float(row[field] or 0)) for k, field in SKILL_TAGS.items()})
    selected = {}
    with args.audit.open() as f:
        for line in f:
            row = json.loads(line)
            bid = row["beatmap_id"]
            if bid not in votes or bid not in meta or row["n_objects"] < 50:
                continue
            m = meta[bid]
            if int(m["mode"]) != 0 or int(m["beatmapset_id"]) != row["beatmapset_id"]:
                raise ValueError(f"Inconsistent ID join for {bid}")
            row.update(skill_labels=votes[bid][0], skill_votes=votes[bid][1],
                       label_status="community_votes_unreviewed_locally", label_source="osu-user-tags",
                       song_key=song_key(m))
            selected[(row["track"], bid)] = row
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "coords").mkdir()
    tracks = {key[0] for key in selected}
    audio_hashes = {}
    retained, exclusions = [], []
    def jobs(shard):
        with tarfile.open(shard, "r:") as tf:
            for member in tf:
                if not member.isfile() or "." not in member.name:
                    continue
                seq, ext = member.name.rsplit(".", 1)
                track = f"{shard.stem}_{seq}"
                if track not in tracks:
                    continue
                with tf.extractfile(member) as f:
                    raw = f.read()
                if ext != "json":
                    audio_hashes[track] = digest(raw)
                    continue
                source = json.loads(raw)
                for b in source["beatmaps"]:
                    key = (track, int(b["beatmap_id"]))
                    if key in selected:
                        r = selected[key]
                        if digest(b["content"].encode()) != r["beatmap_content_sha256"]:
                            raise ValueError(f"Source changed since audit: {key}")
                        yield r, b["content"], meta[key[1]]["file_md5"], str(args.out)
    with Pool(args.workers) as pool:
        for shard in sorted(args.shards.glob("*.tar")):
            for row, excluded in pool.imap(cache_map, jobs(shard), chunksize=8):
                if excluded:
                    exclusions.append(excluded)
                else:
                    retained.append(row)
            print(f"{shard.name}: retained {len(retained)}, excluded {len(exclusions)}", flush=True)
    if not retained:
        raise ValueError("No version-matched maps available")
    rows = freeze_groups(retained, audio_hashes)
    split_stats = {}
    for split in ("train", "validation", "test"):
        subset = sorted((r for r in rows if r["split"] == split), key=lambda r: (r["beatmap_id"], r["track"]))
        target = args.out / f"{split}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
        split_stats[split] = dict(maps=len(subset), groups=len({r["song_group"] for r in subset}),
            labels=dict(Counter(k for r in subset for k in r["skill_labels"])), sha256=digest(target.read_bytes()))
    (args.out / "exclusions.json").write_text(json.dumps(exclusions, indent=2))
    report = dict(schema="autoosu.skill-pilot-data/1", skill_names=SKILL_NAMES, min_votes=args.min_votes,
        source_revision="ea7e5b8b39d45ffa4f8d0ee3769024778bf3ea8f", selected=len(selected), retained=len(rows),
        file_md5_serialization="UTF-8 CRLF reconstructed from HF normalized text",
        exclusions=dict(Counter(r["reason"] for r in exclusions)), splits=split_stats,
        sources={n: digest((args.tags/n).read_bytes()) for n in ("README.md", "tags.csv", "metadata.csv")},
        limits=["Community votes are positive evidence, not exhaustive labels; missing is unknown.",
                "No local human agreement audit yet; this is an exploratory pilot, not M1 release acceptance.",
                "Split holds out audio/artist-title groups from fine-tuning, not from v0 pretraining.",
                "Exact compressed-audio hashes and conservative title grouping do not prove all acoustic near duplicates absent."])
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
