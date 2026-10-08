"""Freeze full-corpus song groups using source content and actual audio bytes, without skill labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile
import unicodedata


def digest(data):
    return hashlib.sha256(data).hexdigest()


def grouped_split(rows, audio_hashes):
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    for row in rows:
        key = "track:" + row["track"]
        identities = ["audio:" + audio_hashes[row["track"]], "set:" + str(row["beatmapset_id"]),
                      "source:" + row["source_audio_hash"]]
        # Punctuation-only titles carry no song identity; exact audio/set identities still apply.
        if all(row["song_key"]):
            identities.append("song:" + json.dumps(row["song_key"]))
        for identity in identities:
            union(key, identity)
    for row in rows:
        group = find("track:" + row["track"])
        row["song_group"] = digest(group.encode())
        row["audio_bytes_sha256"] = audio_hashes[row["track"]]
        bucket = int(digest(("rhythm-v1-20261008:" + group).encode())[:8], 16) % 100
        row["split"] = "test" if bucket < 5 else "validation" if bucket < 10 else "train"
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.audit.read_text().splitlines()]
    rows = [r for r in rows if r["n_objects"] >= 50]
    lookup = {(r["track"], r["beatmap_id"]): r for r in rows}
    audio, found = {}, set()
    normalize = lambda s: re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", s).casefold())
    for shard in sorted(args.shards.glob("*.tar")):
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                seq, ext = member.name.rsplit(".", 1)
                track = f"{shard.stem}_{seq}"
                with archive.extractfile(member) as stream:
                    if ext == "json":
                        meta = json.load(stream)
                        for bm in meta["beatmaps"]:
                            key = track, int(bm["beatmap_id"])
                            if key not in lookup:
                                continue
                            row = lookup[key]
                            raw = bm["content"].encode()
                            if digest(raw) != row["beatmap_content_sha256"]:
                                raise ValueError(f"Source changed: {key}")
                            row["song_key"] = [normalize(bm["artist"]), normalize(bm["title"])]
                            found.add(key)
                    else:
                        h = hashlib.sha256()
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            h.update(chunk)
                        audio[track] = h.hexdigest()
        print(shard.name, len(found), flush=True)
    if found != set(lookup):
        raise ValueError("Audit rows missing from source")
    rows = grouped_split(rows, audio)
    args.out.mkdir(parents=True, exist_ok=False)
    stats = {}
    for split in ("train", "validation", "test"):
        subset = sorted((r for r in rows if r["split"] == split), key=lambda r: digest(f"v1-order:{r['beatmap_id']}".encode()))
        for row in subset:
            for key in ("skill_labels", "source_tags", "candidate_terms", "label_source", "label_status"):
                row.pop(key, None)
        target = args.out / f"{split}.jsonl"
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
        stats[split] = dict(maps=len(subset), groups=len({r["song_group"] for r in subset}),
                            star_bands=dict(Counter(int(r["sr"]) for r in subset)), sha256=digest(target.read_bytes()))
    report = dict(schema="autoosu.architecture-data/1", source_audit_sha256=digest(args.audit.read_bytes()),
                  splits=stats, grouping="transitive audio bytes, declared audio identity, set and nonempty normalized artist/title",
                  maps_without_text_identity=sum(not all(r["song_key"]) for r in rows),
                  limits="No full acoustic near-duplicate certification; v0 baseline may have seen held-out songs.")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
