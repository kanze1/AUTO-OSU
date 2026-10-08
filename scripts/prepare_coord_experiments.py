"""Freeze coordinate training sources from existing song-grouped corpus manifests.

Unsupported source geometry is counted and excluded explicitly, never converted to circles.
The held-out test manifest is read only to validate split disjointness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "coord")]
import numpy as np
from slider import Beatmap
from slider.beatmap import Slider
import torch

from autoosu.ml.coord_context import music_features
from autoosu.ml.coord_objects import complete_object_crops
from autoosu.ml.dataset import read_splits
from autoosu.ml.train_architecture import evaluation_rows, digest
from autoosu.provenance import atomic_json
from osu_diffusion.utils.data_loading import get_data


def tokenize(content, budget):
    bm = Beatmap.parse(content)
    red = [p for p in bm.timing_points if p.parent is None]
    if not red or any(abs(p.ms_per_beat-red[0].ms_per_beat) > 1e-5 for p in red):
        raise ValueError("unsupported_variable_tempo")
    parts = []
    for obj in bm.hit_objects(stacking=False):
        part = get_data(obj)
        if isinstance(obj, Slider) and int(part[0, 3:].argmax()) not in (4, 5):
            raise ValueError("unsupported_slider_tokenization")
        parts.append(part)
    if not parts:
        raise ValueError("empty_map")
    seq = torch.cat(parts).T.numpy()
    if not np.isfinite(seq).all():
        raise ValueError("nonfinite_source_geometry")
    if seq.shape[1] < budget:
        raise ValueError("shorter_than_window")
    crops = complete_object_crops(seq[2], seq[3:].argmax(0), budget)
    return seq, crops, red[0].ms_per_beat, red[0].offset.total_seconds()*1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "prep", "shards", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--train-songs", type=int, default=4096)
    ap.add_argument("--validation-songs", type=int, default=128)
    ap.add_argument("--length", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(2)
    splits = read_splits(args.data)
    selected = {s:evaluation_rows(splits[s], count) for s,count in
                (("train", args.train_songs), ("validation", args.validation_songs))}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "maps").mkdir()
    requested = {r["beatmap_id"]:r for rows in selected.values() for r in rows}
    by_shard = {}
    for row in requested.values():
        by_shard.setdefault(row["shard"], {}).setdefault(row["metadata_member"], []).append(row)
    accepted, rejected = {}, []
    for shard, members in sorted(by_shard.items()):
        with tarfile.open(args.shards / shard, "r:") as archive:
            for member in archive:
                if member.name not in members:
                    continue
                with archive.extractfile(member) as stream:
                    metadata = json.load(stream)
                maps = {int(b["beatmap_id"]):b["content"] for b in metadata["beatmaps"]}
                for row in members[member.name]:
                    content = maps[row["beatmap_id"]]
                    if hashlib.sha256(content.encode()).hexdigest() != row["beatmap_content_sha256"]:
                        raise ValueError("Frozen source content changed")
                    try:
                        seq, crops, beat_ms, offset = tokenize(content, args.length)
                    except (ValueError, AssertionError, ZeroDivisionError) as error:
                        rejected.append(dict(beatmap_id=row["beatmap_id"], split=row["split"],
                                             reason=f"{type(error).__name__}: {error}"))
                        continue
                    mel_path = args.prep / "tracks" / f"{row['track']}.npy"
                    mel = np.load(mel_path, mmap_mode="r")
                    music = music_features(mel, seq[2, :1], beat_ms, offset)
                    path = args.out / "maps" / f"{row['beatmap_id']}.npz"
                    np.savez_compressed(path, seq=seq, crops=np.asarray(crops), beat_ms=beat_ms, offset=offset,
                        song=music['song'].numpy(), song_beats=music['song_beats'].numpy())
                    accepted[row['beatmap_id']] = dict(row, prepared_sha256=digest(path), mel_sha256=digest(mel_path))
        print(json.dumps(dict(shard=shard, accepted=len(accepted), rejected=len(rejected))), flush=True)
    if len(accepted)+len(rejected) != len(requested):
        raise ValueError("Frozen sources missing from archive")
    counts = {}
    for split, rows in selected.items():
        values = [accepted[r['beatmap_id']] for r in rows if r['beatmap_id'] in accepted]
        if not values:
            raise ValueError(f"No supported songs in {split}")
        counts[split] = dict(requested=len(rows), accepted=len(values))
        (args.out / f"{split}.jsonl").write_text(''.join(json.dumps(v)+'\n' for v in values))
    atomic_json(args.out / "report.json", dict(counts=counts, rejected=rejected, length=args.length,
        rejection_counts=dict(Counter(r['reason'] for r in rejected)),
        split_sha256={s:digest(args.data/f'{s}.jsonl') for s in splits},
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__), ROOT/'autoosu/ml/coord_objects.py', ROOT/'coord/osu_diffusion/utils/data_loading.py']},
        scope="Song-grouped development subset; constant tempo and supported complete-object geometry only. Base pretraining exposure unknown. No replacement of rejected songs; no test-set selection."))
    print('COMPLETE', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
