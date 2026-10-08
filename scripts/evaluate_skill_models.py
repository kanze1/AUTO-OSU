"""Evaluate frozen initial/best pilot checkpoints once on the fine-tuning held-out test maps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from autoosu.ml.coord import create_diffusion
from autoosu.ml.coord_infer import load_coord_model
from autoosu.ml.dataset import TickDataset
from autoosu.ml.model import TickTransformer
from autoosu.ml.skills import SKILL_NAMES
from autoosu.ml.train_skills import CoordinateDataset, digest, evaluate, read_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["rhythm", "coord"], required=True)
    ap.add_argument("--initial", type=Path, required=True)
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    rows = read_splits(args.data)["test"]
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=1000, noise_schedule="squaredcos_cap_v2")
    result = dict(kind=args.kind, split="test", test_sha256=digest(args.data / "test.jsonl"),
                  evaluation_seed=1008, maps=len(rows), checkpoints={},
                  scope="Held out from fine-tuning, not necessarily v0 pretraining; no checkpoint selection")
    for name, path in (("initial", args.initial), ("best", args.best)):
        if args.kind == "rhythm":
            model = TickTransformer.load(str(path), "cuda:0")
            dataset = TickDataset(rows, args.prep, 512, train=False, skill_names=SKILL_NAMES)
        else:
            cm = load_coord_model(path, "cuda:0")
            model = cm.net
            dataset = CoordinateDataset(rows, cm, 128, train=False)
        loader = DataLoader(dataset, batch_size=32, num_workers=0)
        metrics = evaluate(model, loader, args.kind, diffusion, device, len(loader))
        result["checkpoints"][name] = dict(sha256=digest(path),
            **{key.replace("val_", "test_"): value for key, value in metrics.items()})
        del model, dataset, loader
        if args.kind == "coord":
            del cm
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
