"""Bounded, frozen-manifest positive-label fine-tuning of rhythm or coordinate v0.

This experimental entry point uses the existing model losses and coordinate representation.
It does not publish weights or use the held-out test manifest for checkpoint selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .coord import DiT_models, create_diffusion, timestep_embedding
from .coord_infer import CoordModel, load_coord_model
from .dataset import TickDataset, read_splits
from .model import TickTransformer
from .skills import SKILL_NAMES, skill_vector
from .train import model_loss, setup_distributed


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class CoordinateDataset(Dataset):
    def __init__(self, rows, cm, seq_len=128, train=True, skill_dropout=0.2):
        self.rows, self.cm, self.seq_len, self.train = rows, cm, seq_len, train
        self.skill_dropout = skill_dropout

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        seq = np.load(row["coord_path"], mmap_mode="r")
        if seq.shape[1] < self.seq_len:
            raise ValueError(f"Coordinate sequence too short: {row['beatmap_id']}")
        maximum = seq.shape[1] - self.seq_len
        start = int(np.random.randint(maximum + 1)) if self.train else maximum // 2
        # Include the actual preceding token when computing distances at crop boundaries.
        points = torch.from_numpy(np.array(seq[:2, start:start + self.seq_len]))
        prior = torch.roll(points, 1, 1)
        prior[:, 0] = torch.from_numpy(np.array(seq[:2, start - 1])) if start else torch.tensor([256., 192.])
        distances = torch.linalg.vector_norm(points - prior, dim=0)
        if not self.train or np.random.random() < 0.5:
            distances.zero_()
        else:
            distances *= torch.pow(2, torch.randn_like(distances) * 0.1)
        if self.train:
            if np.random.random() < 0.5:
                points[0] = 512 - points[0]
            if np.random.random() < 0.5:
                points[1] = 384 - points[1]
        positions = points / torch.tensor([[512.], [384.]]) * 2 - 1
        times = torch.from_numpy(np.array(seq[2, start:start + self.seq_len]))
        if self.train:
            times = times - times[0] + float(np.random.random() * 1000000)
        context = torch.cat([timestep_embedding(times * 0.1, 128).T,
                             timestep_embedding(distances, 128).T,
                             torch.from_numpy(np.array(seq[3:, start:start + self.seq_len]))])
        labels = row["skill_labels"]
        if self.train and np.random.random() < self.skill_dropout:
            labels = {}
        star = None if self.train and np.random.random() < 0.2 else row["sr"]
        cs = None if self.train and np.random.random() < 0.2 else row["cs"]
        return {"positions": positions, "context": context, "cond": self.cm.class_vector(star, cs, labels)}


def initialize_coord(path, names):
    base = load_coord_model(path, "cpu")
    if base.skill_names:
        raise ValueError("Expected an unconditioned v0 base")
    cm = CoordModel(None, "cpu", base.num_diff_classes, base.max_difficulty, base.num_cs_classes,
                    base.arch, tuple(names))
    net = DiT_models[cm.arch](context_size=272, class_size=cm.num_tokens)
    state = dict(base.net.state_dict())
    state["y_embedder.class_embedding.0.weight"] = torch.nn.functional.pad(
        state["y_embedder.class_embedding.0.weight"], (0, len(names)))
    net.load_state_dict(state)
    return net, cm


def batch_loss(model, batch, kind, diffusion, device):
    if kind == "rhythm":
        return model_loss(model, batch, "masked", str(device))[0]
    t = torch.randint(0, diffusion.num_timesteps, (len(batch["positions"]),), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return diffusion.training_losses(model, batch["positions"], t,
            model_kwargs=dict(c=batch["context"], y=batch["cond"]))["loss"].mean()


@torch.no_grad()
def evaluate(model, loader, kind, diffusion, device, max_batches):
    model.eval()
    totals = {"labelled": 0., "unknown": 0.}
    count = 0
    # Fixed crops, masks, diffusion noise and sample order at every checkpoint.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(1008)
        for bi, data in enumerate(loader):
            if bi >= max_batches:
                break
            data = {k: v.to(device) for k, v in data.items()}
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            for condition in totals:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)
                cond = data["cond"].clone()
                if condition == "unknown":
                    cond[:, -len(SKILL_NAMES):] = 0
                loss = batch_loss(model, {**data, "cond": cond}, kind, diffusion, device)
                totals[condition] += float(loss) * len(cond)
            count += len(data["cond"])
    model.train()
    if not count:
        raise ValueError("No validation examples")
    return {f"val_{k}_loss": v / count for k, v in totals.items()} | {"val_maps": count}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["rhythm", "coord"], required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prep", type=Path, default=Path("data/prep"))
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1008)
    args = ap.parse_args()
    rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    splits = read_splits(args.data)
    if args.kind == "rhythm":
        model = TickTransformer.load(str(args.base), "cpu").with_skills(SKILL_NAMES)
        datasets = [TickDataset(splits[split], args.prep, args.seq, train=train, skill_names=SKILL_NAMES)
                    for split, train in (("train", True), ("validation", False))]
        meta = asdict(model.cfg)
    else:
        model, cm = initialize_coord(args.base, SKILL_NAMES)
        datasets = [CoordinateDataset(splits[split], cm, args.seq, train=train)
                    for split, train in (("train", True), ("validation", False))]
        meta = dict(format="autoosu-coord-skills-v1", arch=cm.arch, context_size=272,
                    num_diff_classes=cm.num_diff_classes, max_difficulty=cm.max_difficulty,
                    num_cs_classes=cm.num_cs_classes, skill_names=SKILL_NAMES)
    model = model.to(device).train()
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    sampler = DistributedSampler(datasets[0], world, rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    train_loader = DataLoader(datasets[0], batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(datasets[1], batch_size=args.batch, num_workers=0, shuffle=False)
    if not len(train_loader):
        raise ValueError("Not enough data for a training batch")
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=1000, noise_schedule="squaredcos_cap_v2")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
        recipe = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            world_size=world, base_sha256=digest(args.base),
            split_sha256={s: digest(args.data / f"{s}.jsonl") for s in splits},
            source_sha256={str(p.relative_to(Path(__file__).parents[2])): digest(p)
                           for p in Path(__file__).parent.glob("*.py")}, skill_names=SKILL_NAMES,
            torch_version=str(torch.__version__), model_config=meta)
        (args.out / "recipe.json").write_text(json.dumps(recipe, indent=2))
        log = (args.out / "log.jsonl").open("w")
    if world > 1:
        torch.distributed.barrier()
    def emit(record):
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n"); log.flush()
    def save(name, step, metrics):
        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        ckpt = {"config" if args.kind == "rhythm" else "meta": meta, "state_dict": state,
                "step": step, "eval": metrics, "recipe": recipe}
        temporary = args.out / (name + ".partial")
        torch.save(ckpt, temporary)
        temporary.replace(args.out / name)
    if rank == 0:
        baseline = evaluate(model, val_loader, args.kind, diffusion, device, args.eval_batches)
        emit({"step": 0, **baseline})
        save("initial.pt", 0, baseline)
    if world > 1:
        torch.distributed.barrier()
    start_time = time.monotonic()
    step = epoch = 0
    best = float("inf")
    loss_sum = 0.
    while step < args.steps:
        if sampler:
            sampler.set_epoch(epoch)
        epoch += 1
        for data in train_loader:
            if step >= args.steps:
                break
            step += 1
            scale = min(1., step / args.warmup) * (0.1 + 0.9 * (1 + math.cos(math.pi * step / args.steps)) / 2)
            for group in opt.param_groups:
                group["lr"] = args.lr * scale
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(train_model, data, args.kind, diffusion, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            loss_sum += float(loss.detach())
            if step % 50 == 0:
                emit(dict(step=step, loss=loss_sum / 50, lr=args.lr * scale,
                          grad_norm=float(norm), elapsed_s=round(time.monotonic() - start_time, 1)))
                loss_sum = 0.
            if step % args.eval_every == 0 or step == args.steps:
                if rank == 0:
                    metrics = evaluate(model, val_loader, args.kind, diffusion, device, args.eval_batches)
                    emit({"step": step, **metrics})
                    save("last.pt", step, metrics)
                    score = metrics["val_labelled_loss"] + metrics["val_unknown_loss"]
                    if score < best:
                        best = score
                        save("best.pt", step, metrics)
                if world > 1:
                    torch.distributed.barrier()
    emit(dict(status="complete", step=step, elapsed_s=round(time.monotonic() - start_time, 1)))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
