"""From-scratch rhythm architecture comparison with frozen song splits and free-generation selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from .dataset import TickDataset, read_splits
from .attributes import ATTRIBUTE_NAMES
from .model import ModelConfig, TickTransformer, masked_inputs, random_mask, sample_masked, sample_attributes
from .train import compare_sequences, setup_distributed

ARCHITECTURES = {
    "flat": dict(n_layers=8, audio_ctx_layers=0, audio_frontend="linear"),
    "context2": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="linear"),
    "attributes": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="linear"),
    "spectral_attributes": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="conv"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss_on_batch(model, data, full_mask_probability=.25):
    labels = data["labels"]
    mask = random_mask(labels)
    full = torch.rand(len(labels), 1, device=labels.device) < full_mask_probability
    mask = torch.where(full, labels != -100, mask)
    targets = torch.where(mask, labels, torch.full_like(labels, -100))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=labels.device.type == "cuda"):
        output = model(data["audio"], masked_inputs(labels, mask), data["metrical"], data["extra"],
                       data["cond"], pad_mask=labels == -100, attribute_targets=data.get("attributes"))
    logits, attributes = output if isinstance(output, tuple) else (output, {})
    rhythm_loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100, label_smoothing=.05)
    attribute_losses = []
    for name, predictions in attributes.items():
        truth = data["attributes"][..., ATTRIBUTE_NAMES.index(name)]
        truth = truth[truth != -100]
        loss = (F.cross_entropy(predictions.float(), truth) if len(truth) else predictions.sum() * 0.)
        attribute_losses.append(loss)
    attribute_loss = torch.stack(attribute_losses).mean() if attribute_losses else rhythm_loss * 0.
    return rhythm_loss + .5 * attribute_loss, rhythm_loss.detach(), attribute_loss.detach()


def longest_run(labels):
    best = run = 0
    for value in labels:
        run = run + 1 if value == 1 else 0
        best = max(best, run)
    return best


def evaluation_rows(rows, count):
    """Use one fixed map per song so multiple difficulties cannot dominate selection."""
    seen, selected = set(), []
    for row in rows:
        if row["song_group"] not in seen:
            selected.append(row)
            seen.add(row["song_group"])
            if len(selected) == count:
                break
    return selected


def attribute_priors(rows, directory, vocabulary, count=4096):
    """A frozen training-only frequency baseline, independent of validation labels."""
    counts = {name: np.zeros(len(values), dtype=np.int64) for name, values in vocabulary.items()}
    for row in rows[:count]:
        targets = np.load(directory / "maps" / f"{row['beatmap_id']}.npz")["attributes"]
        for name, histogram in counts.items():
            values = targets[:, ATTRIBUTE_NAMES.index(name)]
            histogram += np.bincount(values[values != -100], minlength=len(histogram))
    return {name: (histogram + 1.) / (histogram.sum() + len(histogram)) for name, histogram in counts.items()}


@torch.no_grad()
def evaluate(model, dataset, device, count=64, priors=None):
    model.eval()
    measurements = []
    attribute_totals = {name: dict(reference_count=0, reference_correct=0, reference_nll=0., prior_nll=0.,
                                  matched_count=0, matched_correct=0, generated_count=0, generated_nondefault=0)
                        for name in model.attribute_heads}
    with torch.random.fork_rng(devices=[device.index]):
        for i in range(min(count, len(dataset))):
            data = {k: v.to(device) for k, v in dataset[i].items()}
            n = int((data["labels"] != -100).sum())
            if n < 2:
                raise ValueError("Empty frozen evaluation crop")
            true = data["labels"][:n].cpu().numpy()
            generator = torch.Generator(device=device).manual_seed(240924 + i)
            pred = sample_masked(model, data["audio"][:n], data["metrical"][:n], data["extra"][:n],
                                 data["cond"], steps=12, temperature=.9, generator=generator)
            metrics = compare_sequences(pred, true)
            metrics.update(longest_run_true=longest_run(true), longest_run_pred=longest_run(pred),
                           density_absolute_error=abs(metrics["gen_density_pred"] - metrics["gen_density_true"]),
                           slider_share_absolute_error=abs(metrics["gen_slider_share_pred"] - metrics["gen_slider_share_true"]))
            if model.attribute_heads:
                attrs = data["attributes"][:n]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, logits = model(data["audio"][:n][None], data["labels"][:n][None], data["metrical"][:n][None],
                                      data["extra"][:n][None], data["cond"][None], attribute_targets=attrs[None])
                sampled = sample_attributes(model, data["audio"][:n], data["metrical"][:n], data["extra"][:n],
                                            data["cond"], pred, generator=generator)
                for name, values in logits.items():
                    target = attrs[:, ATTRIBUTE_NAMES.index(name)]
                    known = target != -100
                    count = int(known.sum())
                    stat = attribute_totals[name]
                    if count:
                        stat["reference_count"] += count
                        stat["reference_correct"] += int((values.argmax(-1) == target[known]).sum())
                        stat["reference_nll"] += float(F.cross_entropy(values.float(), target[known], reduction="sum"))
                        if priors is not None:
                            stat["prior_nll"] += float(-np.log(priors[name][target[known].cpu().numpy()]).sum())
                    actual = sampled[name]
                    expected = target.cpu().numpy()
                    matched = (expected != -100) & (actual != -100) & (pred == true)
                    stat["matched_count"] += int(matched.sum())
                    stat["matched_correct"] += int((actual[matched] == expected[matched]).sum())
                    stat["generated_count"] += int((actual != -100).sum())
                    stat["generated_nondefault"] += int(((actual != -100) & (actual != 0)).sum())
            measurements.append(metrics)
    model.train()
    return {key: float(np.mean([row[key] for row in measurements])) for key in measurements[0]} | {
        "maps": len(measurements), "conditions": "unknown density, center crop, 12-round generation",
        "rows": measurements, "attributes": attribute_totals}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--attributes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-maps", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1008)
    args = ap.parse_args()
    rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank); random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    splits = read_splits(args.data)
    vocabulary = json.loads((args.attributes / "vocabulary.json").read_text()) if args.architecture.endswith("attributes") else {}
    cfg = ModelConfig(d_model=512, n_heads=8, mode="masked", max_len=2048,
                      attribute_vocab=vocabulary, **ARCHITECTURES[args.architecture])
    model = TickTransformer(cfg).to(device)
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    attrs = args.attributes if vocabulary else None
    train = TickDataset(splits["train"], args.prep, args.seq, train=True, density_dropout=.5, attributes_dir=attrs)
    selected = evaluation_rows(splits["validation"], args.eval_maps)
    priors = attribute_priors(splits["train"], args.attributes, vocabulary) if rank == 0 and vocabulary else {}
    val = TickDataset(selected, args.prep, args.seq, train=False,
                      density_mode="unknown", evaluation_crop="center", attributes_dir=attrs)
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    loader = DataLoader(train, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed + rank))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.05)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
        recipe = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            config=asdict(cfg), world_size=world, parameters=sum(p.numel() for p in model.parameters()),
            torch_version=str(torch.__version__), initialization="random", density_dropout=.5, full_mask_probability=.25,
            attribute_loss_weight=.5, attribute_loss="mean cross-entropy over applicable object-attribute heads",
            attribute_prior_training_ids=[r["beatmap_id"] for r in splits["train"][:4096]] if vocabulary else [],
            attributes_report_sha256=digest(args.attributes / "report.json"),
            selection="highest validation free-generation onset F1; no test-set use",
            validation_ids=[r["beatmap_id"] for r in selected],
            data_sha256={s: digest(args.data / f"{s}.jsonl") for s in splits},
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})
        (args.out / "recipe.json").write_text(json.dumps(recipe, indent=2))
        log = (args.out / "log.jsonl").open("w")
    def emit(record):
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n"); log.flush()
    def save(name, step, metrics):
        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        target = args.out / name
        temporary = target.with_suffix(".partial")
        torch.save(dict(config=asdict(cfg), state_dict=state, step=step, eval=metrics, recipe=recipe), temporary)
        temporary.replace(target)
    step = epoch = 0
    best = -1.
    start = time.monotonic()
    loss_sum = rhythm_sum = attribute_sum = 0.
    while step < args.steps:
        if sampler:
            sampler.set_epoch(epoch)
        epoch += 1
        for data in loader:
            if step >= args.steps:
                break
            step += 1
            decay = max(0., (step - args.warmup) / max(1, args.steps - args.warmup))
            lr = args.lr * min(1., step / args.warmup) * (.05 + .95 * (1 + math.cos(math.pi * decay)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, rhythm_loss, attribute_loss = loss_on_batch(train_model, data)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
            rhythm_sum += float(rhythm_loss)
            attribute_sum += float(attribute_loss)
            if step % 50 == 0:
                emit(dict(step=step, loss=loss_sum / 50, rhythm_loss=rhythm_sum / 50,
                          attribute_loss=attribute_sum / 50, lr=lr, grad_norm=float(norm), elapsed_s=time.monotonic() - start))
                loss_sum = rhythm_sum = attribute_sum = 0.
            if step % args.eval_every == 0 or step == args.steps:
                if rank == 0:
                    ev = evaluate(model, val, device, args.eval_maps, priors=priors)
                    emit(dict(step=step, **{k: v for k, v in ev.items() if k != "rows"}))
                    (args.out / f"evaluation-{step}.json").write_text(json.dumps(ev, indent=2))
                    save("last.pt", step, ev)
                    if ev["gen_onset_f1"] > best:
                        best = ev["gen_onset_f1"]
                        save("best.pt", step, ev)
                if world > 1:
                    torch.distributed.barrier()
    emit(dict(status="complete", step=step, elapsed_s=time.monotonic() - start))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
