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
from .model import ModelConfig, TickTransformer, ar_inputs, masked_inputs, random_mask, sample_ar, sample_masked, sample_attributes
from .train import compare_sequences, setup_distributed
from .tracking import ExperimentTracker

ARCHITECTURES = {
    "flat": dict(n_layers=8, audio_ctx_layers=0, audio_frontend="linear"),
    "context2": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="linear"),
    "attributes": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="linear"),
    "spectral_attributes": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="conv"),
    "spectral_context4_attributes": dict(n_layers=4, audio_ctx_layers=4, audio_frontend="conv"),
    "spectral_ar_attributes": dict(n_layers=6, audio_ctx_layers=2, audio_frontend="conv", mode="ar"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss_on_batch(model, data, full_mask_probability=.25):
    labels = data["labels"]
    core = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    if core.causal:
        tokens, targets = ar_inputs(labels, data["prev_label"]), labels
    else:
        mask = random_mask(labels)
        full = torch.rand(len(labels), 1, device=labels.device) < full_mask_probability
        mask = torch.where(full, labels != -100, mask)
        tokens = masked_inputs(labels, mask)
        targets = torch.where(mask, labels, torch.full_like(labels, -100))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=labels.device.type == "cuda"):
        output = model(data["audio"], tokens, data["metrical"], data["extra"],
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
    examples = []
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
            audio_args = (data["audio"][:n], data["metrical"][:n], data["extra"][:n], data["cond"])
            pred = (sample_ar(model, *audio_args, temperature=.9, generator=generator,
                              prev0=int(data["prev_label"])) if model.causal else
                    sample_masked(model, *audio_args, steps=12, temperature=.9, generator=generator))
            metrics = compare_sequences(pred, true)
            if len(examples) < 2:
                examples.append(dict(beatmap_id=dataset.rows[i]['beatmap_id'],
                                     reference=true.tolist(), generated=pred.tolist()))
            metrics.update(longest_run_true=longest_run(true), longest_run_pred=longest_run(pred),
                           density_absolute_error=abs(metrics["gen_density_pred"] - metrics["gen_density_true"]),
                           slider_share_absolute_error=abs(metrics["gen_slider_share_pred"] - metrics["gen_slider_share_true"]))
            if model.attribute_heads:
                attrs = data["attributes"][:n]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = (ar_inputs(data["labels"][:n][None], data["prev_label"][None])
                              if model.causal else data["labels"][:n][None])
                    _, logits = model(data["audio"][:n][None], tokens, data["metrical"][:n][None],
                                      data["extra"][:n][None], data["cond"][None], attribute_targets=attrs[None])
                sampled = sample_attributes(model, data["audio"][:n], data["metrical"][:n], data["extra"][:n],
                                            data["cond"], pred, generator=generator, prev0=int(data["prev_label"]))
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
        "maps": len(measurements), "conditions": "unknown density, center crop, " + ("autoregressive generation" if model.causal else "12-round generation"),
        "rows": measurements, "attributes": attribute_totals, "examples": examples}


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
    ap.add_argument("--wandb-project")
    ap.add_argument("--wandb-group")
    ap.add_argument("--wandb-entity")
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = ap.parse_args()
    rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank); random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    splits = read_splits(args.data)
    vocabulary = json.loads((args.attributes / "vocabulary.json").read_text()) if args.architecture.endswith("attributes") else {}
    cfg = ModelConfig(d_model=512, n_heads=8, max_len=2048,
                      attribute_vocab=vocabulary, **(dict(mode="masked") | ARCHITECTURES[args.architecture]))
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
            torch_version=str(torch.__version__), initialization="random", density_dropout=.5,
            full_mask_probability=0. if cfg.mode=='ar' else .25,
            attribute_loss_weight=.5, attribute_loss="mean cross-entropy over applicable object-attribute heads",
            attribute_prior_training_ids=[r["beatmap_id"] for r in splits["train"][:4096]] if vocabulary else [],
            attributes_report_sha256=digest(args.attributes / "report.json"),
            selection="highest validation free-generation onset F1; no test-set use",
            validation_ids=[r["beatmap_id"] for r in selected],
            data_sha256={s: digest(args.data / f"{s}.jsonl") for s in splits},
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")})
        (args.out / "recipe.json").write_text(json.dumps(recipe, indent=2))
        log = (args.out / "log.jsonl").open("w")
        tracker = ExperimentTracker(args.out, args.wandb_project, args.wandb_group, args.out.name,
                                    recipe, args.wandb_mode, args.wandb_entity)
    def emit(record):
        if rank == 0:
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n"); log.flush()
            tracker.log(record.get("step", 0), record)
    def save(name, step, metrics):
        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        target = args.out / name
        temporary = target.with_suffix(".partial")
        torch.save(dict(config=asdict(cfg), state_dict=state, step=step, eval=metrics, recipe=recipe), temporary)
        temporary.replace(target)
    step = epoch = 0
    best = -1.
    best_step = 0
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
                    emit(dict(step=step, **{k: v for k, v in ev.items() if k not in ("rows", "examples")}))
                    (args.out / f"evaluation-{step}.json").write_text(json.dumps(ev, indent=2))
                    tracker.table(step, "validation/songs", ['beatmap_id','song_group',*ev['rows'][0]],
                                  [[selected[i]['beatmap_id'],selected[i]['song_group'],*row.values()]
                                   for i,row in enumerate(ev['rows'])])
                    for i,example in enumerate(ev['examples']):
                        tracker.table(step, f'samples/rhythm-{i}', ['tick','reference','generated'],
                                      [[j,a,b] for j,(a,b) in enumerate(zip(example['reference'],example['generated']))])
                    save("last.pt", step, ev)
                    state_path = args.out / "state.partial"
                    torch.save(dict(model=model.state_dict(), config=asdict(cfg), optimizer=optimizer.state_dict(),
                        step=step, epoch=epoch, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                        scope="Recovery state; exact DataLoader prefetch replay is not supported"), state_path)
                    state_path.replace(args.out / "state.pt")
                    if ev["gen_onset_f1"] > best:
                        best = ev["gen_onset_f1"]
                        best_step = step
                        save("best.pt", step, ev)
                if world > 1:
                    torch.distributed.barrier()
    emit(dict(status="complete", step=step, best_gen_onset_f1=best, best_step=best_step,
              elapsed_s=time.monotonic() - start))
    if rank == 0:
        tracker.finish()
        log.close()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
