"""Continue an architecture checkpoint until validation plateaus at the minimum LR.

Legacy inference checkpoints lack optimizer state: initialization is explicitly a
weights-only warm start. New state.pt files retain FP32 model/AdamW, controller,
and rank RNG state. DataLoader worker prefetch state is not an exact-resume claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .convergence import Convergence, attribute_nll
from .dataset import TickDataset, read_splits
from .model import TickTransformer
from .train import setup_distributed
from .train_architecture import digest, evaluation_rows, attribute_priors, evaluate, loss_on_batch


def atomic_save(value, path):
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initialize-from", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--attributes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=120000, help="Review boundary, never labelled convergence")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-maps", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2008)
    ap.add_argument("--warmup", type=int, default=500)
    args = ap.parse_args()
    rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank); random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = TickTransformer.load(str(args.initialize_from), str(device))
    model.train()
    initial = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    source_step = initial["step"]
    source_recipe = initial["recipe"]
    splits = read_splits(args.data)
    vocab = json.loads((args.attributes / "vocabulary.json").read_text())
    if model.cfg.attribute_vocab != vocab:
        raise ValueError("Attribute vocabulary differs from the initialization checkpoint")
    split_hashes = {s: digest(args.data / f"{s}.jsonl") for s in splits}
    if split_hashes != source_recipe["data_sha256"]:
        raise ValueError("Continuation must keep the frozen song splits")
    selected = evaluation_rows(splits["validation"], args.eval_maps)
    if args.seq != source_recipe["args"]["seq"] or [r["beatmap_id"] for r in selected] != source_recipe["validation_ids"]:
        raise ValueError("Continuation must keep the frozen validation crops")
    cfg = asdict(model.cfg)
    del initial
    train = TickDataset(splits["train"], args.prep, args.seq, train=True, density_dropout=.5, attributes_dir=args.attributes)
    val = TickDataset(selected, args.prep, args.seq, train=False, density_mode="unknown",
                      evaluation_crop="center", attributes_dir=args.attributes)
    priors = attribute_priors(splits["train"], args.attributes, vocab) if rank == 0 else None
    sampler = DistributedSampler(train, world, rank, shuffle=True, seed=args.seed, drop_last=True) if world > 1 else None
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(train, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0, generator=generator)
    policy = Convergence()
    optimizer = torch.optim.AdamW(model.parameters(), lr=policy.lr, betas=(.9, .95), weight_decay=.05)
    train_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    recipe = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  initialization="weights-only warm start; new AdamW state", source_step=source_step,
                  source_checkpoint_sha256=digest(args.initialize_from), config=cfg, world_size=world,
                  data_sha256=split_hashes, validation_ids=[r["beatmap_id"] for r in selected],
                  policy=asdict(policy) | dict(best_f1=None, best_attribute_nll=None),
                  source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                  exact_dataloader_resume=False, test_set_used=False)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "recipe.json").write_text(json.dumps(recipe, indent=2))
    def emit(record):
        if rank == 0:
            record.update(elapsed_s=time.monotonic() - start)
            with (args.out / "log.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    best = -float("inf")
    best_attrs = float("inf")
    def validate_and_save(step, epoch, batch_index):
        nonlocal best, best_attrs
        action, metrics = "continue", None
        if rank == 0:
            metrics = evaluate(model, val, device, args.eval_maps, priors=priors)
            nll = attribute_nll(metrics)
            action = policy.observe(metrics["gen_onset_f1"], nll, step)
            emit(dict(event="validation", step=step, total_step=source_step + step, action=action,
                      attribute_nll=nll, policy=asdict(policy), **{k:v for k,v in metrics.items() if k != "rows"}))
            (args.out / f"evaluation-{step}.json").write_text(json.dumps(metrics, indent=2))
            checkpoint = dict(config=cfg, state_dict={k:v.detach().cpu().half() for k,v in model.state_dict().items()},
                              step=source_step + step, continuation_step=step, eval=metrics, recipe=recipe)
            atomic_save(checkpoint, args.out / "last.pt")
            if metrics["gen_onset_f1"] > best:
                best = metrics["gen_onset_f1"]
                atomic_save(checkpoint, args.out / "best.pt")
            if nll < best_attrs:
                best_attrs = nll
                atomic_save(checkpoint, args.out / "best-attributes.pt")
        control = [asdict(policy), action]
        if world > 1:
            torch.distributed.broadcast_object_list(control, src=0)
        policy.__dict__.update(control[0])
        rng = dict(rank=rank, torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                   numpy=np.random.get_state(), python=random.getstate(), loader=generator.get_state())
        states = [None] * world if rank == 0 else None
        if world > 1:
            torch.distributed.gather_object(rng, states, dst=0)
        else:
            states = [rng]
        if rank == 0:
            atomic_save(dict(config=cfg, state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                             optimizer=optimizer.state_dict(), policy=asdict(policy), rng=states,
                             step=source_step + step, continuation_step=step, epoch=epoch, batch_index=batch_index,
                             best_f1=best, best_attribute_nll=best_attrs, recipe=recipe), args.out / "state.pt")
        if world > 1:
            torch.distributed.barrier()
        return control[1]
    start = time.monotonic()
    emit(dict(event="started", pid=os.getpid(), source_step=source_step, status="running"))
    validate_and_save(0, 0, -1)
    step = epoch = 0
    total_loss = 0.
    status = "step_limit_pending_review"
    stopped = False
    while step < args.max_steps and not stopped:
        if sampler:
            sampler.set_epoch(epoch)
        for batch_index, data in enumerate(loader):
            if step >= args.max_steps:
                break
            step += 1
            lr = policy.lr * min(1., step / args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            data = {k:v.to(device, non_blocking=True) for k,v in data.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = loss_on_batch(train_model, data)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach())
            if step % 50 == 0:
                emit(dict(event="train", step=step, total_step=source_step + step,
                          loss=total_loss / 50, lr=lr, grad_norm=float(norm)))
                total_loss = 0.
            if step % args.eval_every == 0 or step == args.max_steps:
                if validate_and_save(step, epoch, batch_index) == "converged":
                    status, stopped = "converged_pending_acceptance", True
                    break
        epoch += 1
    emit(dict(event="finished", status=status, step=step, total_step=source_step + step))
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
