"""Bounded spatial architecture comparisons on frozen song-grouped corpus sources."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import torch

from .coord import create_diffusion
from .coord_context import AudioConditionedCoord, MusicContextConfig
from .coord_infer import Sequence, load_coord_model, sequence_context
from .coord_objects import object_layout, complete_object_crops
from .dataset import FRAME_MS, gather_patches
from .tracking import ExperimentTracker
from .train_architecture import digest
from ..provenance import atomic_json

VARIANTS = ('audio-frozen', 'object-frozen', 'object-joint', 'spacing-joint')


class CoordinateDataset:
    def __init__(self, root, prep, split, length):
        self.root, self.prep, self.length = Path(root), Path(prep), length
        self.rows = [json.loads(s) for s in (self.root/f'{split}.jsonl').read_text().splitlines()]

    @lru_cache(maxsize=32)
    def source(self, index):
        row = self.rows[index]
        path = self.root/'maps'/f"{row['beatmap_id']}.npz"
        if digest(path) != row['prepared_sha256']:
            raise ValueError('Coordinate source cache changed')
        with np.load(path) as z:
            source = {k:z[k] for k in z.files}
        mel_path = self.prep/'tracks'/f"{row['track']}.npy"
        if digest(mel_path) != row['mel_sha256']:
            raise ValueError('Prepared audio changed')
        source['mel'] = np.load(mel_path, mmap_mode='r')
        return source

    def example(self, index, rng=None):
        row, source = self.rows[index], self.source(index)
        seq = source['seq']
        kinds = seq[3:].argmax(0)
        if rng is None:
            crops = complete_object_crops(seq[2], kinds, self.length)
            start, stop = crops[len(crops)//2]
        else:
            layout = object_layout(seq[2], kinds)
            heads, ends = layout['heads'].numpy(), layout['ends'].numpy()
            # Crop from a complete object; keep the same token budget for every variant.
            last = max(1, int(np.searchsorted(heads, max(1, seq.shape[1]-self.length))))
            start = int(heads[int(rng.integers(last))])
            last_object = int(np.searchsorted(ends, start+self.length, side='left'))-1
            stop = int(ends[last_object])+1
        crop = seq[:, start:stop]
        times, types = crop[2], crop[3:].argmax(0)
        clean = torch.from_numpy(crop[:2].copy())[None]/torch.tensor([[[512.], [384.]]])*2-1
        music = dict(local=torch.from_numpy(gather_patches(source['mel'], np.rint(times/FRAME_MS).astype(np.int64))).float()[None]/255.,
            song=torch.from_numpy(source['song'])[None], song_beats=torch.from_numpy(source['song_beats'])[None],
            query_beats=torch.from_numpy(((times-source['offset'])/source['beat_ms']).astype(np.float32))[None])
        return dict(clean=clean, c=sequence_context(Sequence(times, types, []))[None], music=music,
                    objects=object_layout(times, types), row=row, start=start, times=times, types=types)


def to_device(e, cm, device):
    return dict(e, clean=e['clean'].to(device), c=e['c'].to(device),
        music={k:v.to(device) for k,v in e['music'].items()}, objects={k:v.to(device) for k,v in e['objects'].items()},
        y=cm.class_vector(e['row']['sr'], e['row']['cs'])[None].to(device))


def model_kwargs(model, e):
    kwargs = dict(c=e['c'], y=e['y'])
    if isinstance(model, AudioConditionedCoord):
        kwargs.update(music=e['music'], objects=e['objects'])
    return kwargs


@torch.no_grad()
def evaluate(model, cm, dataset, diffusion, device, oracle=False):
    model.eval()
    rows = []
    generator = torch.Generator().manual_seed(1949)
    for index in range(len(dataset.rows)):
        e = to_device(dataset.example(index), cm, device)
        plan_nll = None
        if isinstance(model, AudioConditionedCoord) and model.cfg.spacing_plan:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, parameters = model.encode_condition(e['c'], e['music'], e['objects'])
                plan_nll = float(model.planner.loss(parameters, e['clean'], e['objects']))
        for timestep in (100, 500, 900):
            noise = torch.randn(e['clean'].shape, generator=generator).to(device)
            t = torch.tensor([timestep], device=device)
            kw = model_kwargs(model, e)
            if oracle:
                # Oracle uses the legacy consecutive-token distance convention, not object exits.
                xy = ((e['clean'][0].T+1)*e['clean'].new_tensor([256., 192.])).cpu().numpy()
                distances = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=1)
                kw['c'] = sequence_context(Sequence(e['times'], e['types'], []), distances)[None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(diffusion.q_sample(e['clean'], t, noise=noise), t, **kw)[:, :2].float()
            error = (prediction-noise).square().mean(1)[0]
            owners = e['objects']['owners']
            counts = torch.bincount(owners)
            objects = torch.zeros(len(counts), device=device).scatter_add_(0, owners, error)/counts
            row = dict(beatmap_id=e['row']['beatmap_id'], song_group=e['row']['song_group'], crop=e['start'],
                       timestep=timestep, mse=float(error.mean()), object_mse=float(objects.mean()))
            if plan_nll is not None:
                row['plan_nll'] = plan_nll
            rows.append(row)
    summary = {key:float(np.mean([r[key] for r in rows])) for key in ('mse', 'object_mse')}
    if rows and 'plan_nll' in rows[0]:
        summary['plan_nll'] = float(np.mean([r['plan_nll'] for r in rows]))
    return dict(summary=summary, rows=rows)


@torch.no_grad()
def sample(model, cm, dataset, device, out, tracker, step):
    model.eval()
    diffusion = create_diffusion(timestep_respacing=[100], diffusion_steps=1000, noise_schedule='squaredcos_cap_v2')
    reports = []
    for index in range(min(4, len(dataset.rows))):
        e = to_device(dataset.example(index), cm, device)
        generator = torch.Generator(device=device).manual_seed(5678+index)
        torch.manual_seed(5678+index)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            positions = diffusion.p_sample_loop(model, e['clean'].shape,
                noise=torch.randn(e['clean'].shape, generator=generator, device=device), clip_denoised=True,
                model_kwargs=model_kwargs(model, e), device=device)
        if not torch.isfinite(positions).all():
            raise FloatingPointError('Nonfinite generated geometry')
        values = positions[0].T.float().cpu().numpy()
        np.savez(out/f'sample-{index}.npz', positions=values, times=e['times'], types=e['types'])
        heads = np.isin(e['types'], [0, 1, 4, 5])
        reports.append(dict(beatmap_id=e['row']['beatmap_id'], tokens=len(values), heads=int(heads.sum()),
            raw_heads_outside=int((np.any(np.abs(values)>1, axis=1)&heads).sum())))
        tracker.table(step, f'samples/geometry-{index}', ['time_ms', 'type', 'x', 'y'],
            [[float(t), int(k), float(p[0]), float(p[1])] for t,k,p in zip(e['times'], e['types'], values)])
    atomic_json(out/'samples.json', dict(reports=reports, checkpoint_sha256=digest(out/'best.pt'),
        scope='Best validation checkpoint, fixed reference rhythm, raw pre-projection coordinates; no playable export or client test'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'prep', 'base', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--variant', choices=VARIANTS, required=True)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--length', type=int, default=256)
    ap.add_argument('--accumulate', type=int, default=4)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--lr', type=float, default=.0001)
    ap.add_argument('--base-lr', type=float, default=.00001)
    ap.add_argument('--plan-weight', type=float, default=.1)
    ap.add_argument('--wandb-project')
    ap.add_argument('--wandb-group')
    ap.add_argument('--wandb-entity')
    ap.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = ap.parse_args()
    if min(args.steps, args.length, args.accumulate, args.eval_every) < 1:
        raise ValueError('Training budgets must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = 'cuda:0'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    noise_rng = torch.Generator().manual_seed(args.seed)
    cm = load_coord_model(args.base, device)
    train = CoordinateDataset(args.data, args.prep, 'train', args.length)
    val = CoordinateDataset(args.data, args.prep, 'validation', args.length)
    if {r['song_group'] for r in train.rows} & {r['song_group'] for r in val.rows}:
        raise ValueError('Coordinate song split overlap')
    diffusion = create_diffusion(timestep_respacing='', diffusion_steps=1000, noise_schedule='squaredcos_cap_v2')
    cfg = MusicContextConfig(object_context=args.variant!='audio-frozen', spacing_plan=args.variant=='spacing-joint')
    recipe = dict(args={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, config=asdict(cfg),
        base_sha256=digest(args.base), data_sha256=digest(args.data/'report.json'),
        splits_sha256={s:digest(args.data/f'{s}.jsonl') for s in ('train', 'validation')},
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        scope='Frozen song-grouped development experiment. Base pretraining overlap unknown. Oracle diagnostic is not an inference candidate.')
    atomic_json(args.out/'recipe.json', recipe)
    tracker = ExperimentTracker(args.out, args.wandb_project, args.wandb_group, args.out.name,
                                recipe, args.wandb_mode, args.wandb_entity)
    def emit(step, record):
        value = dict(step=step, **record)
        with (args.out/'log.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
        tracker.log(step, record)
    baseline = evaluate(cm.net, cm, val, diffusion, device)
    oracle = evaluate(cm.net, cm, val, diffusion, device, oracle=True)
    atomic_json(args.out/'baseline.json', dict(unconditional=baseline, oracle_token_distance=oracle))
    emit(0, dict(baseline=baseline['summary'], oracle_diagnostic=oracle['summary']))
    torch.manual_seed(args.seed)
    model = AudioConditionedCoord(cm.net, cfg).to(device)
    adapter_parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [dict(params=adapter_parameters, lr=args.lr)]
    if args.variant.endswith('joint'):
        frozen = [p for p in model.base.parameters() if not p.requires_grad]
        for parameter in frozen:
            parameter.requires_grad_(True)
        groups.append(dict(params=frozen, lr=args.base_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    best = baseline['summary']['mse']
    # Save the initial state as a valid candidate; failed training cannot silently replace it.
    def save(path, step, metric, state=False):
        record = dict(format='autoosu-coordinate-experiment/1', config=asdict(cfg), model=model.state_dict(),
                      base_sha256=recipe['base_sha256'], variant=args.variant, step=step, metric=metric)
        if state:
            record.update(optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                          noise_rng=noise_rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = path.with_suffix('.partial')
        torch.save(record, temporary)
        temporary.replace(path)
    save(args.out/'best.pt', 0, best)
    started = time.monotonic()
    for step in range(1, args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, plans = [], []
        for _ in range(args.accumulate):
            e = to_device(train.example(int(rng.integers(len(train.rows))), rng), cm, device)
            t = torch.randint(0, 1000, (1,), generator=noise_rng).to(device)
            noise = torch.randn(e['clean'].shape, generator=noise_rng).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = diffusion.training_losses(model, e['clean'], t, model_kwargs=model_kwargs(model, e), noise=noise)['loss'].mean()
                plan = loss*0.
                if cfg.spacing_plan:
                    _, parameters = model.encode_condition(e['c'], e['music'], e['objects'])
                    plan = model.planner.loss(parameters, e['clean'], e['objects'])
                total = loss+args.plan_weight*plan
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite coordinate objective')
            (total/args.accumulate).backward()
            losses.append(float(loss.detach())); plans.append(float(plan.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step%50 == 0 or step==1:
            emit(step, dict(train=dict(diffusion_loss=float(np.mean(losses)), plan_nll=float(np.mean(plans)),
                                     grad_norm=float(norm)), elapsed_s=time.monotonic()-started))
        if step%args.eval_every == 0 or step==args.steps:
            measured = evaluate(model, cm, val, diffusion, device)
            atomic_json(args.out/f'evaluation-{step}.json', measured)
            emit(step, dict(validation=measured['summary'], elapsed_s=time.monotonic()-started))
            tracker.table(step, 'validation/songs', list(measured['rows'][0]), [list(r.values()) for r in measured['rows']])
            metric = measured['summary']['mse']
            if metric < best:
                best = metric
                save(args.out/'best.pt', step, best)
            save(args.out/'state.pt', step, metric, state=True)
    best_state = torch.load(args.out/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best_state['model'])
    sample(model, cm, val, device, args.out, tracker, args.steps)
    emit(args.steps, dict(status='complete', best_mse=best, best_step=best_state['step'],
        improves_base=best_state['step']>0, elapsed_s=time.monotonic()-started))
    tracker.finish()


if __name__ == '__main__':
    main()
