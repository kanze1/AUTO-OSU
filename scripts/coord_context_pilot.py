"""Bounded architecture probe, not full coordinate training or release acceptance.

Uses existing coordinate tokenization, zero-distance inference conditions, and a
frozen coord v0. Compare local audio with local + whole-song audio under equal steps.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'coord'))

import numpy as np
import rosu_pp_py as rosu
from slider import Beatmap
from slider.beatmap import Slider
import torch

from autoosu.ml.coord import create_diffusion
from autoosu.ml.coord_context import AudioConditionedCoord, MusicContextConfig, music_features
from autoosu.ml.coord_objects import object_layout, complete_object_crops
from autoosu.ml.coord_infer import Sequence, load_coord_model, sequence_context
from autoosu.ml.sample import song_mel_uint8
from autoosu.provenance import atomic_json
from osu_diffusion.utils.data_loading import get_data
from scripts.audition_skill_models import digest


def prepare(record, device, length):
    path = Path(record['map'])
    bm = Beatmap.parse(path.read_text(encoding='utf-8-sig'))
    # Strict: do not use the legacy wrapper that silently skips failed objects.
    parts=[]
    for obj in bm.hit_objects(stacking=False):
        part=get_data(obj)
        if isinstance(obj,Slider) and int(part[0,3:].argmax()) not in (4,5):
            raise ValueError('Legacy tokenizer cannot represent this slider; refusing silent conversion')
        parts.append(part)
    seq = torch.cat(parts, dim=0).T
    if seq.shape[1] < length:
        raise ValueError('Map is shorter than the declared crop length')
    red = [p for p in bm.timing_points if p.parent is None]
    if not red or any(abs(p.ms_per_beat - red[0].ms_per_beat) > 1e-5 for p in red):
        raise ValueError('Pilot supports constant-tempo sources only')
    beat_ms, offset = red[0].ms_per_beat, red[0].offset.total_seconds() * 1000
    mel = song_mel_uint8(record['audio'])
    sr = rosu.Difficulty().calculate(rosu.Beatmap(path=str(path))).stars
    examples = []
    for start,stop in complete_object_crops(seq[2].numpy(),seq[3:].argmax(0).numpy(),length):
        crop = seq[:, start:stop]
        times = crop[2].numpy()
        types = crop[3:].argmax(0).numpy()
        context = sequence_context(Sequence(times, types, []))[None].to(device)
        music = {k:v[None].to(device) for k,v in music_features(mel, times, beat_ms, offset).items()}
        clean = (crop[:2] / torch.tensor([[512.], [384.]]) * 2 - 1)[None].to(device)
        objects={k:v.to(device) for k,v in object_layout(times,types).items()}
        examples.append(dict(name=record['name'],start=int(start),clean=clean,c=context,music=music,
                             objects=objects,stars=sr,cs=bm.circle_size))
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--length',type=int,default=128)
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--variants',nargs='+',choices=['local-only','local-and-song','object-aware'],default=['local-only','local-and-song','object-aware'])
    ap.add_argument('--seeds',type=int,nargs='+',default=[20261008])
    ap.add_argument('--milestones',type=int,nargs='*',default=[])
    args=ap.parse_args()
    if args.steps < 1 or args.length < 2:
        raise ValueError('Positive steps and at least two coordinate tokens are required')
    records=json.loads(args.recipe.read_text(encoding='utf-8'))
    if {r['split'] for r in records} != {'train','validation'}:
        raise ValueError('Recipe must contain train and validation sources')
    hashes={}
    for r in records:
        for key in ('audio','map'):
            value=digest(r[key])
            if value != r[key+'_sha256']:
                raise ValueError('Frozen source hash mismatch')
        h=r['audio_sha256']
        if h in hashes and hashes[h] != r['split']:
            raise ValueError('Audio identity crosses the pilot split')
        hashes[h]=r['split']
    args.out.mkdir(parents=True,exist_ok=False)
    base_hash=digest(args.base)
    atomic_json(args.out/'recipe.json',dict(sources=records,base_sha256=base_hash,steps=args.steps,length=args.length,
        learning_rate=.0002,seeds=args.seeds,variants=args.variants,milestones=args.milestones,script_sha256=digest(__file__),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                      [ROOT/'autoosu/ml/coord_context.py',ROOT/'autoosu/ml/coord_objects.py',ROOT/'coord/osu_diffusion/utils/data_loading.py']},
        scope='Development architecture probe only; base pretraining overlap unknown. No release-quality claim.'))
    examples={split:[] for split in ('train','validation')}
    for r in records:
        print('PREPARE',r['name'],flush=True)
        examples[r['split']] += prepare(r,args.device,args.length)
    atomic_json(args.out/'crops.json',{split:[dict(song=e['name'],start=e['start'],tokens=e['clean'].shape[-1],
        objects=len(e['objects']['heads'])) for e in values] for split,values in examples.items()})
    diffusion=create_diffusion(timestep_respacing='',diffusion_steps=1000,noise_schedule='squaredcos_cap_v2')
    eval_bank=[]
    generator=torch.Generator(device='cpu').manual_seed(1949)
    for e in examples['validation']:
        for timestep in (100,500,900):
            noise=torch.randn(e['clean'].shape,generator=generator).to(args.device)
            eval_bank.append((e,timestep,noise))
    results=[]
    for name,seed in [(name,seed) for seed in args.seeds for name in args.variants]:
        torch.manual_seed(seed)
        cm=load_coord_model(args.base,args.device)
        for split in examples:
            for e in examples[split]:
                e['y']=cm.class_vector(e['stars'],e['cs'])[None].to(args.device)

        @torch.no_grad()
        def evaluate(model,shift_audio=False):
            rows=[]
            model.eval()
            for e,timestep,noise in eval_bank:
                t=torch.tensor([timestep],device=args.device)
                x=diffusion.q_sample(e['clean'],t,noise=noise)
                kw=dict(c=e['c'],y=e['y'])
                if isinstance(model,AudioConditionedCoord):
                    music=e['music']
                    if shift_audio:
                        music=dict(music,local=torch.roll(music['local'],len(music['local'][0])//2,1),
                            song=torch.roll(music['song'],len(music['song'][0])//2,1))
                    kw['music']=music
                    kw['objects']=e['objects']
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                    predicted=model(x,t,**kw)[:,:2].float()
                token_error=(predicted-noise).square().mean(dim=1)[0]
                owners=e['objects']['owners']
                object_error=torch.zeros(len(e['objects']['heads']),device=args.device).scatter_add_(0,owners,token_error)
                object_error=object_error/torch.bincount(owners,minlength=len(object_error))
                rows.append(dict(song=e['name'],crop=e['start'],timestep=timestep,mse=float(token_error.mean()),
                                 object_mse=float(object_error.mean())))
            return dict(mean_mse=float(np.mean([r['mse'] for r in rows])),
                        mean_object_mse=float(np.mean([r['object_mse'] for r in rows])),rows=rows)

        baseline=evaluate(cm.net)
        cfg=MusicContextConfig(whole_song=name!='local-only',object_context=name=='object-aware')
        torch.manual_seed(seed)
        model=AudioConditionedCoord(cm.net,cfg).to(args.device)
        warm=evaluate(model)
        warm_difference=max(abs(a['mse']-b['mse']) for a,b in zip(baseline['rows'],warm['rows']))
        if warm_difference != 0:
            raise AssertionError('Zero audio branch changed pretrained predictions')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
        trace, milestones=[], []
        generator=torch.Generator(device='cpu').manual_seed(seed)
        begin=time.monotonic()
        for step in range(args.steps):
            model.train()
            e=examples['train'][step%len(examples['train'])]
            t=torch.randint(0,1000,(1,),generator=generator).to(args.device)
            noise=torch.randn(e['clean'].shape,generator=generator).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
                losses=diffusion.training_losses(model,e['clean'],t,
                    model_kwargs=dict(c=e['c'],y=e['y'],music=e['music'],objects=e['objects']),noise=noise)
                loss=losses['loss'].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pilot loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            optimizer.step()
            trace.append(dict(step=step+1,loss=float(loss.detach()),grad_norm=float(norm)))
            if (step+1)%64==0:
                print('TRAIN',name,seed,trace[-1],flush=True)
            if step+1 in args.milestones:
                milestones.append(dict(step=step+1,validation=evaluate(model)))
        measured=evaluate(model)
        shifted=evaluate(model,shift_audio=True)
        model.save_adapter(args.out/f'{name}-{seed}.pt',base_hash)
        # Actual pure-noise inference on a real source crop, separately from denoising loss.
        # This checks sampler integration, not slider fitting, export or playability.
        e=examples['validation'][0]
        sampler=create_diffusion(timestep_respacing=[100],diffusion_steps=1000,noise_schedule='squaredcos_cap_v2')
        torch.manual_seed(5678)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.device.startswith('cuda')):
            positions=sampler.p_sample_loop(model,e['clean'].shape,
                noise=torch.randn_like(e['clean']),clip_denoised=True,
                model_kwargs=dict(c=e['c'],y=e['y'],music=e['music'],objects=e['objects']),device=args.device)
        if positions.shape != e['clean'].shape or not torch.isfinite(positions).all():
            raise AssertionError('Invalid pure-noise coordinate sample')
        # The vendored diffusion clips x0 to [-2, 2], not the [-1, 1] in its
        # inherited docstring. Record pre-projection geometry rather than hiding it.
        types=e['c'][0,256:].argmax(0)
        heads=(types==0)|(types==1)|(types==4)|(types==5)
        outside=positions[0].abs().gt(1).any(0)
        sample_path=args.out/f'{name}-{seed}-positions.npy'
        np.save(sample_path,positions.float().cpu().numpy())
        row=dict(name=name,seed=seed,config=asdict(cfg),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            steps=args.steps,elapsed_s=time.monotonic()-begin,warm_start_max_mse_difference=warm_difference,
            baseline=baseline,validation=measured,shifted_audio=shifted,train_trace=trace,milestones=milestones,
            sampling=dict(source=e['name'],crop=e['start'],steps=100,shape=list(positions.shape),
                finite=True,normalized_range=[float(positions.min()),float(positions.max())],
                raw_heads_outside_playfield=int((outside&heads).sum()),raw_tokens_outside_playfield=int(outside.sum()),
                sha256=digest(sample_path),scope='Raw coordinate tokens only; no slider fitting, osu export or playtest'))
        results.append(row)
        atomic_json(args.out/'results.json',results)
        print('RESULT',name,baseline['mean_mse'],measured['mean_mse'],shifted['mean_mse'],flush=True)
        del model,cm,optimizer
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
