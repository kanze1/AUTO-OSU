"""Frozen comparisons of v0 preference controls; keep songs and maps in out/.

Freeze the policy on development songs before evaluating a fresh reserved set.
These objective response checks do not replace independent human playtesting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from autoosu.provenance import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(args):
    from autoosu import __version__
    from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
    from autoosu.preferences import pattern_metrics
    corpus=read(args.corpus/'corpus.json')
    songs=[s for s in corpus['songs'] if s['partition']==args.partition
           and (not args.groups or any(s['group'].startswith(p) for p in args.groups))]
    if not songs:
        raise ValueError('No matching songs')
    settings=dict(seed=240924,decode_steps=12,coord_steps=100,temperature=.9,cfg_scale=1.,device='cuda',star_rating=6.5)
    core=['autoosu/generate.py','autoosu/controls.py','autoosu/preferences.py','autoosu/metrics.py',
          'autoosu/ml/sample.py','autoosu/ml/coord_infer.py','autoosu/rhythm.py','autoosu/watermark.py']
    recipe=dict(version=__version__,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        settings=settings,partition=args.partition,timing=args.timing,target=args.target,
        groups=[s['group'] for s in songs],corpus_sha256=digest(args.corpus/'corpus.json'),
        code_sha256={p:digest(ROOT/p) for p in core},
        models={k:digest(ROOT/f'models/{k}_v0.pt') for k in ('rhythm','coord')},
        environment=dict(python=platform.python_version(),packages={n:version(n) for n in ('torch','numpy','scipy','librosa','rosu-pp-py')}),
        modes=['balanced','jumps','streams'],highlight='default auto',
        scope='Song-grouped preference response; training overlap and human skill labels unverified')
    recipe_path=args.out/'recipe.json'
    if recipe_path.exists() and read(recipe_path)!=recipe:
        raise ValueError('Recipe changed; choose a fresh output directory')
    atomic_json(recipe_path,recipe)
    for name, sha in recipe['code_sha256'].items():
        data=(ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==sha
        snapshot=args.out/'code'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(data)
    rows=read(args.out/'results.json') if (args.out/'results.json').exists() else []
    done={(r['group'],r['mode']) for r in rows}
    cache={}
    for song in songs:
        audio=args.corpus/song['audio']
        assert digest(audio)==song['audio_sha256'],'Frozen audio changed'
        for mode in recipe['modes']:
            if (song['group'],mode) in done:
                continue
            options=dict(skill_preference=mode,target_stars=args.target)
            started=time.perf_counter()
            result=generate(audio,['Insane'],args.out/'generated'/song['group']/mode,
                bpm=song['bpm'] if args.timing=='reference' else None,
                offset_ms=song['offset_ms']+OSU_TIMING_SHIFT_MS if args.timing=='reference' else None,
                rhythm_model=str(ROOT/'models/rhythm_v0.pt'),coord_model=str(ROOT/'models/coord_v0.pt'),
                controls=options,model_cache=cache,records_dir=args.out/'records',log=lambda *_:None,**settings)
            diff=result.diffs[0]
            path=result.osz.with_name('map.osu')
            with zipfile.ZipFile(result.osz) as z:
                path.write_bytes(z.read(next(n for n in z.namelist() if n.endswith('.osu'))))
            patterns=pattern_metrics(diff.beatmap,result.timing.beat_length)
            rows.append(dict(group=song['group'],mode=mode,summary=diff.summary(),measurement=diff.measurement,
                diagnostics=diff.diagnostics,patterns=patterns,elapsed_s=time.perf_counter()-started,
                bpm=result.timing.bpm,offset_ms=result.timing.offset_ms,sha256=digest(path),
                path=path.relative_to(args.out).as_posix(),osz=result.osz.relative_to(args.out).as_posix()))
            atomic_json(args.out/'results.json',rows)
            print(f"{len(rows)}/{len(songs)*3} {song['group'][:8]} {mode}: {diff.measurement['stars']:.3f} stars, "
                  f"jumps {patterns['jump_fraction']:.3f}, tapping {patterns['tapping_fraction']:.3f}, "
                  f"{diff.control_report.get('preference',{}).get('status','baseline')}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=ROOT/'out/difficulty-evaluation')
    parser.add_argument('--partition',choices=['development','heldout'],default='development')
    parser.add_argument('--timing',choices=['reference','automatic'],default='reference')
    parser.add_argument('--groups',nargs='+')
    parser.add_argument('--target',type=float)
    main(parser.parse_args())
