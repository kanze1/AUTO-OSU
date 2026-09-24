"""Generate a frozen 7/8/9-star listening/playtesting set from the local Q1 corpus.

These previously reviewed development songs are audition examples, not a new
independent quality benchmark. Audio and beatmaps stay under the ignored out/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from autoosu import __version__
from autoosu.provenance import atomic_json, fingerprint


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_set(args):
    from autoosu.controls import validate_controls
    from autoosu.generate import generate, OSU_TIMING_SHIFT_MS

    corpus=read(args.corpus/'corpus.json')
    songs=[next(s for s in corpus['songs'] if s['group'].startswith(prefix)) for prefix in args.songs]
    for target in args.targets:
        validate_controls(dict(target_stars=target))
    settings=dict(seed=240924,decode_steps=12,coord_steps=100,temperature=.9,cfg_scale=1.,device='cuda')
    recipe=dict(version=__version__,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        settings=settings,targets=args.targets,groups=[s['group'] for s in songs],timing='reference + export offset',
        manual_first_song='35%-65% of original audio, same region as previous preferred C version',
        corpus_sha256=digest(args.corpus/'corpus.json'),
        code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted((ROOT/'autoosu').rglob('*.py'))},
        models={k:digest(ROOT/f'models/{k}_v0.pt') for k in ('rhythm','coord')},
        environment=dict(python=platform.python_version(),packages={n:version(n) for n in ('torch','numpy','scipy','librosa','rosu-pp-py')}),
        scope='Known development songs for maintainer audition, not an independent evaluation')
    path=args.out/'recipe.json'
    if path.exists() and read(path)!=recipe:
        raise ValueError('Recipe changed; choose a fresh output directory')
    atomic_json(path,recipe)
    rows=read(args.out/'results.json') if (args.out/'results.json').exists() else []
    done={(r['group'],r['mode'],r['target']) for r in rows}
    cache={}
    for song_index,song in enumerate(songs):
        audio=args.corpus/song['audio']
        assert digest(audio)==song['audio_sha256'],'Frozen audio changed'
        for mode in (('auto','manual') if song_index==0 else ('auto',)):
            for target in args.targets:
                if (song['group'],mode,target) in done:
                    continue
                # Omit highlight_mode for auto to exercise the new public default.
                controls=dict(target_stars=target)
                if mode=='manual':
                    controls.update(highlight_mode='manual',highlights=[dict(start_s=round(song['duration_s']*.35,2),
                                                                          end_s=round(song['duration_s']*.65,2))])
                started=time.perf_counter()
                result=generate(audio,['Insane'],args.out/'generated'/song['group']/f'{mode}-{target:g}',
                    bpm=song['bpm'],offset_ms=song['offset_ms']+OSU_TIMING_SHIFT_MS,
                    rhythm_model=str(ROOT/'models/rhythm_v0.pt'),coord_model=str(ROOT/'models/coord_v0.pt'),
                    controls=controls,model_cache=cache,records_dir=args.out/'records',log=lambda *_:None,**settings)
                diff=result.diffs[0]
                assert diff.control_report['highlight_plan']['mode']==mode
                map_path=result.osz.with_name('map.osu')
                with zipfile.ZipFile(result.osz) as archive:
                    map_path.write_bytes(archive.read(next(n for n in archive.namelist() if n.endswith('.osu'))))
                rows.append(dict(group=song['group'],mode=mode,target=target,summary=diff.summary(),measurement=diff.measurement,
                    diagnostics=diff.diagnostics,elapsed_s=time.perf_counter()-started,
                    path=map_path.relative_to(args.out).as_posix(),osz=result.osz.relative_to(args.out).as_posix(),sha256=digest(map_path)))
                atomic_json(args.out/'results.json',rows)
                print(f"{len(rows)} {song['group'][:8]} {mode} target {target:g}: {diff.measurement['stars']:.3f} stars "
                      f"({'met' if diff.control_report['target_met'] else 'missed'})",flush=True)


def package(args):
    from autoosu.metrics import measure_difficulty
    songs=read(args.corpus/'corpus.json')['songs']
    rows=read(args.out/'results.json')
    packs=args.out/'playtest';packs.mkdir(exist_ok=True)
    manifest=[]
    for index,prefix in enumerate(args.songs,1):
        song=next(s for s in songs if s['group'].startswith(prefix))
        meta={}
        for line in (args.corpus/song['reference']).read_text(encoding='utf-8-sig').splitlines():
            if line.startswith(('Title:','Artist:')):
                key,value=line.split(':',1);meta[key]=value
        title=f'AUTO-OSU High {index:02d} - '+meta['Title']
        path=packs/f'{index:02d}-{prefix}-high.osz'
        assets={}
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
            for row in (r for r in rows if r['group']==song['group']):
                assert digest(args.out/row['path'])==row['sha256'],'Generated map changed'
                with zipfile.ZipFile(args.out/row['osz']) as src:
                    for name in src.namelist():
                        if name.endswith(('.osu','.json')):
                            continue
                        data=src.read(name)
                        if name in assets:
                            assert hashlib.sha256(data).hexdigest()==assets[name]
                        else:
                            dst.writestr(name,data);assets[name]=hashlib.sha256(data).hexdigest()
                    original=src.read(next(n for n in src.namelist() if n.endswith('.osu')))
                stars=row['measurement']['stars']
                label=f"{row['mode'].title()} target {row['target']:g} - measured {stars:.2f}"
                text=original.decode('utf-8-sig').replace('\r\r\n','\n').replace('\r\n','\n')
                for key,value in dict(Title=title,TitleUnicode=title,Artist=meta['Artist'],ArtistUnicode=meta['Artist'],
                    Creator='AUTO-OSU Review',Version=label,BeatmapID='0',BeatmapSetID='-1').items():
                    text=re.sub(r'^'+key+':.*$',lambda m,k=key,v=value:k+':'+v,text,flags=re.M)
                copy=text.encode('utf-8')
                assert fingerprint(copy)['content_sha256']==fingerprint(original)['content_sha256']
                assert abs(measure_difficulty(text)['stars']-stars)<1e-8
                dst.writestr(label+'.osu',copy)
                manifest.append(dict(pack=path.name,filename=label+'.osu',group=row['group'],title=meta['Title'],
                    mode=row['mode'],target=row['target'],stars=stars,target_met=row['summary']['controls']['target_met'],
                    original_sha256=row['sha256'],review_sha256=hashlib.sha256(copy).hexdigest(),
                    watermark=row['summary']['watermark']['status']))
        print(path.name,'packaged',flush=True)
    atomic_json(args.out/'review-manifest.json',dict(schema='autoosu.review-copy/1',maps=manifest,
        note='Metadata-only review copies; original generation manifests omitted because raw hashes changed'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['generate','package'])
    parser.add_argument('--out',type=Path,default=ROOT/'out/high-difficulty-rc1')
    parser.add_argument('--corpus',type=Path,default=ROOT/'out/difficulty-evaluation')
    parser.add_argument('--songs',nargs='+',default=['1b03caae','193214fe','10483e0c','02f892e2'])
    parser.add_argument('--targets',type=float,nargs='+',default=[7.,8.,9.])
    args=parser.parse_args()
    {'generate':generate_set,'package':package}[args.action](args)
