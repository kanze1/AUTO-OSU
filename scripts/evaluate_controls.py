"""Frozen development ablations and one-time reserved-song acceptance generation.

No human highlight labels are inferred. Manual regions are fixed central song
intervals for response testing. Sources and generated audio/maps remain in out/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from autoosu.provenance import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def variants(song, partition):
    manual=dict(highlight_mode="manual",highlights=[dict(start_s=round(song["duration_s"]*.35,2),
                                                        end_s=round(song["duration_s"]*.65,2))])
    if partition=="heldout":
        return [("target65",dict(target_stars=6.5,highlight_mode="legacy")),("auto",dict(highlight_mode="auto"))]
    return [("baseline",dict(highlight_mode="legacy")),("spacing85",dict(spacing_scale=.85,highlight_mode="legacy")),
            ("spacing100",dict(spacing_scale=1.,highlight_mode="legacy")),
            ("spacing120",dict(spacing_scale=1.2,highlight_mode="legacy")),("target65",dict(target_stars=6.5,highlight_mode="legacy")),
            ("manual_density",dict(manual,highlight_spacing=False)),("manual",manual),
            ("auto",dict(highlight_mode="auto"))]


def run(args):
    from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
    from autoosu.controls import region_metrics

    corpus=read(args.corpus/"corpus.json")
    songs=[s for s in corpus["songs"] if s["partition"]==args.partition]
    settings=dict(seed=240924,decode_steps=12,coord_steps=100,temperature=.9,cfg_scale=1.,device="cuda",star_rating=6.5)
    recipe=dict(partition=args.partition,settings=settings,corpus_sha256=digest(args.corpus/"corpus.json"),
                models={kind:digest(ROOT/f"models/{kind}_v0.pt") for kind in ("rhythm","coord")},
                inference_code={name:digest(ROOT/name) for name in (
                    "autoosu/generate.py","autoosu/controls.py","autoosu/ml/sample.py","autoosu/ml/coord_infer.py",
                    "autoosu/rhythm.py","autoosu/metrics.py","autoosu/watermark.py")},
                source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                variants={s["group"]:[(n,o) for n,o in variants(s,args.partition) if not args.variants or n in args.variants] for s in songs},
                timing=args.timing,human_labels=False)
    recipe_path=args.out/f"{args.partition}-recipe.json"
    # Compare the JSON form so tuple/list spelling never silently changes a recipe.
    recipe=json.loads(json.dumps(recipe))
    if recipe_path.exists() and read(recipe_path)!=recipe:
        raise ValueError("Recipe changed; choose another output directory, never tune on reserved-song results")
    atomic_json(recipe_path,recipe)
    results_path=args.out/f"{args.partition}-results.json"
    rows=read(results_path) if results_path.exists() else []
    completed={(r["group"],r["variant"]) for r in rows}
    cache={}
    for song in songs:
        audio=args.corpus/song["audio"]
        if digest(audio)!=song["audio_sha256"]:
            raise ValueError("Frozen audio changed")
        for name,options in variants(song,args.partition):
            if args.variants and name not in args.variants:
                continue
            if (song["group"],name) in completed:
                continue
            started=time.perf_counter()
            result=generate(audio,["Insane"],args.out/"generated"/args.partition/song["group"]/name,
                            bpm=song["bpm"] if args.timing=="reference" else None,
                            offset_ms=song["offset_ms"]+OSU_TIMING_SHIFT_MS if args.timing=="reference" else None,
                            rhythm_model=str(ROOT/"models/rhythm_v0.pt"),coord_model=str(ROOT/"models/coord_v0.pt"),
                            controls=options,model_cache=cache,records_dir=args.out/"records",log=lambda *_:None,**settings)
            diff=result.diffs[0]
            path=result.osz.with_name("map.osu")
            with zipfile.ZipFile(result.osz) as archive:
                path.write_bytes(archive.read(next(n for n in archive.namelist() if n.endswith(".osu"))))
            probe=variants(song,"development")[6][1]["highlights"]
            rows.append(dict(group=song["group"],variant=name,summary=diff.summary(),measurement=diff.measurement,
                             timing=dict(bpm=result.timing.bpm,offset_ms=result.timing.offset_ms),
                             diagnostics=diff.diagnostics,probe_region=region_metrics(diff.beatmap,diff.measurement,probe,26),
                             elapsed_s=time.perf_counter()-started,path=path.relative_to(args.out).as_posix(),
                             osz=result.osz.relative_to(args.out).as_posix(),sha256=digest(path)))
            atomic_json(results_path,rows)
            print(f"{len(rows)} {song['group'][:8]} {name}: {diff.measurement['stars']:.3f} stars, "
                  f"{len(diff.control_report['candidates'])} candidates",flush=True)


def report(args):
    import numpy as np
    rows=read(args.out/f"{args.partition}-results.json")
    groups=sorted({r["group"] for r in rows})
    for row in rows:
        if digest(args.out/row["path"])!=row["sha256"]:
            raise ValueError("Output changed since generation")
    summary={}
    for variant in sorted({r["variant"] for r in rows}):
        selected=[r for r in rows if r["variant"]==variant]
        summary[variant]=dict(maps=len(selected),median_stars=float(np.median([r["summary"]["measured_stars"] for r in selected])),
            target_met=sum(r["summary"]["controls"]["target_met"] is True for r in selected),
            invalid=sum(any(r["diagnostics"][k] for k in ("overlapping_objects","heads_outside_playfield","invalid_sliders","simultaneous_or_reversed_starts")) for r in selected),
            proposed_highlights=sum(bool(r["summary"]["controls"]["highlight_plan"]["regions"]) for r in selected),
            median_elapsed_s=float(np.median([r["elapsed_s"] for r in selected])))
    response=[]
    for group in groups:
        by={r["variant"]:r for r in rows if r["group"]==group}
        response.append(dict(group=group,stars={k:r["summary"]["measured_stars"] for k,r in by.items()},
                             spacing={k:r["diagnostics"]["spacing_p50_px"] for k,r in by.items()},
                             central_region={k:r["probe_region"] for k,r in by.items()}))
    output=dict(recipe=read(args.out/f"{args.partition}-recipe.json"),summary=summary,responses=response,
                limits="Timing mode is recorded in the recipe. Automatic locations are proposals without human labels; no player-quality claim.")
    atomic_json(args.out/f"{args.partition}-report.json",output)
    print(json.dumps(summary,indent=2),flush=True)


def curves(args):
    """Independent parse of every output and sampled paths for flagged anchors."""
    from importlib.metadata import version
    import numpy as np
    from slider import Beatmap
    from slider.beatmap import Slider

    rows=read(args.out/f"{args.partition}-results.json")
    reports=[]
    for row in rows:
        path=args.out/row["path"]
        if digest(path)!=row["sha256"]:
            raise ValueError("Output changed since generation")
        objects=Beatmap.parse(path.read_text(encoding="utf-8-sig")).hit_objects(stacking=False)
        if len(objects)!=row["diagnostics"]["objects"]:
            raise ValueError("Independent parser object count differs")
        flagged=[o for o in objects if isinstance(o,Slider)
                 and any(not(0<=p.x<=512 and 0<=p.y<=384) for p in o.curve.points)]
        outside=[]
        for obj in flagged:
            points=np.asarray([tuple(obj.curve(float(t))) for t in np.linspace(0,1,1001)])
            if not np.isfinite(points).all() or np.any(points<[-.01,-.01]) or np.any(points>[512.01,384.01]):
                outside.append(dict(time_ms=obj.time.total_seconds()*1000,
                                    minimum=points.min(0).tolist(),maximum=points.max(0).tolist()))
        reports.append(dict(group=row["group"],variant=row["variant"],sha256=row["sha256"],
                            flagged_sliders=len(flagged),paths_outside=outside))
    result=dict(parser="slider",version=version("slider"),maps=len(rows),stacking=False,
                flagged_sliders=sum(r["flagged_sliders"] for r in reports),samples_per_curve=1001,
                paths_outside=sum(len(r["paths_outside"]) for r in reports),rows=reports,
                limits="Flagged-anchor curves only; sampled center paths, not exact extrema or client playtesting.")
    atomic_json(args.out/f"{args.partition}-curves.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["run","report","curves"])
    parser.add_argument("--out",type=Path,default=ROOT/"out/generation-controls")
    parser.add_argument("--corpus",type=Path,default=ROOT/"out/difficulty-evaluation")
    parser.add_argument("--partition",choices=["development","heldout"],default="development")
    parser.add_argument("--timing",choices=["reference","automatic"],default="reference")
    parser.add_argument("--variants",nargs="+",choices=["baseline","spacing85","spacing100","spacing120","target65","manual_density","manual","auto"])
    args=parser.parse_args()
    globals()[args.action](args)
