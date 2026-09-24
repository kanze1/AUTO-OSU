"""Q2.1 density/condition screen against the frozen 0.3.0.dev2 generator.

First export commit 2d2e0bd to out/q2-screen/code using git archive, keeping the
current checkout unchanged. Requires the existing frozen Q1 corpus and weights.
Use --out to create a fresh experiment; completed results are never overwritten.
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code",type=Path,default=ROOT/"out/q2-screen/code")
parser.add_argument("--out",type=Path,default=ROOT/"out/q2-density-response")
args=parser.parse_args()
if not (args.code/"autoosu/generate.py").is_file() or (args.out/"recipe.json").exists():
    raise ValueError("Supply the frozen source and a new experiment directory")
sys.path.insert(0,str(args.code.resolve()))
from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
from autoosu.provenance import atomic_json
from autoosu import __version__
if __version__!="0.3.0.dev2":
    raise ValueError("This screen requires the archived 0.3.0.dev2 source")
corpus=json.loads((ROOT/"out/difficulty-evaluation/corpus.json").read_text())
variants=[("baseline",6.5,None),("density8",6.5,8.),("density12",6.5,12.),("density16",6.5,16.),("condition8",8.,None)]
settings=dict(seed=240924,decode_steps=12,coord_steps=100,temperature=.9,cfg_scale=1.,device="cuda")
recipe=dict(commit="2d2e0bd570acc45a9f3d8211a970e0623667c3ce",variants=variants,settings=settings,
            code_sha256={p.relative_to(args.code).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((args.code/"autoosu").rglob("*.py"))},
            models={k:hashlib.sha256((ROOT/f"models/{k}_v0.pt").read_bytes()).hexdigest() for k in ("rhythm","coord")},
            corpus_sha256=hashlib.sha256((ROOT/"out/difficulty-evaluation/corpus.json").read_bytes()).hexdigest())
atomic_json(args.out/"recipe.json",recipe)
rows=[]; cache={}
for song in (s for s in corpus["songs"] if s["partition"]=="development"):
    for name,star,density in variants:
        start=time.perf_counter()
        result=generate(ROOT/"out/difficulty-evaluation"/song["audio"],["Insane"],args.out/"generated"/song["group"]/name,
                        star_rating=star,density=density,bpm=song["bpm"],offset_ms=song["offset_ms"]+OSU_TIMING_SHIFT_MS,
                        rhythm_model=str(ROOT/"models/rhythm_v0.pt"),coord_model=str(ROOT/"models/coord_v0.pt"),
                        model_cache=cache,records_dir=args.out/"records",log=lambda *_:None,**settings)
        diff=result.diffs[0]
        path=result.osz.with_name("map.osu")
        with zipfile.ZipFile(result.osz) as archive:
            path.write_bytes(archive.read(next(n for n in archive.namelist() if n.endswith(".osu"))))
        rows.append(dict(group=song["group"],variant=name,summary=diff.summary(),diagnostics=diff.diagnostics,
                         measurement=diff.measurement,elapsed=time.perf_counter()-start,
                         path=str(path.relative_to(args.out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        atomic_json(args.out/"results.json",rows)
        print(len(rows),song["group"][:8],name,round(diff.measurement["stars"],3),flush=True)
