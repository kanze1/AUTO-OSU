"""Materialize the registered two-family experiment queue; does not start training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autoosu.ml.train_architecture import digest
from autoosu.provenance import atomic_json
from scripts.run_experiment_queue import validate_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--coord-base', type=Path, required=True)
    ap.add_argument('--coord-data', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.recipe.read_text())
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()[:4]
    if len(gpus)!=4:
        raise ValueError('This recipe requires physical GPUs 0 through 3')
    python = str(args.base/'.venv/bin/python')
    jobs = []
    for seed in spec['seeds']:
        for family in ('rhythm', 'coordinate'):
            cfg = spec[family]
            for variant in cfg['variants']:
                name = f'{family}-{variant}-{seed}'
                out = args.run/'training'/name
                module = 'autoosu.ml.train_architecture' if family=='rhythm' else 'autoosu.ml.train_coord_experiment'
                command = [python, '-u', '-m', module, '--out', str(out), '--seed', str(seed),
                    '--prep', str(args.base/'data/prep'), '--wandb-project', spec['wandb_project'],
                    '--wandb-entity', spec['wandb_entity'], '--wandb-group', spec['group']]
                if family=='rhythm':
                    command += ['--architecture',variant,'--data',str(args.base/'data/rhythm_v1_20261008'),
                                '--attributes',str(args.base/'data/attributes_v1_20261008')]
                else:
                    command += ['--variant',variant,'--data',str(args.coord_data),'--base',str(args.coord_base)]
                for key,value in cfg['training'].items():
                    command += ['--'+key.replace('_','-'), str(value)]
                jobs.append(dict(name=name, family=family, command=command, output=str(out),
                                 gpu_pool=list(gpus[:2] if family=='rhythm' else gpus[2:])))
    manifest = dict(gpus=gpus, source=str(args.source), lock_directory=str(args.base/'runs/gpu-locks'),
        recipe_sha256=digest(args.recipe), source_revision=(args.source/'source-revision.txt').read_text().strip(), jobs=jobs)
    validate_manifest(manifest)
    Path(manifest['lock_directory']).mkdir(exist_ok=True)
    destination = args.run/'queue-manifest.json'
    if destination.exists():
        raise FileExistsError('Queue manifest already exists; do not overwrite a dispatched recipe')
    atomic_json(destination, manifest)
    print(destination)


if __name__=='__main__':
    main()
