"""Run a finite Linux GPU queue exactly once, with durable status and no failed-job retries."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autoosu.provenance import atomic_json


def occupied_gpus():
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True).stdout
    return {line.split(',')[0].strip() for line in output.splitlines() if line.strip()}


def validate_manifest(manifest):
    jobs = manifest['jobs']
    if not jobs or len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Queue job names must be nonempty and unique')
    if len(set(manifest['gpus'])) != len(manifest['gpus']):
        raise ValueError('Duplicate queue GPUs')
    for job in jobs:
        if not job['command'] or not set(job['gpu_pool']) <= set(manifest['gpus']):
            raise ValueError('Invalid command or GPU pool')
        if not job['gpu_pool']:
            raise ValueError('Job has no GPU lane')


def read_terminal(path):
    if not path.exists() or not path.stat().st_size:
        return {}
    try:
        return json.loads(path.read_text().splitlines()[-1])
    except json.JSONDecodeError as error:
        return dict(status='invalid_terminal_record', error=str(error))


def main():
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    args.out.mkdir(parents=True, exist_ok=False)
    locks = []
    for gpu in manifest['gpus']:
        lock = (Path(manifest['lock_directory'])/f'{gpu}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    occupied = occupied_gpus() & set(manifest['gpus'])
    if occupied:
        raise RuntimeError(f'Selected GPUs already occupied: {sorted(occupied)}')
    state = dict(pid=os.getpid(), owner=os.getuid(), stage='running', gpu_uuids=manifest['gpus'],
                 jobs=[dict(j, status='pending') for j in manifest['jobs']])
    children, handles = {}, {}
    def update():
        state['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(args.out/'dispatch.json', state)
    def stopped(signum, frame):
        state['stage'] = 'stopped_by_user'
        for gpu, (child, job, fd) in children.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                job['status'] = 'stop_requested'
        update()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    update()
    while any(j['status']=='pending' for j in state['jobs']) or children:
        for gpu in manifest['gpus']:
            if gpu in children:
                continue
            pending = next((j for j in state['jobs'] if j['status']=='pending' and gpu in j['gpu_pool']), None)
            if pending is None:
                continue
            if gpu in occupied_gpus():
                for job in state['jobs']:
                    if job['status']=='pending' and gpu in job['gpu_pool']:
                        job['gpu_pool'].remove(gpu)
                        if not job['gpu_pool']:
                            job['status']='blocked_gpu_occupied'
                update()
                continue
            with (args.out/f"{pending['name']}.log").open('x') as log:
                child = subprocess.Popen(pending['command'], cwd=manifest['source'],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES':gpu, 'PYTHONUNBUFFERED':'1',
                         'OMP_NUM_THREADS':'2', 'WANDB_CONSOLE':'off'},
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            fd = os.pidfd_open(child.pid)
            children[gpu] = (child, pending, fd)
            handles[fd] = gpu
            pending.update(status='running', pid=child.pid, gpu=gpu,
                           started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            update()
        if not children:
            break
        ready, _, _ = select.select(list(handles), [], [])
        for fd in ready:
            gpu = handles.pop(fd)
            child, job, _ = children.pop(gpu)
            code = child.wait()
            os.close(fd)
            # Exit zero alone is insufficient: require the trainer's terminal record and W&B identity.
            out = Path(job['output'])
            log_path = out/'log.jsonl'
            terminal = read_terminal(log_path)
            valid = code==0 and terminal.get('status')=='complete' and (out/'wandb-run.json').exists()
            job.update(status='complete' if valid else 'failed', exit_code=code, terminal=terminal,
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if (out/'wandb-run.json').exists():
                job['wandb'] = json.loads((out/'wandb-run.json').read_text())
            update()
    state['stage'] = 'complete_pending_review' if all(j['status']=='complete' for j in state['jobs']) else 'finished_with_failures'
    comparison = []
    for job in state['jobs']:
        row = dict(name=job['name'], family=job['family'], status=job['status'], wandb=job.get('wandb'))
        evaluations = []
        for path in sorted(Path(job['output']).glob('evaluation-*.json')):
            ev = json.loads(path.read_text())
            evaluations.append((int(path.stem.split('-')[-1]), ev))
        if evaluations:
            key = (lambda item:-item[1]['gen_onset_f1']) if job['family']=='rhythm' else (lambda item:item[1]['summary']['mse'])
            step, best = min(evaluations, key=key)
            row.update(best_trained_step=step, metrics={k:v for k,v in best.items() if k not in ('rows','examples')})
        comparison.append(row)
    atomic_json(args.out/'comparison.json', dict(runs=comparison,
        scope='Development metrics only; compare families separately. Coordinate initial base may beat every trained checkpoint. No automatic publication or replacement.'))
    update()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
