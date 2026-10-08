"""Linux handoff from a successful architecture dispatcher to convergence training.

Waits on the exact owned controller process. A failed original run, occupied GPU,
or failed continuation is reported and never automatically retried.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import select
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--original-run", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), owner=os.getuid(), stage="checking")
    def update(**changes):
        state.update(changes, observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        temporary = args.out / "dispatch.partial"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.out / "dispatch.json")
        print(json.dumps(state), flush=True)
    try:
        original = json.loads((args.original_run / "dispatch.json").read_text())
        pid = original["pid"]
        process = Path(f"/proc/{pid}")
        if process.exists():
            handle = os.pidfd_open(pid)
            try:
                if process.stat().st_uid != os.getuid() or str(args.original_run / "launch_training.py").encode() not in (process / "cmdline").read_bytes().split(b"\0"):
                    raise RuntimeError("Original controller identity does not match")
                update(stage="waiting_for_original_training", original_controller_pid=pid,
                       original_run=str(args.original_run))
                select.select([handle], [], [])
            finally:
                os.close(handle)
        original = json.loads((args.original_run / "dispatch.json").read_text())
        if original["stage"] != "training_complete_pending_acceptance" or original["final"]["exit_code"] != 0:
            raise RuntimeError("Original training did not finish successfully; continuation not started")
        uuids = original["gpu_uuids"]
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                   check=True, capture_output=True, text=True).stdout.splitlines()
        if any(line.split(",")[0].strip() in uuids for line in processes):
            raise RuntimeError("Selected GPUs have active processes; continuation not started")
        command = [str(args.base / ".venv/bin/python"), "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", "--nproc-per-node=4", "-m", "autoosu.ml.train_convergence",
                   "--initialize-from", str(args.original_run / "final/last.pt"),
                   "--data", str(args.base / "data/rhythm_v1_20261008"), "--prep", str(args.base / "data/prep"),
                   "--attributes", str(args.base / "data/attributes_v1_20261008"), "--out", str(args.out / "training")]
        with (args.out / "training.stdout.log").open("w") as log:
            child = subprocess.Popen(command, cwd=args.source,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(uuids)}, stdout=log, stderr=subprocess.STDOUT)
        update(stage="training", child_pid=child.pid, command=command, gpu_uuids=uuids)
        code = child.wait()
        if code:
            raise RuntimeError(f"Continuation exited with status {code}")
        last = json.loads((args.out / "training/log.jsonl").read_text().splitlines()[-1])
        if last.get("event") != "finished":
            raise RuntimeError("Continuation lacks a terminal training record")
        update(stage=last["status"], exit_code=code, final=last)
    except Exception as error:
        update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
