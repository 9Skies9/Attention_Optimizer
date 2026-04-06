#!/usr/bin/env python3
"""
Distributed experiment runner with shared queue.
4 processes coordinate via a JSON state file to distribute work.

Usage:
    # Terminal 1 (GPU 0)
    python run_distributed.py --gpu 0 --state-file ./experiment_state.json

    # Terminal 2 (GPU 1)
    python run_distributed.py --gpu 1 --state-file ./experiment_state.json

    # Terminal 3 (GPU 2)
    python run_distributed.py --gpu 2 --state-file ./experiment_state.json

    # Terminal 4 (GPU 3)
    python run_distributed.py --gpu 3 --state-file ./experiment_state.json

Or use the launcher:
    bash launch_distributed.sh
"""

import argparse
import json
import os
import subprocess
import time
import fcntl
from pathlib import Path
from configs.runs import RUNS


class ExperimentQueue:
    """File-based distributed queue using fcntl locking."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self._init_state()

    def _init_state(self):
        """Initialize state file if it doesn't exist."""
        if not self.state_file.exists():
            all_run_ids = list(RUNS.keys())
            state = {
                "total_runs": len(all_run_ids),
                "completed": {},
                "in_progress": {},
                "failed": {},
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

    def _read_state(self):
        """Read state with shared lock."""
        with open(self.state_file, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            state = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return state

    def _write_state(self, state):
        """Write state with exclusive lock."""
        with open(self.state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get_next_run(self, gpu_id: int) -> str | None:
        """Atomically claim the next available run."""
        state = self._read_state()
        all_run_ids = list(RUNS.keys())

        # Find first unclaimed run
        for run_id in all_run_ids:
            if run_id not in state["completed"] and run_id not in state["in_progress"]:
                # Claim it
                state["in_progress"][run_id] = gpu_id
                self._write_state(state)
                return run_id

        return None

    def mark_completed(self, run_id: str, success: bool):
        """Mark a run as completed or failed."""
        state = self._read_state()

        if run_id in state["in_progress"]:
            del state["in_progress"][run_id]

        if success:
            state["completed"][run_id] = "success"
        else:
            state["failed"][run_id] = "failed"

        self._write_state(state)

    def get_progress(self) -> tuple[int, int, int]:
        """Return (completed, failed, total)."""
        state = self._read_state()
        return len(state["completed"]), len(state["failed"]), state["total_runs"]

    def print_status(self):
        """Print current status."""
        completed, failed, total = self.get_progress()
        print(
            f"  Progress: {completed}/{total} completed, {failed} failed, {total - completed - failed} remaining"
        )


def run_training(run_id: str, gpu_id: int, extra_args: list[str]) -> bool:
    """Launch training for a single run."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["python", "train.py", "--run_id", run_id]
    if extra_args:
        cmd.extend(extra_args)
    print(f"[GPU {gpu_id}] Starting: {run_id}")
    print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Distributed experiment runner")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID")
    parser.add_argument(
        "--state-file",
        type=str,
        default="./experiment_state.json",
        help="Shared state file",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=5, help="Seconds to wait when queue empty"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Max retries per run"
    )
    parser.add_argument("--max_steps", type=str, default=None)
    parser.add_argument("--max_tokens", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    queue = ExperimentQueue(args.state_file)

    print(f"[GPU {args.gpu}] Started. State file: {args.state_file}")
    queue.print_status()

    runs_completed = 0
    runs_failed = 0

    extra_args = []
    if args.max_steps:
        extra_args.extend(["--max_steps", args.max_steps])
    if args.max_tokens:
        extra_args.extend(["--max_tokens", args.max_tokens])
    if args.checkpoint_every:
        extra_args.extend(["--checkpoint_every", args.checkpoint_every])
    if args.resume_from:
        extra_args.extend(["--resume_from", args.resume_from])

    while True:
        run_id = queue.get_next_run(args.gpu)

        if run_id is None:
            # Check if all done
            completed, failed, total = queue.get_progress()
            if completed + failed >= total:
                print(f"[GPU {args.gpu}] All runs complete!")
                break
            else:
                print(f"[GPU {args.gpu}] Queue empty, waiting {args.poll_interval}s...")
                print(f"[GPU {args.gpu}] Current status: ", end="")
                queue.print_status()
                time.sleep(args.poll_interval)
                continue

        print(
            f"\n[GPU {args.gpu}] === ({runs_completed + runs_failed + 1}/{65}) Running: {run_id} ==="
        )

        for retry in range(args.max_retries + 1):
            if retry > 0:
                print(f"[GPU {args.gpu}] Retry {retry}/{args.max_retries} for {run_id}")

            if run_training(run_id, args.gpu, extra_args):
                queue.mark_completed(run_id, success=True)
                runs_completed += 1
                print(f"[GPU {args.gpu}] SUCCESS: {run_id}")
                break
            else:
                if retry == args.max_retries:
                    queue.mark_completed(run_id, success=False)
                    runs_failed += 1
                    print(
                        f"[GPU {args.gpu}] FAILED after {args.max_retries} retries: {run_id}"
                    )
                else:
                    print(f"[GPU {args.gpu}] Failed, retrying in 5s...")
                    time.sleep(5)

        queue.print_status()

    print(f"\n[GPU {args.gpu}] === FINAL ===")
    print(f"[GPU {args.gpu}] Completed: {runs_completed}, Failed: {runs_failed}")


if __name__ == "__main__":
    main()
