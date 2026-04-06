#!/usr/bin/env python3
"""
Run queue manager for distributed experiment Sweeps across 4 separate GPUs.
Each GPU runs its own process and picks the next available run_id.

Usage:
    # Start queue manager on GPU 0
    python run_queue.py --gpu 0 --master-port 29500 &

    # Start queue manager on GPU 1
    python run_queue.py --gpu 1 --master-port 29500 &

    # etc.

Or use the convenience launcher:
    bash launch_queue.sh
"""

import argparse
import os
import subprocess
import time
import socket
from configs.runs import RUNS


def get_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_next_run(gpu_id, port):
    """Query the master for the next run_id."""
    # In a simple setup, each GPU just picks from its assigned slice
    # For true coordination, you'd use a Redis queue or similar
    pass


def get_gpu_assigned_runs(gpu_id, total_gpus, all_run_ids):
    """Assign runs to this GPU in a round-robin fashion."""
    return [run_id for i, run_id in enumerate(all_run_ids) if i % total_gpus == gpu_id]


def run_training(run_id, gpu_id):
    """Launch training for a single run."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["python", "train.py", "--run_id", run_id]
    print(f"[GPU {gpu_id}] Starting: {run_id}")
    print(f"[GPU {gpu_id}] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run queue manager for GPU experiment sweeps"
    )
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use")
    parser.add_argument(
        "--total-gpus", type=int, default=4, help="Total number of GPUs"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Max retries per run"
    )
    args = parser.parse_args()

    all_run_ids = list(RUNS.keys())
    my_runs = get_gpu_assigned_runs(args.gpu, args.total_gpus, all_run_ids)

    print(f"[GPU {args.gpu}] Assigned {len(my_runs)} runs")
    print(f"[GPU {args.gpu}] Runs: {my_runs}")

    success_count = 0
    fail_count = 0

    for i, run_id in enumerate(my_runs):
        print(f"\n[GPU {args.gpu}] === Run {i + 1}/{len(my_runs)}: {run_id} ===")

        for retry in range(args.max_retries + 1):
            if retry > 0:
                print(f"[GPU {args.gpu}] Retry {retry}/{args.max_retries} for {run_id}")

            if run_training(run_id, args.gpu):
                success_count += 1
                break
            else:
                if retry == args.max_retries:
                    print(
                        f"[GPU {args.gpu}] FAILED after {args.max_retries} retries: {run_id}"
                    )
                    fail_count += 1
                else:
                    print(f"[GPU {args.gpu}] Failed, retrying in 10s...")
                    time.sleep(10)

    print(f"\n[GPU {args.gpu}] === DONE ===")
    print(f"[GPU {args.gpu}] Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    main()
