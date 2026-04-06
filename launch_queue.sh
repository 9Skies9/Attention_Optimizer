#!/bin/bash
# Convenience launcher for 4-GPU experiment sweep
# Starts 4 background processes, one per GPU

set -e

GPUS=${1:-4}
echo "Launching experiment sweep across $GPUS GPUs..."

for gpu in $(seq 0 $((GPUS-1))); do
    echo "Starting GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python run_queue.py --gpu $gpu --total-gpus $GPUS &
done

echo "All $GPUS workers launched in background."
echo "Monitor with: ps aux | grep run_queue"
echo "Kill all with: pkill -f run_queue.py"
echo ""
echo "Log files go to: logs/<run_id>/metrics.jsonl"
