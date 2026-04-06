#!/bin/bash
# Launch distributed experiment across 4 GPUs
# Each GPU coordinates via a shared state file

set -e

GPUS=${1:-4}
STATE_FILE="./experiment_state.json"

# Remove old state file if exists (fresh start)
rm -f $STATE_FILE

echo "Launching distributed experiment on $GPUS GPUs..."
echo "State file: $STATE_FILE"

for gpu in $(seq 0 $((GPUS-1))); do
    echo "Starting GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python run_distributed.py --gpu $gpu --state-file $STATE_FILE &
done

echo ""
echo "All $GPUS workers launched in background."
echo "Monitor progress with: cat $STATE_FILE"
echo "Or watch with: watch -n 5 'cat $STATE_FILE'"
echo ""
echo "Kill all with: pkill -f run_distributed.py"
