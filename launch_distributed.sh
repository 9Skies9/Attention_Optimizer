#!/bin/bash
# Launch distributed experiment across 4 GPUs
# Each GPU coordinates via a shared state file

set -e

GPUS=4
EXTRA_ARGS=("$@")
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    GPUS="$1"
    EXTRA_ARGS=("${@:2}")
fi
STATE_FILE="./experiment_state.json"

# Remove old state file if exists (fresh start)
rm -f $STATE_FILE

echo "Launching distributed experiment on $GPUS GPUs..."
echo "State file: $STATE_FILE"

for gpu in $(seq 0 $((GPUS-1))); do
    echo "Starting GPU $gpu..."
    CUDA_VISIBLE_DEVICES=$gpu python run_distributed.py --gpu $gpu --state-file $STATE_FILE "${EXTRA_ARGS[@]}" &
done

echo ""
echo "All $GPUS workers launched in background."
echo "Monitor progress with: cat $STATE_FILE"
echo "Or watch with: watch -n 5 'cat $STATE_FILE'"
echo ""
echo "Kill all with: pkill -f run_distributed.py"
