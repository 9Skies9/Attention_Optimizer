#!/bin/bash
# Simple launcher: run a single run on a specific GPU
# Usage: ./run_single.sh <gpu_id> <run_id> [extra args]
# Example: ./run_single.sh 0 MUON --max_tokens 2000000000

set -e

GPU=$1
RUN_ID=$2
shift 2

if [ -z "$GPU" ] || [ -z "$RUN_ID" ]; then
    echo "Usage: $0 <gpu_id> <run_id> [extra args]"
    echo "Example: $0 0 MUON --max_tokens 2000000000"
    echo ""
    echo "Available runs:"
    python -c "
from configs.runs import RUNS
for i, run_id in enumerate(RUNS.keys()):
    print(f'  {run_id}')
"
    exit 1
fi

source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU
python train.py --run_id "$RUN_ID" "$@"
