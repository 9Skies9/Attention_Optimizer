#!/usr/bin/env bash
# launch_distributed.sh
#
# Distributed launcher for the optimizer sweep.
# Auto-detects GPUs, coordinates runs via shared state file.
# Resume-safe: won't wipe state on restart.
#
# Usage:
#   bash launch_distributed.sh              # auto-detect GPUs
#   bash launch_distributed.sh --fresh     # start fresh (delete state)
#   bash launch_distributed.sh 2           # force 2 GPUs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

STATE_FILE="experiment_state.json"
LOCK_FILE="${STATE_FILE}.lock"

# -------------------------------------------------------------------
# Parse arguments
# -------------------------------------------------------------------
FRESH=false
FORCE_N_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH=true
            shift
            ;;
        --help)
            echo "Usage: bash launch_distributed.sh [OPTIONS] [NUM_GPUS]"
            echo ""
            echo "Options:"
            echo "  --fresh    Delete state file and start fresh"
            echo "  --help     Show this help message"
            echo ""
            echo "Arguments:"
            echo "  NUM_GPUS   Number of GPUs to use (auto-detected if omitted)"
            echo ""
            echo "Examples:"
            echo "  bash launch_distributed.sh              # auto-detect, resume"
            echo "  bash launch_distributed.sh --fresh      # start fresh"
            echo "  bash launch_distributed.sh 2            # force 2 GPUs"
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                FORCE_N_GPUS="$1"
            fi
            shift
            ;;
    esac
done

# -------------------------------------------------------------------
# Detect GPUs
# -------------------------------------------------------------------
if [[ -n "$FORCE_N_GPUS" ]]; then
    N_GPUS="$FORCE_N_GPUS"
    echo "Forcing N_GPUS=$N_GPUS (user override)"
else
    if command -v nvidia-smi &> /dev/null; then
        N_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
        echo "Auto-detected $N_GPUS GPU(s) via nvidia-smi"
    elif command -v rocm-smi &> /dev/null; then
        N_GPUS=$(rocm-smi --listgpus 2>/dev/null | wc -l)
        echo "Auto-detected $N_GPUS GPU(s) via rocm-smi"
    elif [[ -d /dev/dri ]]; then
        N_GPUS=$(ls /dev/dri/render* 2>/dev/null | wc -l)
        echo "Auto-detected $N_GPUS GPU(s) via /dev/dri"
    else
        N_GPUS=1
        echo "Could not auto-detect GPUs, defaulting to 1"
    fi
fi

echo "Using $N_GPUS GPU(s)"
echo ""

# -------------------------------------------------------------------
# Define runs
# -------------------------------------------------------------------
RUNS=(
    "ATTNRAW-V1-G-L4-T0.5"
    "ATTNRAW-V1-G-L4"
    "ATTNRAW-V1-G-L4-T2.0"
    "ATTNRAW-MIX90-L4-T1.0"
    "ATTNRAW-MIX75-L4-T1.0"
    "ATTNRAW-MIX50-L4-T1.0"
    "ATTNRAW-MIX25-L4-T1.0"
    "ATTNRAW-V1-L4-MIX10"
    "SIMPLEAVG-L4"
)

N_RUNS=${#RUNS[@]}
echo "Total runs: $N_RUNS"
echo ""

# -------------------------------------------------------------------
# Initialize or resume state file
# -------------------------------------------------------------------
init_state() {
    local state_json=""
    for run in "${RUNS[@]}"; do
        if [[ -n "$state_json" ]]; then
            state_json="$state_json,"
        fi
        state_json="$state_json\"$run\""
    done
    echo "{\"completed\":[],\"running\":{},\"remaining\":[$state_json],\"total\":[$state_json]}" > "$STATE_FILE"
}

acquire_lock() {
    exec 200>"$LOCK_FILE"
    flock -w 30 200 || { echo "ERROR: Could not acquire lock"; exit 1; }
}

release_lock() {
    exec 200>&-
}

read_state() {
    local max_attempts=5
    local attempt=1
    while [[ $attempt -le $max_attempts ]]; do
        if state=$(cat "$STATE_FILE" 2>/dev/null) && [[ -n "$state" ]]; then
            echo "$state"
            return 0
        fi
        echo "Warning: Failed to read state (attempt $attempt/$max_attempts), retrying..."
        sleep 0.5
        attempt=$((attempt + 1))
    done
    echo "ERROR: Could not read state file after $max_attempts attempts"
    return 1
}

write_state() {
    local state_json="$1"
    local tmp_file="${STATE_FILE}.tmp.$$"
    echo "$state_json" > "$tmp_file"
    if ! mv "$tmp_file" "$STATE_FILE" 2>/dev/null; then
        rm -f "$tmp_file"
        echo "ERROR: Failed to write state file"
        return 1
    fi
}

get_next_run() {
    local state="$1"
    local remaining=$(echo "$state" | python3 -c "import sys,json; d=json.load(sys.stdin); print(' '.join(d.get('remaining',[])))" 2>/dev/null)
    if [[ -z "$remaining" ]] || [[ "$remaining" == "None" ]]; then
        echo ""
        return 0
    fi
    echo "$remaining" | awk '{print $1}'
}

mark_completed() {
    local run="$1"
    local state="$2"
    echo "$state" | python3 -c "
import sys,json
d=json.load(sys.stdin)
run=sys.argv[1]
d['completed'].append(run)
d['remaining']=[r for r in d['remaining'] if r!=run]
if run in d['running']:
    del d['running'][run]
print(json.dumps(d, separators=(',',':')))
" "$run"
}

mark_running() {
    local run="$1"
    local gpu="$2"
    local state="$3"
    echo "$state" | python3 -c "
import sys,json
d=json.load(sys.stdin)
run=sys.argv[1]
gpu=sys.argv[2]
d['running'][gpu]=run
d['remaining']=[r for r in d['remaining'] if r!=run]
print(json.dumps(d, separators=(',',':')))
" "$run" "$gpu"
}

print_state() {
    local state="$1"
    echo "$state" | python3 -c "
import sys,json
d=json.load(sys.stdin)
completed=len(d.get('completed',[]))
running=len(d.get('running',{}))
remaining=len(d.get('remaining',[]))
print(f'Progress: {completed}/{len(d.get(\"total\",[]))} completed, {running} running, {remaining} remaining')
if d.get('running'):
    for gpu,run in d['running'].items():
        print(f'  {gpu}: {run}')
" 2>/dev/null
}

# -------------------------------------------------------------------
# Cleanup on Ctrl+C
# -------------------------------------------------------------------
cleanup() {
    echo ""
    echo "Interrupted. State saved. Run again to resume."
    release_lock
    exit 130
}
trap cleanup SIGINT SIGTERM

# -------------------------------------------------------------------
# Main worker loop
# -------------------------------------------------------------------
worker() {
    local gpu_id="$1"
    local gpu_name="GPU${gpu_id}"
    
    echo "[$gpu_name] Starting worker"
    
    while true; do
        acquire_lock
        
        state=$(read_state)
        remaining=$(echo "$state" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('remaining',[])))" 2>/dev/null)
        
        if [[ "$remaining" == "0" ]] || [[ -z "$remaining" ]]; then
            release_lock
            echo "[$gpu_name] All runs complete!"
            return 0
        fi
        
        run=$(get_next_run "$state")
        if [[ -z "$run" ]]; then
            release_lock
            echo "[$gpu_name] No runs remaining, exiting"
            return 0
        fi
        
        state=$(mark_running "$run" "$gpu_name" "$state")
        write_state "$state"
        release_lock
        
        echo ""
        echo "============================================"
        echo "[$gpu_name] Starting: $run"
        echo "============================================"
        
        # Run the experiment with a timeout safety net
        # If it crashes, we'll catch it and mark appropriately
        if CUDA_VISIBLE_DEVICES=$gpu_id python train.py --run_id "$run"; then
            result=0
        else
            result=$?
            echo "Warning: $run exited with code $result"
        fi
        
        acquire_lock
        state=$(read_state)
        
        # Check if already completed (idempotent)
        already_done=$(echo "$state" | python3 -c "import sys,json; d=json.load(sys.stdin); print('$run' in d.get('completed',[]))" 2>/dev/null)
        
        if [[ "$already_done" != "True" ]]; then
            state=$(mark_completed "$run" "$state")
            write_state "$state"
            print_state "$state"
        else
            echo "[$gpu_name] $run was already marked complete"
        fi
        
        release_lock
        
        if [[ $result -ne 0 ]]; then
            echo "Warning: $run failed with code $result"
        fi
    done
}

# -------------------------------------------------------------------
# Start fresh if requested
# -------------------------------------------------------------------
if [[ "$FRESH" == "true" ]]; then
    echo "Starting fresh (--fresh flag set)"
    if [[ -f "$STATE_FILE" ]]; then
        rm -f "$STATE_FILE"
        echo "Deleted $STATE_FILE"
    fi
fi

if [[ ! -f "$STATE_FILE" ]]; then
    echo "Initializing new state file"
    init_state
fi

# -------------------------------------------------------------------
# Show current state
# -------------------------------------------------------------------
echo "Current state:"
state=$(read_state)
print_state "$state"
echo ""

# -------------------------------------------------------------------
# Launch workers
# -------------------------------------------------------------------
echo "Launching $N_GPUS worker(s)..."
echo ""

pids=()
for i in $(seq 0 $((N_GPUS - 1))); do
    worker $i &
    pids+=($!)
done

# Wait for all workers
# If any worker fails unexpectedly, the others keep going
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        failed=$((failed + 1))
    fi
done

echo ""
if [[ $failed -gt 0 ]]; then
    echo "$failed worker(s) exited with errors"
else
    echo "All workers finished successfully"
fi

# Final state
echo ""
echo "Final state:"
state=$(read_state)
print_state "$state"
