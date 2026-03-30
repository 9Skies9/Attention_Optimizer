#!/usr/bin/env bash
# launchers/launch_local.sh
#
# Runs the optimizer-study experiment matrix sequentially on one GPU.
# Good fit for a single Vast.ai instance.

set -e

RUN_IDS=(
    BASE-SGD
    BASE-ADAM
    BASE-MUON
    AVG-8
    AVG-8R
    ATTNRAW-8
    ATTNRAW-8R
)

echo "Starting sequential training of ${#RUN_IDS[@]} runs..."
echo ""

for RUN_ID in "${RUN_IDS[@]}"; do
    echo "============================================"
    echo " Starting: $RUN_ID"
    echo "============================================"
    python train.py --run_id "$RUN_ID"
    echo ""
    echo " Done: $RUN_ID"
    echo ""
done

echo "All runs complete."
