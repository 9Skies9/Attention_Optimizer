#!/usr/bin/env bash
# launchers/launch_single.sh
#
# Run a single optimizer-study experiment by run ID.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash launchers/launch_single.sh <RUN_ID>"
    echo ""
    echo "Available run IDs:"
    echo "  Baselines: BASE-SGD BASE-ADAM BASE-MUON"
    echo "  History:   AVG-8 AVG-8R ATTNRAW-8 ATTNRAW-8R"
    exit 1
fi

RUN_ID="$1"
echo "Running: $RUN_ID"
python train.py --run_id "$RUN_ID"
