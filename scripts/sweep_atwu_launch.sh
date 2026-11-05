#!/bin/bash
# Launch the ATWU sensitivity + k-fold sweep, sharded one process per GPU.
#
# 25 runs total: 20 exp-1 (rho/beta sensitivity) + 5 exp-2 (k-fold scorer
# transfer). With 5 GPUs each shard gets 4 exp-1 runs + 1 fold, ~1.8 h.
#
# Usage:
#   bash scripts/sweep_atwu_launch.sh                    # 5 shards, GPUs 0-4
#   GPUS=0,1,2,3 bash scripts/sweep_atwu_launch.sh       # 4 shards
#   GPUS=0 bash scripts/sweep_atwu_launch.sh             # everything on GPU 0 (~9 h)
#   EXTRA="--dry-run" bash scripts/sweep_atwu_launch.sh  # print commands only
#   EXTRA="--include-uniform" bash scripts/sweep_atwu_launch.sh   # +1 control run
#
# Shards are independent processes; each skips runs that already have a record,
# so re-running this script resumes an interrupted sweep. Follow progress with
#   tail -f saves/sweep_atwu/logs/shard0_gpu0.log
# and collect results with
#   .venv/bin/python scripts/sweep_atwu_sensitivity.py --collect

set -euo pipefail
cd "$(dirname "$0")/.."

# The repo deps live in .venv; the driver passes its own interpreter down to the
# train/eval subprocesses, so picking it here is enough.
PY=${PY:-}
if [[ -z "$PY" ]]; then
    if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python; fi
fi

GPUS=${GPUS:-0,1,2,3,4}
EXTRA=${EXTRA:-}
SWEEP_ROOT=${ATWU_SWEEP_ROOT:-saves/sweep_atwu}
LOGDIR="$SWEEP_ROOT/logs"
mkdir -p "$LOGDIR"

IFS=',' read -ra gpu_arr <<< "$GPUS"
n=${#gpu_arr[@]}

# Fail early on a GPU that does not exist rather than 2 h into a shard.
if command -v nvidia-smi >/dev/null 2>&1; then
    n_visible=$(nvidia-smi -L | wc -l)
    for g in "${gpu_arr[@]}"; do
        if (( g >= n_visible )); then
            echo "ERROR: GPU $g requested but only $n_visible GPU(s) visible." >&2
            echo "       Set GPUS to the devices you actually have, e.g. GPUS=0" >&2
            exit 1
        fi
    done
else
    echo "WARNING: nvidia-smi not found; not validating GPU ids." >&2
fi

echo "Launching $n shard(s) over GPUs: ${gpu_arr[*]}"
"$PY" scripts/sweep_atwu_sensitivity.py --list --num-shards "$n"

pids=()
for i in "${!gpu_arr[@]}"; do
    g=${gpu_arr[$i]}
    log="$LOGDIR/shard${i}_gpu${g}.log"
    nohup "$PY" scripts/sweep_atwu_sensitivity.py \
        --shard "$i" --num-shards "$n" --gpu "$g" $EXTRA > "$log" 2>&1 &
    pids+=($!)
    echo "  shard $i -> GPU $g (pid ${pids[-1]}, log $log)"
done

echo
echo "Waiting for $n shard(s)..."
status=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        echo "shard $i FAILED (see $LOGDIR)" >&2
        status=1
    fi
done

echo
"$PY" scripts/sweep_atwu_sensitivity.py --collect
exit $status
