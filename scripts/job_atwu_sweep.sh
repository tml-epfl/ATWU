#!/bin/bash
# ATWU rho/beta sensitivity + 5-fold scorer-transfer sweep job (Run:AI). Same env
# setup as the gencorp jobs: cd into the repo, load .env (HF_TOKEN), activate the
# venv, point HF_HOME at the offline cache. All args are forwarded to
# sweep_atwu_sensitivity.py, so this covers the shards and the reference model:
#   bash scripts/job_atwu_sweep.sh --shard 0 --num-shards 5 --gpu 0
#   bash scripts/job_atwu_sweep.sh --reference-model --gpu 0
#   bash scripts/job_atwu_sweep.sh --only fold3 --gpu 0
#
# No API keys are needed — the win-rate metric is deleted from both eval stages,
# so the whole sweep is GPU-only.
#
# Every run is resumable: one that already has a record JSON is skipped, and a
# fold whose scorer.pt exists skips straight to the unlearning stage. A preempted
# job just needs resubmitting under the same name.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
python_bin="${ATWU_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python}"; fi

exec "$python_bin" scripts/sweep_atwu_sensitivity.py "$@"
