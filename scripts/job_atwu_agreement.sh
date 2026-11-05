#!/bin/bash
# Scorer-agreement analysis job (Run:AI). Pairwise agreement between the 5 fold
# scorers (and the co-trained reference scorers) on identical forget10 batches —
# one model forward pass, no training, no gradients. Run this AFTER the 5 fold
# jobs have produced their scorer.pt artifacts; it auto-discovers them.
#
# Same env setup as the sweep job. All args are forwarded to scorer_agreement.py:
#   bash scripts/job_atwu_agreement.sh --gpu 0
#   bash scripts/job_atwu_agreement.sh --gpu 0 --scorers fold1=path/a.pt fold2=path/b.pt
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
python_bin="${ATWU_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python}"; fi

exec "$python_bin" scripts/scorer_agreement.py "$@"
