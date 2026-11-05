#!/usr/bin/env bash
# Submit the ATWU rho/beta sensitivity + 5-fold scorer-transfer sweep to Run:AI —
# ONE GPU per shard, 5 shards covering 25 runs (TOFU forget10, Llama-3.2-1B-Instruct,
# seed 0). Each shard gets 4 exp-1 runs + exactly 1 fold, ~1.8 h:
#
#   shard 0: reference rho0.6 beta0 beta5  fold1     shard 3: rho0.4 rho0.9 beta3 beta9  fold4
#   shard 1: rho0.1 rho0.7 beta1 beta6     fold2     shard 4: rho0.5 rho1   beta4 beta10 fold5
#   shard 2: rho0.3 rho0.8 beta2 beta8     fold3
#
# EXP 1 (20 runs): rho in {0.1..1.0} and beta in {0..10}, one 1-D sweep each through
# the paper operating point (rho=0.2, beta=7), which belongs to both grids and so is
# run once. Shard 0 runs it FIRST and prints a per-metric delta against the on-disk
# anchor run — if that diverges, kill the sweep before the other 24 runs commit.
#
# EXP 2 (5 runs): forget10's 400 rows split into 5 author-disjoint folds of 80 (4
# authors each — verified by scripts/check_tofu_folds.py). Per fold: fit the scorer
# on that fold alone with the model FROZEN, then unlearn ALL of forget10 with that
# scorer frozen. Tests whether the token scorer has to be relearned per forget set.
#
# Metrics: forget/retain extraction strength + MMLU + repetitiveness. No API keys
# needed (win-rate is deleted), so nothing here can stall on a judge.
#
# Every run is resumable — one that already has a record JSON is skipped, and a fold
# whose scorer.pt exists skips straight to the unlearning stage. A preempted job just
# needs resubmitting under the same name. Model weights (2.4 GB/run) are deleted once
# a run's metrics are recorded; scorer.pt, logs and metric JSONs are kept.
#
# Run from the directory that contains run.py (same as your other launchers).
# Override the shape with SHARDS=/EXTRA= env (EXTRA="--include-uniform" adds the
# frozen-uniform control as a 26th run, on shard 0).
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARDS="${SHARDS:-5}"
EXTRA="${EXTRA:-}"

# ── Reference model (gold lines for the plots): retain90 evaluated through the
#    identical two eval stages as the sweep runs. ~16 min, independent of the
#    shards, skipped if its record already exists. ──
python run.py --name "sens-refmodel" --gpu 1 "bash $R/job_atwu_sweep.sh --reference-model --gpu 0"

# ── Sweep shards: one job / GPU each ──
for i in $(seq 0 $((SHARDS - 1))); do
    python run.py --name "sens-shard$i" --gpu 1 \
        "bash $R/job_atwu_sweep.sh --shard $i --num-shards $SHARDS --gpu 0 $EXTRA"
done

# ── Scorer agreement (1 GPU; run AFTER the 5 fold jobs land) ──
# python run.py --name "sens-agreement" --gpu 1 "bash $R/job_atwu_agreement.sh --gpu 0"

# ── Results table + figures (no GPU; run locally once the shards land) ──
# .venv/bin/python scripts/sweep_atwu_sensitivity.py --collect
# .venv/bin/python scripts/plot_atwu_sensitivity.py      # needs: .venv/bin/pip install matplotlib
