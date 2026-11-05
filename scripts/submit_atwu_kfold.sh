#!/usr/bin/env bash
# Submit the leave-one-fold-out unlearning experiment to Run:AI — ONE GPU per shard,
# 10 runs over 5 shards (2 each, ~40 min).
#
# Each run unlearns 4 of the 5 forget10 folds with ATWU running NORMALLY (scorer
# co-trained, nothing frozen), then evaluates all 400 rows. Because the eval writes
# per-example scores, they split into the 320 rows the unlearning saw and the 80 it
# did not — so AUROC answers whether forgetting reached data it never trained on.
#
# Two fold schemes, 5 runs each (forget10 is author-ordered, 20 authors x 20 QA):
#   kfa1..5  hold out 4 whole AUTHORS      -> generalisation to unseen entities
#   kfq1..5  hold out 4 QA rows of EVERY author -> generalisation to unseen questions
#            about entities the unlearning did target
#
# AUROC convention (scripts/kfold_auroc.py, held-out = positive class):
#   0.5 = held-out indistinguishable from unlearned -> forgetting generalised
#   1.0 = perfectly separable -> forgetting stayed on the rows it was trained on
#
# The tofu_full baseline job is the control that makes those numbers readable: on a
# model that was never unlearned the same fold split must give AUROC ~0.5, since the
# folds are arbitrary. It also supplies the pre-unlearning per-example scores for the
# normalised generalisation fraction. Runs on its own GPU, independent of the shards.
#
# Run from the directory that contains run.py (same as your other launchers).
# Override with SCHEME=author|question|both and SHARDS=.
set -euo pipefail

R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEME="${SCHEME:-both}"
SHARDS="${SHARDS:-5}"

# ── Baseline: the un-unlearned model, same per-example metrics as the fold runs ──
python run.py --name "kf-baseline" --gpu 1 \
    "bash $R/job_atwu_sweep.sh --reference-model open-unlearning/tofu_Llama-3.2-1B-Instruct_full --kfold $SCHEME --gpu 0"

# ── Fold runs: one job / GPU each ──
for i in $(seq 0 $((SHARDS - 1))); do
    python run.py --name "kf-shard$i" --gpu 1 \
        "bash $R/job_atwu_sweep.sh --kfold $SCHEME --shard $i --num-shards $SHARDS --gpu 0"
done

# ── AUROC analysis (no GPU; run locally once the runs land) ──
# .venv/bin/python scripts/kfold_auroc.py \
#     --baseline saves/sweep_atwu/reference_model/tofu_Llama-3.2-1B-Instruct_full/TOFU_EVAL.json
