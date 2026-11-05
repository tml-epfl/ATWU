#!/bin/bash
# ATWU — Unlearn + Paraphrase Eval (TOFU)
# Model: Llama-3.1-8B-Instruct | forget split passed as $1 (forget01|forget05|forget10)
#
# Usage:
#   bash community/methods/ATWU/tofu-llama3.1-8b-instruct.sh forget10
#   GPU=1 bash community/methods/ATWU/tofu-llama3.1-8b-instruct.sh forget05
#
# The win-rate step (stage 3) compares against the RETAIN model's repetitiveness
# baseline (e.g. tofu_<model>_retain90 for forget10). Generate it once per
# model/split with:
#   MODEL_PATH=open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90 \
#   OUTPUT_DIR=saves/eval/paraphrase_baselines/Llama-3.1-8B-Instruct/retain90_forget10/paraphrase_evals \
#   BASELINE_ONLY=1 bash community/methods/ATWU/repetitiveness-winrate-eval.sh
# or point WINRATE_BASELINE at an existing baseline .jsonl.

set -a
source .env
set +a

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# ── Args ──
forget_split="${1:-forget10}"
case "$forget_split" in
    forget01) retain_split="retain99"; holdout_split="holdout01" ;;
    forget05) retain_split="retain95"; holdout_split="holdout05" ;;
    forget10) retain_split="retain90"; holdout_split="holdout10" ;;
    *) echo "Unknown forget_split: $forget_split (expected forget01|forget05|forget10)"; exit 1 ;;
esac

# ── Constants ──
model="Llama-3.1-8B-Instruct"
GPU=${GPU:-0}
num_epochs=10
batch_size=8
grad_accum=4
input_dim=4096

# ── Hyperparameter source ──
# Priority: $SUMMARY_JSON (explicit) > fresh tuning output (scripts/hpsearch_atwu.py)
# > frozen paper params (scripts/selected_tofu.json, Table 6) — so running the
# tuning first is optional.
TUNED_JSON="hyperparam/TOFU_pop15/bayesian_summary.json"
if [[ -z "${SUMMARY_JSON:-}" ]]; then
    if [[ -f "$TUNED_JSON" ]]; then
        SUMMARY_JSON="$TUNED_JSON"
    else
        SUMMARY_JSON="scripts/selected_tofu.json"
    fi
fi
echo "Reading hyperparameters from: ${SUMMARY_JSON}"
HP_KEY="ATWU_8B"
MODEL_SIZE="8B"

get_param() {
    python3 - "$SUMMARY_JSON" "$1" "$2" "$MODEL_SIZE" <<'PY'
import json, sys
path, key, param, size = sys.argv[1:5]
d = json.load(open(path))
if "methods" in d:  # scripts/selected_tofu.json (frozen paper params)
    m = d["methods"][key.removesuffix("_8B")]
    print(m[f"lr_{size}"] if param == "lr" else m[param])
else:               # hyperparam/*/bayesian_summary.json (tuning output)
    print(d[key]["best_params"][param])
PY
}

# ── Hyperparameters (override via env vars) ──
ATWU_lr=${ATWU_lr:-$(get_param $HP_KEY lr)}
ATWU_gamma=${ATWU_gamma:-$(get_param $HP_KEY gamma)}
ATWU_alpha=${ATWU_alpha:-$(get_param $HP_KEY alpha)}
ATWU_beta=${ATWU_beta:-$(get_param $HP_KEY beta)}
ATWU_scorer_lr=${ATWU_scorer_lr:-$(get_param $HP_KEY scorer_lr)}

update_every_n_steps=${UPDATE_EVERY_N_STEPS:-5}
TASK_SUFFIX=""
if [[ -n "${UPDATE_EVERY_N_STEPS:-}" ]]; then
    TASK_SUFFIX="_uev${UPDATE_EVERY_N_STEPS}"
fi
task_name="ATWU_TOFU/${model}/${forget_split}/ATWU${TASK_SUFFIX}"
model_output="saves/unlearn/${task_name}"
winrate_baseline="${WINRATE_BASELINE:-saves/eval/paraphrase_baselines/${model}/${retain_split}_${forget_split}/paraphrase_evals/repetitiveness/model.jsonl}"

# ══════════════════════════════════════════
#  1. Unlearn
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " ATWU — Unlearn (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=ATWU \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_${model}_full \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    forget_split=$forget_split \
    retain_split=$retain_split \
    trainer.args.num_train_epochs=$num_epochs \
    trainer.args.eval_on_start=false \
    trainer.args.eval_strategy=no \
    trainer.args.per_device_train_batch_size=$batch_size \
    trainer.args.gradient_accumulation_steps=$grad_accum \
    trainer.args.learning_rate=$ATWU_lr \
    trainer.method_args.gamma=$ATWU_gamma \
    trainer.method_args.alpha=$ATWU_alpha \
    trainer.method_args.beta=$ATWU_beta \
    trainer.method_args.scorer.cfg.input_dimension=$input_dim \
    trainer.method_args.scorer_trainer.optim_cfg.lr=$ATWU_scorer_lr \
    trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=$update_every_n_steps \
    +trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear \
    trainer.method_args.scorer_trainer.lambda_entropy=1 \
    trainer.method_args.scorer_trainer.lambda_population=15 \
    trainer.method_args.scorer_trainer.budget=0.2 \
    trainer.method_args.scorer_trainer.lambda_l2=1

# ══════════════════════════════════════════
#  2. Paraphrase Generation (GPU)
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " ATWU — Paraphrase Generation (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=$GPU \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=400 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    '++eval.paraphrase.metrics.repetitiveness.generation={max_new_tokens:128,do_sample:false,temperature:0.0}' \
    +eval.paraphrase.metrics.forget_quality.generation_only=true \
    +eval.paraphrase.metrics.retain_quality.generation_only=true \
    '~eval.paraphrase.metrics.winrate' \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals

# ══════════════════════════════════════════
#  3. Paraphrase Eval (API judging, no GPU)
# ══════════════════════════════════════════
echo ""
echo "============================================"
echo " ATWU — Paraphrase Judging (${forget_split})"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES="" \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase_tofu]' \
    model=$model \
    task_name=$task_name \
    model.model_args.pretrained_model_name_or_path=$(pwd)/${model_output} \
    +model.model_args.token=$HF_TOKEN \
    +model.tokenizer_args.token=$HF_TOKEN \
    ++model.model_args.device_map='auto' \
    eval.paraphrase.forget_split=$forget_split \
    eval.paraphrase.retain_split=$retain_split \
    eval.paraphrase.holdout_split=$holdout_split \
    eval.paraphrase.metrics.forget_quality.max_samples=400 \
    eval.paraphrase.metrics.retain_quality.max_samples=400 \
    eval.paraphrase.metrics.forget_quality.datasets.TOFU_para_forget_eval.args.num_train_paraphrases=10 \
    eval.paraphrase.metrics.retain_quality.datasets.TOFU_para_retain_eval.args.num_train_paraphrases=10 \
    +eval.paraphrase.metrics.winrate.baseline_path=${winrate_baseline} \
    +judge_only=true \
    paths.output_dir=$(pwd)/${model_output}/paraphrase_evals
