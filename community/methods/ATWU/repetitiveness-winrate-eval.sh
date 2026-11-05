#!/bin/bash
# Repetitiveness + win-rate paraphrase eval for an (ATWU-)unlearned model.
#
# Runs the base `paraphrase` eval suite (repetitiveness + winrate). The winrate
# metric compares the unlearned model's repetitiveness output against the
# split's RETAIN model's repetitiveness output via an LLM judge.
#
# Prerequisites: a baseline repetitiveness file must already exist. Either
#   (a) set BASELINE_PATH to point at it, or
#   (b) leave BASELINE_PATH unset, in which case winrate looks for
#       ${OUTPUT_DIR}/repetitiveness/pretrained.jsonl
# To produce the baseline file, run this script once with MODEL_PATH pointing
# at the RETAIN model for the split (e.g. open-unlearning/tofu_<model>_retain99
# for forget01, _retain90 for forget10) and BASELINE_ONLY=1.
#
# Run from the repository root.
#
# Required:
#   MODEL_PATH    absolute path (or HF id) of the model to evaluate
#
# Optional (env vars):
#   MODEL         architecture config name (default: Llama-3.2-1B-Instruct)
#   TASK_NAME     task name for outputs (default: atwu_repet_winrate)
#   OUTPUT_DIR    eval output dir (default: ${MODEL_PATH}/paraphrase_evals)
#   BASELINE_PATH path to baseline repetitiveness .jsonl (default: see above)
#   BASELINE_ONLY if =1, skip winrate (use to produce the baseline file)
#   GPU           CUDA_VISIBLE_DEVICES (default: 0)

set -euo pipefail

set -a
source .env
set +a

: "${MODEL_PATH:?set MODEL_PATH=/abs/path/to/checkpoint (or HF id)}"

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
TASK_NAME=${TASK_NAME:-atwu_repet_winrate}
OUTPUT_DIR=${OUTPUT_DIR:-${MODEL_PATH}/paraphrase_evals}
BASELINE_PATH=${BASELINE_PATH:-}
BASELINE_ONLY=${BASELINE_ONLY:-0}
SKIP_WINRATE=${SKIP_WINRATE:-0}
GPU=${GPU:-0}

extra=()
if [[ -n "${BASELINE_PATH}" ]]; then
    extra+=("+eval.paraphrase.metrics.winrate.baseline_path=${BASELINE_PATH}")
fi
# Either flag drops the winrate metric. BASELINE_ONLY=1 is for generating
# the pretrained baseline; SKIP_WINRATE=1 is for temporarily disabling
# winrate eval downstream while still computing repetitiveness.
if [[ "${BASELINE_ONLY}" == "1" || "${SKIP_WINRATE}" == "1" ]]; then
    extra+=("~eval.paraphrase.metrics.winrate")
fi

echo "============================================"
echo " Paraphrase eval: repetitiveness + winrate"
echo "   model arch:  ${MODEL}"
echo "   model path:  ${MODEL_PATH}"
echo "   output dir:  ${OUTPUT_DIR}"
echo "   baseline:    ${BASELINE_PATH:-<default: ${OUTPUT_DIR}/repetitiveness/pretrained.jsonl>}"
echo "   baseline-only: ${BASELINE_ONLY}  skip-winrate: ${SKIP_WINRATE}"
echo "============================================"

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${GPU} \
    python src/eval.py \
    experiment=eval/paraphrase/default.yaml \
    'eval=[paraphrase]' \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    +model.model_args.token=${HF_TOKEN} \
    +model.tokenizer_args.token=${HF_TOKEN} \
    ++model.model_args.device_map='auto' \
    paths.output_dir=${OUTPUT_DIR} \
    '~eval.paraphrase.metrics.mmlu' \
    '++eval.paraphrase.metrics.repetitiveness.generation={max_new_tokens:128,do_sample:false,temperature:0.0}' \
    "${extra[@]}"
