# Reproducing the TOFU results

This document describes how to reproduce the TOFU experiments of

> **Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance**
> Gizem Yüce, Giorgos Nikolaou, Nicolas Flammarion (EPFL) — https://arxiv.org/abs/2606.06320

covering environment setup, hardware, hyperparameter tuning, and the final
unlearning + evaluation runs for ATWU and all baselines. (For upstream
open-unlearning's own baseline reproduction, see `docs/repro.md` — this
document is specific to the paper above.) All commands are run from the
repository root.

## 1. Environment

Python ≥ 3.11 (the paper runs used 3.12.7). With a plain venv at the
repository root (`.venv`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install ".[lm-eval,paraphrase-judge,hpsearch]"
# flash-attn note: its installer downloads a prebuilt wheel into $TMPDIR and
# os.rename()s it into pip's cache — this fails with "Invalid cross-device
# link" when /tmp and the cache are on different filesystems. Point TMPDIR
# at the same filesystem as the repo first:
mkdir -p .pip-tmp && TMPDIR=$(pwd)/.pip-tmp pip install --no-build-isolation flash-attn==2.6.3 && rm -rf .pip-tmp
pip install einops  # flash-attn runtime dep; install explicitly in case the wheel step was interrupted

# Required for the DPO baseline: the "I don't know" alternate dataset
# (downloads data/idk.jsonl):
python setup_data.py --idk

# Optional: retain-model evaluation logs — only needed if you run the full
# upstream TOFU suite (eval=[tofu], e.g. for forget_quality); the reproduction
# scripts use the slim eval=[tofu_repro] panel, which doesn't need them:
python setup_data.py --eval
```

(A conda env with `python=3.11` and the same `pip install` lines works
equally well.)

The extras: `lm-eval` (MMLU metric), `paraphrase-judge` (openai + google-genai +
nltk for the paraphrase quality / win-rate judges and the repetitiveness metric),
`hpsearch` (optuna for the tuning script).

**Version note:** the pins above are the *paper-exact* versions
(transformers 4.45.1, tokenizers 0.20.3, lm-eval 0.4.8, nltk 3.9.2) —
evaluating the paper's final checkpoints with this stack reproduces the
recorded metrics bit-for-bit. The code also runs on newer transformers
(≥ 4.46, `processing_class` API) via a compat shim in `src/trainer`, but
training numerics then differ slightly (e.g. transformers 4.51 changes
forget/retain ES by ~1–2% and lm-eval 0.4.11 shifts MMLU by ~2 points).

Create a `.env` file at the repository root:

```bash
HF_TOKEN=hf_...          # gated Llama models + open-unlearning TOFU checkpoints
OPENAI_API_KEY=sk-...    # judge for paraphrase quality + win-rate (gpt-5.4-mini)
```

Everything below is run from the repository root.

## 2. Hardware and cost

- Every TOFU run (tuning trial or final run) uses a **single NVIDIA
  A100-SXM4-80GB** GPU: 10 epochs, per-device batch size 8 with 4
  gradient-accumulation steps (effective batch 32), bf16, `paged_adamw_32bit`,
  weight decay 0.01, 1-epoch linear warmup then linear decay, grad clip 1.0
  (the repository defaults).
- The 1B model fits on much smaller GPUs; the 8B runs were done on A100-80GB.
  Bit-exact reproduction requires the same GPU model.
- The judge-based evals (paraphrase quality, win-rate) are **CPU + OpenAI API**
  (`gpt-5.4-mini`, seed 42). A full final-evaluation pass over all methods costs
  roughly **USD $180** in judge API calls — run `--with-paraphrase` deliberately.

## 3. Data and base models

Downloaded automatically from the HuggingFace Hub on first use:

- `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` and
  `open-unlearning/tofu_Llama-3.1-8B-Instruct_full` — TOFU-finetuned checkpoints.
- `ngiorgos/TOFU` (config `qa_with_paraphrases`) — TOFU with per-question
  paraphrases, used by the paraphrase quality evals.
- `jinzhuoran/RWKU` (config `utility_fluency`) — generation prompts for the
  repetitiveness metric.

Splits: `forget01`/`retain99`, `forget05`/`retain95`, `forget10`/`retain90`.

## 4. Hyperparameter tuning (ATWU)

Two-stage Bayesian search (Optuna TPE), as in Appendix D.2 of the paper:

```bash
python scripts/hpsearch_atwu.py --gpus 0                # phase 1 + phase 2
python scripts/hpsearch_atwu.py --gpus 0 --skip-phase2  # 1B search only
```

- **Phase 1** — 50 trials on Llama-3.2-1B-Instruct / `forget10` over
  `lr, gamma, alpha, beta` (scorer lr fixed at 0.05), objective
  `retain_extraction_strength − extraction_strength` from `TOFU_SUMMARY.json`.
- **Phase 2** — 5-trial lr-only transfer to Llama-3.1-8B-Instruct within
  ±5× of the phase-1 best lr.

The search is resumable (state in `scripts/hpsearch_atwu_pop15.db`); best
parameters are written to `hyperparam/TOFU_pop15/bayesian_summary.json`
(keys `ATWU`, `ATWU_8B`), which the ATWU run scripts read.

The hyperparameters this search selected — and those of all baselines — are
frozen in **`scripts/selected_tofu.json`** (Table 6 of the paper, full
precision). Baseline tuning used the analogous per-method searches described
in Appendix D.2/D.3.

## 5. Win-rate baselines (once per model/split)

The win-rate judge compares an unlearned model's generations against the
split's **retain model** (`tofu_<model>_<retain_split>`). Generate the
baselines first:

```bash
# repeat for each model / split pair (retain99_forget01, retain95_forget05, retain90_forget10)
MODEL=Llama-3.2-1B-Instruct \
MODEL_PATH=open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
OUTPUT_DIR=saves/eval/paraphrase_baselines/Llama-3.2-1B-Instruct/retain90_forget10/paraphrase_evals \
BASELINE_ONLY=1 bash community/methods/ATWU/repetitiveness-winrate-eval.sh
```

## 6. Final runs

### ATWU

The community scripts run unlearning plus the full paraphrase eval pipeline.
They resolve hyperparameters in this order: `$SUMMARY_JSON` (explicit
override) → `hyperparam/TOFU_pop15/bayesian_summary.json` (fresh tuning
output from step 4, if present) → `scripts/selected_tofu.json` (frozen paper
parameters) — so **step 4 is optional**; without it the runs use the exact
parameters reported in the paper:

```bash
bash community/methods/ATWU/tofu-llama3.2-1b-instruct.sh forget10   # also forget01/forget05
bash community/methods/ATWU/tofu-llama3.1-8b-instruct.sh forget10
```

### Any paper method (frozen paper parameters)

`scripts/reproduce_tofu.py` reproduces a final run for any (method, model,
split) triple from the paper, using the selected hyperparameters in
`scripts/selected_tofu.json`:

```bash
# Default eval panel: TOFU forget/retain extraction strength + model utility,
# plus repetitiveness + MMLU
python scripts/reproduce_tofu.py --method NPO  --model 1B --forget-split forget10

# Additionally run the paraphrase quality + win-rate evals (judge API costs!)
python scripts/reproduce_tofu.py --method ATWU --model 8B --forget-split forget05 --with-paraphrase
```

Methods: `GradDiff, DPO, NPO, SimNPO, JensUn, RMU, WGA, SatImp, ETW, ATWU`
(RMU is 8B-only, ETW is 1B-only, matching the paper). Note: the paper's 8B
runs of the seven standard baselines were trained with gradient checkpointing
enabled — it changes backward-pass numerics, so the script applies
`trainer.args.gradient_checkpointing=True` automatically for exactly those
method/model combinations (never for ATWU/RMU/ETW or any 1B run). Models: `1B`
(Llama-3.2-1B-Instruct) and `8B` (Llama-3.1-8B-Instruct). Splits:
`forget01, forget05, forget10`. Useful flags: `--gpu`, `--skip-unlearn`
(re-evaluate an existing checkpoint), `--dry-run` (print the exact commands).

Outputs land in `saves/unlearn/TOFU_repro/<model>/<split>/<method>` (ATWU
community scripts use `saves/unlearn/ATWU_TOFU/...`).

## 7. Consolidating results

```bash
bash scripts/consolidate_atwu.sh   # ATWU runs -> results_archive/final_runs_atwu_short.json
python scripts/consolidate_final_runs_short.py \
    --runs-dir saves/unlearn/TOFU_repro \
    --output results_archive/final_runs_repro_short.json
```

Each consolidated record contains the TOFU summary, paraphrase summary
(quality, repetitiveness, win-rate, MMLU), training stats, and the sanitized
training config.

## 8. End-to-end

The whole pipeline (tuning → win-rate baselines → ATWU final runs → all
baseline final runs → consolidation) is a sequential composition of the steps
above; a local (untracked) `reproduce_paper_tofu.sh` at the repository root
can be assembled directly from them, with one stage per section.
