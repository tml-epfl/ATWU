# ATWU — Alternating Token-Weighted Unlearning

- Paper: *Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance* — Gizem Yüce⋆, Giorgos Nikolaou⋆, Nicolas Flammarion (EPFL). https://arxiv.org/abs/2606.06320

ATWU is an unlearning method that reweights the per-token forget loss with a small learned scorer. A one-layer MLP maps the model's last-layer hidden states to per-token importance weights in [0, 1]; the forget objective is the negated, scorer-weighted, saturation-dampened NLL (the saturation term `exp(-nll)^(beta * score)` suppresses gradient ascent on tokens that are already forgotten), combined with a standard retain NLL:

```
L = alpha * retain_NLL - gamma * weighted_forget_NLL
```

The scorer is co-trained during unlearning by a dedicated `ScorerTrainerATWU` (matched forget objective plus entropy, population-budget, and L2 penalties on the scores), updated every `update_every_n_steps` model optimizer steps with its own SGD optimizer and (optionally) a linear schedule.

Key code:

- Trainer: `src/trainer/unlearn/ATWU.py` (config: `configs/trainer/ATWU.yaml`)
- Scorer trainer: `src/trainer/unlearn/Scorer.py`
- Scorer: `src/scorers/LearnedScorer.py` (base classes in `src/scorers/base.py`, forward cache in `src/data/Cache.py`)
- Loss: `reweighted_NLL` in `src/trainer/utils.py`

The related **ETW** baseline (`src/trainer/unlearn/ETW.py`, `configs/trainer/ETWGradDiff.yaml`) uses the same reweighted NLL but with a *non-learnable* entropy scorer (`src/scorers/ETW.py`) and no saturation term.

## Hyperparameters & Search Space

Tuned with `scripts/hpsearch_atwu.py` (Optuna TPE; requires `pip install ".[hpsearch]"`):

- Phase 1 (Llama-3.2-1B-Instruct, TOFU forget10): `lr` in [1e-5, 5e-5] (log), `gamma` in [0.5, 8.0] (log), `alpha` in [0.2, 2.0] (log), `beta` in [3.0, 10.0]. Fixed: `scorer_lr=0.05`, `update_every_n_steps=5`, `lambda_entropy=1`, `lambda_population=15`, `budget=0.2`, `lambda_l2=1`, batch 8 × grad-accum 4, 10 epochs. Objective: `retain_extraction_strength - extraction_strength` from `TOFU_SUMMARY.json`.
- Phase 2 (Llama-3.1-8B-Instruct): lr-only transfer within ±5× of the phase-1 best lr; all other params frozen.

```bash
python scripts/hpsearch_atwu.py --gpus 0                # both phases
python scripts/hpsearch_atwu.py --gpus 0 --skip-phase2  # 1B search only
```

Best parameters land in `hyperparam/TOFU_pop15/bayesian_summary.json` (keys `ATWU`, `ATWU_8B`). The run scripts read that file when it exists and otherwise fall back to the frozen paper parameters in `scripts/selected_tofu.json` — so the search is optional; `SUMMARY_JSON=<path>` overrides either.

## Computational Setup

Single-GPU runs (1B fits comfortably on any modern GPU; 8B was run on an A100-80GB). The paraphrase judging stage is API-only (OpenAI by default) and runs on CPU.

## Results

Unlearn + evaluate (TOFU metrics via the standard pipeline; paraphrase quality/repetitiveness/win-rate via the paraphrase eval suite):

```bash
# Requires .env with HF_TOKEN (and OPENAI_API_KEY for the judging stage)
bash community/methods/ATWU/tofu-llama3.2-1b-instruct.sh forget10
bash community/methods/ATWU/tofu-llama3.1-8b-instruct.sh forget10
```

The win-rate metric needs a pretrained-model repetitiveness baseline; generate it once per model/split with `repetitiveness-winrate-eval.sh` and `BASELINE_ONLY=1` (see the header of `tofu-llama3.2-1b-instruct.sh`).

`tofu-llama3.2-1b-instruct-trace.sh` is a scorer ablation that checkpoints the scorer during training (`scorer_checkpoints/step_*.pt`).

Consolidate results across runs:

```bash
bash scripts/consolidate_atwu.sh   # -> results_archive/final_runs_atwu_short.json
```

To reproduce any paper method (including the baselines) with the frozen selected
hyperparameters from `scripts/selected_tofu.json`, use `scripts/reproduce_tofu.py`;
the full end-to-end protocol is documented in [`REPRODUCE.md`](REPRODUCE.md)
(in this directory).
