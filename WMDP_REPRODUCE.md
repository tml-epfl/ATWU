# WMDP reproduction

This branch contains the deterministic corpus generator, the tuning procedure,
the frozen selected configurations, and the three-seed final evaluator. All
WMDP scripts live in `scripts/wmdp/`:

| Script | Purpose |
|---|---|
| `generate_corpus.py` | Generate the training corpus; `--check` verifies an existing one |
| `tune_atwu.py`, `tune_baselines.py` | Steps 1–2: 20-trial search and top-trial selection |
| `evaluate.py` | Step 3: retrain the selected configurations on seeds 1–3 |
| `_common.py` | Shared training, evaluation, and corpus helpers |

```bash
pip install -e '.[lm-eval,hpsearch,paraphrase-judge]'
pip install --no-build-isolation flash-attn==2.6.3 einops
```

## Pipeline

Every method goes through the same three steps:

1. **Tune.** Run 20 Optuna trials at training seed 0. The first trial uses the
   method's initial parameters from preliminary sweeps; the other 19 are
   proposed by Optuna's TPE sampler. The objective is
   `MMLU − (mean(max(0.25, WMDP-Bio), max(0.25, WMDP-Cyber)) − 0.25)`.
2. **Select.** Keep the trial with the highest objective.
3. **Evaluate.** Retrain the selected configuration with seeds 1, 2, and 3
   and report the mean and sample standard deviation of WMDP, MMLU,
   repetitiveness, and win rate.

## 1. Generate the released corpus

```bash
python scripts/wmdp/generate_corpus.py
python scripts/wmdp/generate_corpus.py --check
```

The generator uses the committed subtopics, pinned Zephyr revision, seed, batch
size, and sampling parameters. Its manifest records the environment and output
hashes. Repeated runs are expected to be byte-identical in the same software
and hardware environment; cross-environment byte identity is not guaranteed.

The original experimental corpus is not distributed because it contains
long-form cybersecurity and biological-hazard material. The deterministic
pipeline was finalized after the original experiments, so it does not recreate
that corpus exactly.

## 2. Reproduce the main results

The selected configurations are stored in `results/wmdp.json`.
Hyperparameters were selected on the unavailable preliminary corpus. For every
method, the top three search candidates were evaluated on seeds 0–2 on that
corpus, and the winner was frozen. The main results transfer those frozen
configurations to the released corpus without retuning and evaluate seeds 1–3.
Evaluation remains on the standard WMDP and utility benchmarks.

```bash
export WMDP_STATE=saves/wmdp-final
python scripts/wmdp/evaluate.py --gpus 0
```

Per-seed metrics are written to `$WMDP_STATE/final/results.json`.

The evaluator is resumable. It records the hyperparameters in each run
directory and refuses to reuse a directory trained with different ones. Pass
one or more values to `--method` to restrict a run, and pass `--no-paraphrase`
for a WMDP/MMLU-only run. Win-rate evaluation requires the same judge
credentials used for TOFU; WMDP, MMLU, and repetitiveness do not require a
judge API.

## 3. Rerun the pipeline

This runs the three pipeline steps on the released corpus. It reproduces the
procedure, not the published hyperparameters. Each search writes its top trial
to `$WMDP_STATE/selected.json` when its 20 trials have completed; rerunning a
finished search only rewrites that entry.

```bash
export WMDP_STATE=saves/wmdp-retune

python scripts/wmdp/tune_atwu.py --gpus 0 --n-trials 20
for method in graddiff jensun wga satimp npo simnpo rmu; do
  python scripts/wmdp/tune_baselines.py \
    --method "$method" --gpus 0 --n-trials 20
done

python scripts/wmdp/evaluate.py \
  --params "$WMDP_STATE/selected.json" --state "$WMDP_STATE" --gpus 0
```

The initial configurations came from preliminary sweeps. Their values and all
search ranges are included directly in the tuning scripts. Set `ATWU_PYTHON` to
override the interpreter and `HF_HOME` to use a non-default model cache.
