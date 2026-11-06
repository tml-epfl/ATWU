<div align="center">

# Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance

### Official implementation of **A**lternating **T**oken-**W**eighted **U**nlearning (**ATWU**)

**Gizem Yüce<sup>*</sup>, Giorgos Nikolaou<sup>*</sup>, Nicolas Flammarion**<br>
EPFL · <sup>*</sup>Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2606.06320-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.06320)
[![GitHub stars](https://img.shields.io/github/stars/tml-epfl/ATWU?style=flat&logo=github&logoColor=white)](https://github.com/tml-epfl/ATWU)
[![Repo Size](https://img.shields.io/github/repo-size/tml-epfl/ATWU?logo=github&logoColor=white)](https://github.com/tml-epfl/ATWU)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

Machine unlearning aims to remove targeted knowledge from a model while
preserving its general capabilities. For autoregressive language models,
however, not every token in a forget sample is equally relevant to the
knowledge being removed.

We introduce **Alternating Token-Weighted Unlearning (ATWU)**, a lightweight
framework that jointly learns token-level forget-specificity and updates the
language model during unlearning. ATWU uses a small scorer over the model's
hidden states to identify tokens whose forgetting interferes least with the
retain objective. It requires no external token-level annotations or auxiliary
language models.

ATWU achieves state-of-the-art forget-retain trade-offs on **TOFU** and
**RWKU**, while adding minimal computational overhead. Unless stated otherwise,
this README and the `main` branch describe our **TOFU experiments**, which are
implemented on top of
[OpenUnlearning](https://github.com/locuslab/open-unlearning).

The RWKU implementation and the additional WMDP experiments are described
separately at the end.

## Setup

Python 3.11 or newer and a CUDA-capable GPU are required. The paper experiments
used Python 3.12.7 and NVIDIA A100-SXM4-80GB GPUs. The 1B TOFU experiments can
run on smaller modern GPUs, while the 8B experiments require substantially more
memory.

```bash
git clone https://github.com/tml-epfl/ATWU.git
cd ATWU

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install ".[lm-eval,paraphrase-judge,hpsearch]"
pip install --no-build-isolation flash-attn==2.6.3
pip install einops
```

Create a `.env` file in the repository root:

```bash
HF_TOKEN=hf_...          # Hugging Face access for gated Llama models
OPENAI_API_KEY=sk-...    # Only needed for judge-based evaluations
```

The DPO baseline additionally requires:

```bash
python setup_data.py --idk
```

## Reproducing the TOFU results

The selected hyperparameters reported in the paper are provided in
[`scripts/selected_tofu.json`](scripts/selected_tofu.json).

To reproduce ATWU on Llama-3.2-1B-Instruct and the TOFU `forget10` split:

```bash
python scripts/reproduce_tofu.py \
    --method ATWU \
    --model 1B \
    --forget-split forget10
```

The default evaluation includes TOFU forget/retain extraction strength, model
utility, repetitiveness, and MMLU. To additionally run the API-based paraphrase
quality and win-rate evaluations:

```bash
python scripts/reproduce_tofu.py \
    --method ATWU \
    --model 1B \
    --forget-split forget10 \
    --with-paraphrase
```

The script supports:

| Option | Available values |
|---|---|
| Models | `1B`, `8B` |
| Forget splits | `forget01`, `forget05`, `forget10` |
| Methods | `GradDiff`, `DPO`, `NPO`, `SimNPO`, `JensUn`, `RMU`, `WGA`, `SatImp`, `ETW`, `ATWU` |

For example, to reproduce a baseline:

```bash
python scripts/reproduce_tofu.py \
    --method NPO \
    --model 8B \
    --forget-split forget05
```

RMU was evaluated only on the 8B model and ETW only on the 1B model, matching
the experiments in the paper. Outputs are written under:

```text
saves/unlearn/TOFU_repro/<model>/<split>/<method>
```

### Hyperparameter search

Re-running the ATWU search is optional. The reproduction commands above use the
frozen paper parameters by default.

```bash
# Full two-stage search: 1B search followed by learning-rate transfer to 8B
python scripts/hpsearch_atwu.py --gpus 0

# Run only the 1B search
python scripts/hpsearch_atwu.py --gpus 0 --skip-phase2
```

For the complete TOFU protocol, including win-rate baseline generation,
bit-exact package versions, hardware details, and result consolidation, see
[`community/methods/ATWU/REPRODUCE.md`](community/methods/ATWU/REPRODUCE.md).

### TOFU ablations

The sensitivity sweeps and scorer-transfer experiments are released on the
[`tofu-ablations` branch](https://github.com/tml-epfl/ATWU/tree/tofu-ablations).
See its
[`TOFU_ABLATIONS.md`](https://github.com/tml-epfl/ATWU/blob/tofu-ablations/TOFU_ABLATIONS.md)
for the complete reproduction procedure and reference results.

## Other benchmarks

### RWKU

The code used for our RWKU experiments is **not included in this repository**.

Rather than porting RWKU into the OpenUnlearning-based implementation used for
our TOFU experiments, we adapted the
[official RWKU codebase](https://github.com/jinzhuoran/RWKU) to support ATWU and
the other methods evaluated in our paper. We encountered significant
compatibility issues when attempting to integrate the RWKU pipeline into
OpenUnlearning; the reverse direction--integrating ATWU and the additional
baselines into the original RWKU codebase--was substantially more direct.

Our adaptation adds:

- ATWU and its learned token-level scorer;
- retain-set training, which the upstream RWKU release does not provide;
- GradDiff, SimNPO, JensUn, WGA, and SatImp; and
- the common evaluation protocol used for our method comparisons.

RWKU does not explicitly provide retain data. We therefore use a disjoint set
of ten public figures as a surrogate retain set. Our main experiment performs
10-target batch unlearning: a single run jointly removes knowledge about a
fixed set of ten target figures using Phi-3-Mini-4k-Instruct.

The exact target and retain construction, training procedure, hyperparameter
search, baselines, and evaluation protocol are documented in the experimental
details and appendices of the
[paper](https://arxiv.org/abs/2606.06320). Because the corresponding adapted
RWKU code is not distributed here, this repository does not provide RWKU
reproduction commands.

### WMDP

The WMDP experiments and corpus-generation pipeline are released on the
dedicated [`wmdp` branch](https://github.com/tml-epfl/ATWU/tree/wmdp). Switch
to that branch before running the commands in this section:

```bash
git fetch origin
git switch wmdp

pip install -e ".[lm-eval,hpsearch,paraphrase-judge]"
pip install --no-build-isolation flash-attn==2.6.3
pip install einops
```

#### Why we construct a fixed corpus

We distinguish the
[WMDP evaluation benchmark](https://arxiv.org/abs/2403.03218) from the corpora
used to perform unlearning. WMDP contains 3,668 multiple-choice questions,
whereas its released forget corpora contain approximately 24,500 Bio documents
and 1,000 Cyber documents. The benchmark does not define a canonical,
method-independent training subset for a fixed unlearning budget.

For example, the official Zephyr RMU reproduction alternates between Bio and
Cyber for 150 updates with batch size 4, without first shuffling either corpus.
It therefore consumes the first 300 documents from each domain--approximately
1.2% of Bio but 30% of Cyber. This makes the training data dependent on dataset
order. Implementations that shuffle or sample the corpora may consequently
train on substantially different documents. In preliminary experiments in
which the consumed subset changed with the seed, we observed significant
variation across seeds for **every unlearning method**, with the effect most
pronounced on WMDP-Bio. Document selection can therefore obscure differences
between unlearning algorithms.

To remove corpus identity as a hidden experimental variable, we construct a
fixed and balanced corpus shared by every method and seed. Following the
model-elicitation principle used by
[RWKU](https://arxiv.org/abs/2406.10890), we prompt the target
[Zephyr-7B-beta](https://arxiv.org/abs/2310.16944) checkpoint to verbalize its
knowledge of public Bio and Cyber topics. This makes the unlearning objective
operate on knowledge the target model can express, rather than on arbitrary
documents it may not have internalized.

The corpus contains:

- 3,215 Bio forget passages;
- 3,215 Cyber forget passages; and
- 4,029 retain passages covering all 57 MMLU subjects.

The generator targets 3,200 passages per forget domain and 4,000 retain
passages. It checks the count after each batch of 48 generations, so it stops
at the first batch that reaches the target and slightly overshoots it.

Each training run uses a fixed subset of 2,400 passages per forget domain (150
steps of 32 passages, alternating between domains), selected by a seed-0
shuffle of the full file; the training seed changes only their order. The
remaining 815 passages per domain are not used for training. Each forget
passage is paired with a retain passage drawn without replacement from the
shared retain file.

The forget passages cover 12 high-level categories per domain, expanded into
up to 50 technical subtopics per category (600 Bio, 596 Cyber) and expressed
using eight expository framings. The retain corpus is generated by the same model and in the same
style, preventing methods from separating forget and retain examples through
superficial source or formatting differences.

No WMDP question, answer, or answer choice is used during corpus generation.
Evaluation remains unchanged on the standard WMDP multiple-choice questions.

#### Generate the corpus

We do not distribute the original generated corpora used in our experiments.
Because they contain long-form material about cybersecurity and biological
hazards, we chose not to directly release and further disseminate those
documents. Instead, we release the prompts, frozen subtopics, filtering rules,
and generation code so that researchers can construct a comparable corpus
locally.

The main results use the corpus produced by this generator. Repeated runs in
the same software and hardware environment produce byte-identical files;
identity across environments is not guaranteed. One limitation remains: the
hyperparameters were selected earlier, on a preliminary corpus generated
before the pipeline was made deterministic. That preliminary corpus is not
released and cannot be reconstructed exactly.

```bash
python scripts/wmdp/generate_corpus.py
python scripts/wmdp/generate_corpus.py --check
```

By default, generation uses the committed subtopics, a pinned Zephyr revision,
seed 0, temperature 0.9, top-p 0.95, and a repetition penalty of 1.05. The
outputs are:

```text
data/wmdp/generated/bio-forget.text.jsonl
data/wmdp/generated/cyber-forget.text.jsonl
data/wmdp/generated/general-retain.text.jsonl
data/wmdp/generated/generation_manifest.json
```

To resample both the subtopics and passages:

```bash
python scripts/wmdp/generate_corpus.py --regenerate-subtopics
```

Using the committed frozen subtopics is recommended for generating the released
reference corpus. Exact reconstruction of the original experimental corpus is
not possible because that corpus is not distributed. The manifest records the
model revision, package versions, GPU, generation parameters, and output
hashes.

#### Pipeline

Every method goes through the same three steps:

1. **Tune.** Run 20 Optuna trials at training seed 0, starting from the
   method's initial parameters from preliminary sweeps.
2. **Select.** Keep the trial with the highest search objective.
3. **Evaluate.** Retrain the selected configuration with seeds 1, 2, and 3.

#### Reproduce the main results

Hyperparameters were selected on a preliminary model-generated corpus that is
not released and cannot be reconstructed exactly. For each method, the top
three search configurations were evaluated on seeds 0, 1, and 2 on that corpus,
and the resulting winner was frozen. The main results retrain those frozen
configurations on the deterministic corpus generated above, using seeds 1, 2,
and 3. No method is retuned on the released corpus. The standard WMDP
multiple-choice benchmark and utility evaluations remain unchanged.

The frozen parameters are committed in the WMDP branch's
[`results/wmdp.json`](https://github.com/tml-epfl/ATWU/blob/wmdp/results/wmdp.json).
Run their three-seed evaluation with:

```bash
export WMDP_STATE=saves/wmdp-final
python scripts/wmdp/evaluate.py --gpus 0
```

Per-seed metrics are written to `$WMDP_STATE/final/results.json`.

Pass `--no-paraphrase` to run only WMDP and MMLU. The complete main-table run
also evaluates repetitiveness and win rate.

#### Rerun the pipeline

The released code can rerun the pipeline on a supplied corpus, but not
reproduce the original tuning outcomes because the preliminary corpus is
unavailable. Each search is initialized with parameters obtained from
preliminary sweeps; those values, the search spaces, and the frozen selected
configurations are all included in this branch.

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

The search scripts use separate resumable SQLite studies. When a search
completes its 20 trials, it writes the top trial to
`$WMDP_STATE/selected.json`, which `evaluate.py` then retrains on
seeds 1, 2, and 3.

WMDP and MMLU evaluation use `lm-eval` and require no judge API. The
repetitiveness and win-rate stage uses the same judge credentials as the TOFU
evaluation.

For the complete protocol and branch results, see:

- [`WMDP_REPRODUCE.md`](https://github.com/tml-epfl/ATWU/blob/wmdp/WMDP_REPRODUCE.md)
- [`results/wmdp.json`](https://github.com/tml-epfl/ATWU/blob/wmdp/results/wmdp.json)

## Citation

```bibtex
@article{yuce2026learning,
  title={Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance},
  author={Y{\"u}ce, Gizem and Nikolaou, Giorgos and Flammarion, Nicolas},
  journal={arXiv preprint arXiv:2606.06320},
  year={2026}
}
```
