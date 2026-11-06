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

The RWKU implementation is described separately below. The WMDP experiments
are maintained on a dedicated branch.

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
[`scripts/selected_tofu.json`](scripts/selected_tofu.json), and the reference
metrics for every reported model, split, and method are collected in
[`results/tofu.json`](results/tofu.json).

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
reproduction commands. The selected configurations and reference metrics are
provided in [`results/rwku.json`](results/rwku.json).

### WMDP

The WMDP experiments and reproduction pipeline are maintained on the dedicated
[`wmdp` branch](https://github.com/tml-epfl/ATWU/tree/wmdp). See
[`WMDP_REPRODUCE.md`](https://github.com/tml-epfl/ATWU/blob/wmdp/WMDP_REPRODUCE.md)
on that branch for the complete instructions.

## Citation

```bibtex
@article{yuce2026learning,
  title={Learning What to Forget: Improving LLM Unlearning via Learned Token-Level Importance},
  author={Y{\"u}ce, Gizem and Nikolaou, Giorgos and Flammarion, Nicolas},
  journal={arXiv preprint arXiv:2606.06320},
  year={2026}
}
```
