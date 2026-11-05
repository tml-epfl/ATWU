#!/usr/bin/env python
"""ATWU sensitivity (rho, beta) + 5-fold scorer-transfer sweep.

TOFU forget10, Llama-3.2-1B-Instruct, single seed. 25 runs total:

  EXP 1 — sensitivity (20 runs)
      rho  in {0.1, 0.2, ..., 1.0}   (scorer_trainer.budget)
      beta in {0, 1, ..., 10}        (method_args.beta, propagates to the scorer
                                      objective through `beta: ${..beta}`)
      Everything else frozen at the selected paper hyperparameters. The point
      (rho=0.2, beta=7) is the reference and belongs to BOTH grids, so 10 + 11
      slots are 20 distinct runs; it runs first and is checked against the
      on-disk anchor run as a pipeline validation.

  EXP 2 — 5-fold scorer transfer (5 runs)
      forget10's 400 rows split into 5 disjoint contiguous folds of 80. Per fold:
        stage A: fit the scorer on that fold alone with the base model FROZEN
                 (src/train.py scorer_only=true) — keeps the scorer's input
                 distribution matched to the tofu_full hidden states it is later
                 applied to.
        stage B: full ATWU unlearning on ALL of forget10 with that scorer loaded
                 and frozen. Each scorer has seen 20% of the target forget set.
      Scored by the identical standard forget10 eval, so numbers are directly
      comparable to the in-domain co-trained reference already on disk.

Metrics per run (four, in two eval calls): forget/retain extraction strength
(TOFU eval), repetitiveness + MMLU (paraphrase eval, winrate deleted -> no API
calls anywhere). Model utility comes along free in the same TOFU pass and is
recorded but not part of the analysis.

Each run also records the realised token-selection statistics parsed from
ATWU.log, which is what tells you whether the population constraint still binds
at each rho.

Sharding: --shard i --num-shards N runs slice i of the manifest, so N processes
on N GPUs cover everything. With 5 shards each gets 4 exp-1 runs + 1 fold
(~1.8 h). Model weights are deleted after a run's evals succeed (2.4 GB each);
scorer.pt, logs and metric JSONs are kept.

Usage:
    python scripts/sweep_atwu_sensitivity.py --shard 0 --num-shards 5 --gpu 0
    python scripts/sweep_atwu_sensitivity.py --list
    python scripts/sweep_atwu_sensitivity.py --collect
    bash scripts/sweep_atwu_launch.sh          # all shards, GPUs 0-4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import reproduce_tofu as rt  # noqa: E402  (command shapes: single source of truth)

# Same interpreter for the subprocesses as for this driver (the repo deps live
# in .venv, so "python" from an outer shell would be the wrong one).
PY = sys.executable

# ── Sweep definition ──
MODEL_KEY = "1B"
MODEL = rt.MODELS[MODEL_KEY]
INPUT_DIM = rt.INPUT_DIM[MODEL_KEY]
FORGET_SPLIT = "forget10"
RETAIN_SPLIT, HOLDOUT_SPLIT = rt.SPLITS[FORGET_SPLIT]
SEED = 0

RHO_GRID = [round(0.1 * i, 1) for i in range(1, 11)]   # 0.1 .. 1.0
BETA_GRID = [float(b) for b in range(0, 11)]           # 0 .. 10
REF_RHO, REF_BETA = 0.2, 7.0

N_FOLDS = 5
FORGET10_ROWS = 400
QA_PER_AUTHOR = 20                                     # forget10 = 20 authors x 20 QA
FOLD_SIZE = FORGET10_ROWS // N_FOLDS                   # 80 rows = 4 authors
# Scorer-only pretrain steps for a fold. The reference run performs 120/5 = 24
# scorer updates, each accumulating GRAD_ACCUM micro-batches; _pretrain_scorer
# steps once per micro-batch, so 24*4 = 96 matches micro-batches of gradient
# signal. Acceptance is by the score statistics below, not by this number.
SCORER_PRETRAIN_STEPS = 96

# Reference-run tail statistics (from saves/unlearn/TOFU_repro/.../forget10/ATWU).
# A fold scorer whose scores do not land in this regime has not converged and is
# not a fair test of transfer — the driver warns and records the deviation.
REF_SELECTED_PCT = (15.0, 25.0)   # reference tail: 17-21%
REF_SCORE_MEAN = (0.15, 0.28)     # reference tail: 0.20-0.23

SWEEP_ROOT = Path(os.environ.get("ATWU_SWEEP_ROOT", "saves/sweep_atwu"))
RECORDS_DIR = SWEEP_ROOT / "records"

# On-disk anchor for the reference config; run 1 must reproduce these.
ANCHOR = {
    "extraction_strength": 0.03530649146779941,
    "retain_extraction_strength": 0.7526790032545378,
    "model_utility": 0.6015994333374167,
    "mmlu": 0.44993590656601623,
    "repetitiveness": 574.5493225108538,
}
ANCHOR_RUN = (f"saves/unlearn/TOFU_repro/{MODEL}/{FORGET_SPLIT}/ATWU")


def selected_params() -> dict:
    """The frozen paper hyperparameters for ATWU (single source of truth)."""
    p = json.loads((REPO_ROOT / "scripts" / "selected_tofu.json").read_text())
    return p["methods"]["ATWU"]


# ── Manifest ──

def build_manifest() -> list[dict]:
    """Ordered run list. Reference first, then the rho grid, then the beta grid,
    then the 5 folds. Sharding strides this list, so exp-1 runs spread evenly and
    each shard of 5 gets exactly one fold."""
    p = selected_params()
    lr, alpha, gamma, scorer_lr = p["lr_1B"], p["alpha"], p["gamma"], p["scorer_lr"]

    runs: list[dict] = [{
        "run_id": "reference",
        "kind": "exp1",
        "rho": REF_RHO, "beta": REF_BETA,
        "lr": lr, "alpha": alpha, "gamma": gamma, "scorer_lr": scorer_lr,
        "out_dir": str(SWEEP_ROOT / "exp1" / "reference"),
        "is_reference": True,
    }]
    for rho in RHO_GRID:
        if rho == REF_RHO:
            continue
        runs.append({
            "run_id": f"rho{rho:g}", "kind": "exp1",
            "rho": rho, "beta": REF_BETA,
            "lr": lr, "alpha": alpha, "gamma": gamma, "scorer_lr": scorer_lr,
            "out_dir": str(SWEEP_ROOT / "exp1" / f"rho{rho:g}"),
        })
    for beta in BETA_GRID:
        if beta == REF_BETA:
            continue
        runs.append({
            "run_id": f"beta{beta:g}", "kind": "exp1",
            "rho": REF_RHO, "beta": beta,
            "lr": lr, "alpha": alpha, "gamma": gamma, "scorer_lr": scorer_lr,
            "out_dir": str(SWEEP_ROOT / "exp1" / f"beta{beta:g}"),
        })
    for k in range(N_FOLDS):
        lo, hi = k * FOLD_SIZE, (k + 1) * FOLD_SIZE
        runs.append({
            "run_id": f"fold{k + 1}", "kind": "exp2",
            "rho": REF_RHO, "beta": REF_BETA,
            "lr": lr, "alpha": alpha, "gamma": gamma, "scorer_lr": scorer_lr,
            "fold": k + 1, "fold_slice": [lo, hi],
            "scorer_dir": str(SWEEP_ROOT / "exp2" / f"fold{k + 1}" / "scorer"),
            "out_dir": str(SWEEP_ROOT / "exp2" / f"fold{k + 1}" / "unlearn"),
        })
    return runs


def fold_indices(scheme: str, k: int) -> list[int]:
    """Row indices of held-out fold k (1-based) under either fold scheme.

    forget10 is author-ordered: 20 authors x 20 consecutive QA rows (verified by
    scripts/check_tofu_folds.py).

    "author"   fold k = 4 whole authors  -> tests generalisation to unseen ENTITIES
    "question" fold k = 4 QA rows of EVERY author -> tests generalisation to unseen
               questions about entities the unlearning did target
    """
    if scheme == "author":
        return list(range((k - 1) * FOLD_SIZE, k * FOLD_SIZE))
    if scheme == "question":
        per = QA_PER_AUTHOR // N_FOLDS          # 4 rows per author per fold
        return [a * QA_PER_AUTHOR + (k - 1) * per + j
                for a in range(FORGET10_ROWS // QA_PER_AUTHOR) for j in range(per)]
    raise ValueError(f"unknown fold scheme {scheme!r}")


def _slice_expr(indices: list[int]) -> str:
    """Contiguous index runs as a HF `split` expression, e.g. train[0:4]+train[20:24]."""
    runs: list[tuple[int, int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][1]:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    return "+".join(f"train[{lo}:{hi}]" for lo, hi in runs)


def kfold_holdout_manifest(scheme: str) -> list[dict]:
    """Leave-one-fold-out runs: unlearn on 4 folds (320 rows) with ATWU running
    normally (scorer co-trained, nothing frozen), then evaluate on ALL 400 forget10
    rows. Splitting the per-example eval scores into unlearned vs held-out gives the
    AUROC — i.e. does the unlearning reach forget data it never saw?
    """
    p = selected_params()
    runs = []
    for k in range(1, N_FOLDS + 1):
        held = fold_indices(scheme, k)
        trained = [i for i in range(FORGET10_ROWS) if i not in set(held)]
        runs.append({
            "run_id": f"kf{scheme[0]}{k}", "kind": "exp3",
            "rho": REF_RHO, "beta": REF_BETA,
            "lr": p["lr_1B"], "alpha": p["alpha"], "gamma": p["gamma"],
            "scorer_lr": p["scorer_lr"],
            "fold_scheme": scheme, "held_out_fold": k,
            "held_out_indices": held, "trained_indices": trained,
            "train_split_expr": _slice_expr(trained),
            "out_dir": str(SWEEP_ROOT / "exp3" / f"{scheme}_holdout{k}"),
        })
    return runs


def uniform_control_run() -> dict:
    """Optional 26th run: frozen zero-init scorer. zero_init + sigmoid gives every
    token a score of exactly 0.5, i.e. uniform weighting — the floor the k-fold
    numbers need in order to mean anything. Off by default (25-run budget).

    Note when reading its logs: the selection statistic counts scores strictly
    > 0.5, so this run reports selected=0% while in fact weighting every token at
    0.5. That is the control working as intended, not a broken scorer."""
    p = selected_params()
    return {
        "run_id": "uniform", "kind": "exp2",
        "rho": REF_RHO, "beta": REF_BETA,
        "lr": p["lr_1B"], "alpha": p["alpha"], "gamma": p["gamma"],
        "scorer_lr": p["scorer_lr"],
        "uniform": True,
        "out_dir": str(SWEEP_ROOT / "exp2" / "uniform"),
    }


# ── Command construction ──

def atwu_overrides(run: dict) -> list[str]:
    ma = "trainer.method_args"
    return [
        f"{ma}.alpha={run['alpha']}",
        f"{ma}.gamma={run['gamma']}",
        f"{ma}.beta={run['beta']}",
        f"{ma}.scorer.cfg.input_dimension={INPUT_DIM}",
        f"{ma}.scorer_trainer.optim_cfg.lr={run['scorer_lr']}",
        f"{ma}.scorer_trainer.optim_cfg.update_every_n_steps=5",
        f"+{ma}.scorer_trainer.optim_cfg.scheduler=linear",
        f"{ma}.scorer_trainer.lambda_entropy=1",
        f"{ma}.scorer_trainer.lambda_population=15",
        f"{ma}.scorer_trainer.budget={run['rho']}",
        f"{ma}.scorer_trainer.lambda_l2=1",
    ]


def base_train_cmd(run: dict, out_dir: str, task_name: str) -> list[str]:
    return [
        PY, "src/train.py", "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
        "trainer=ATWU",
        f"model={MODEL}",
        f"task_name={task_name}",
        f"paths.output_dir={out_dir}",
        f"model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{MODEL}_full",
        *rt.hf_token_overrides(),
        "++model.model_args.device_map=auto",
        f"forget_split={FORGET_SPLIT}",
        f"retain_split={RETAIN_SPLIT}",
        f"trainer.args.num_train_epochs={rt.NUM_EPOCHS}",
        f"trainer.args.per_device_train_batch_size={rt.BATCH_SIZE}",
        f"trainer.args.gradient_accumulation_steps={rt.GRAD_ACCUM}",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        f"trainer.args.learning_rate={run['lr']}",
        f"trainer.args.seed={SEED}",
        *atwu_overrides(run),
    ]


def scorer_stage_cmd(run: dict) -> list[str]:
    """Stage A: fit the scorer on one fold with the model frozen."""
    lo, hi = run["fold_slice"]
    task_name = f"sweep_atwu/exp2/fold{run['fold']}/scorer"
    cmd = base_train_cmd(run, run["scorer_dir"], task_name)
    cmd += [
        "+scorer_only=true",
        f"trainer.method_args.scorer_pretrain.steps={SCORER_PRETRAIN_STEPS}",
        f"trainer.method_args.scorer_pretrain.lr={run['scorer_lr']}",
        # Quoted so hydra's override grammar reads a string, not a list literal.
        f"data.forget.TOFU_QA_forget.args.hf_args.split='train[{lo}:{hi}]'",
    ]
    return cmd


def unlearn_cmd(run: dict) -> list[str]:
    task_name = f"sweep_atwu/{run['kind']}/{run['run_id']}"
    cmd = base_train_cmd(run, run["out_dir"], task_name)
    if run.get("train_split_expr"):
        # Leave-one-fold-out: unlearn on 4 folds only. ATWU is otherwise untouched
        # (scorer co-trained as normal); the eval still covers all 400 rows.
        cmd += [f"data.forget.TOFU_QA_forget.args.hf_args.split="
                f"'{run['train_split_expr']}'"]
    if run.get("fold"):
        cmd += [
            f"trainer.method_args.scorer.pretrained_path={run['scorer_dir']}/scorer.pt",
            "trainer.method_args.scorer_pretrain.freeze_after=true",
        ]
    elif run.get("uniform"):
        # No pretrain, no checkpoint, frozen from step 0 -> constant 0.5 scores.
        cmd += ["trainer.method_args.scorer_pretrain.freeze_after=true"]
    return cmd


def tofu_eval_cmd(run: dict) -> list[str]:
    """ES_forget + ES_retain (+ model utility, free in the same pass).

    Leave-one-fold-out runs use `tofu_kfold` instead, which adds the per-example
    forget probability/loss needed for the second AUROC. Either way the eval covers
    ALL 400 forget10 rows — the fold split is applied afterwards to value_by_index.
    """
    out_dir = run["out_dir"]
    model_path = run.get("model_path", out_dir)
    eval_set = run.get("eval_set") or (
        "tofu_kfold" if run.get("train_split_expr") else "tofu_repro")
    return [
        PY, "src/eval.py", "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        f"eval=[{eval_set}]",
        f"model={MODEL}",
        f"task_name=sweep_atwu/{run['kind']}/{run['run_id']}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        *rt.hf_token_overrides(),
        "++model.model_args.device_map=auto",
        f"forget_split={FORGET_SPLIT}",
        f"holdout_split={HOLDOUT_SPLIT}",
        f"paths.output_dir={out_dir}",
        "eval.tofu.overwrite=true",
    ]


def para_eval_cmd(run: dict) -> list[str]:
    """Repetitiveness + MMLU in one model load. winrate deleted -> no API calls."""
    out_dir = run["out_dir"]
    model_path = run.get("model_path", out_dir)
    return [
        PY, "src/eval.py",
        "experiment=eval/paraphrase/default.yaml",
        "eval=[paraphrase]",
        f"model={MODEL}",
        f"task_name=sweep_atwu/{run['kind']}/{run['run_id']}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        *rt.hf_token_overrides(),
        "++model.model_args.device_map=auto",
        f"paths.output_dir={out_dir}/paraphrase_evals",
        "++eval.paraphrase.metrics.repetitiveness.generation="
        "{max_new_tokens:128,do_sample:false,temperature:0.0}",
        "~eval.paraphrase.metrics.winrate",
    ]


# ── Execution ──

def run_cmd(cmd: list[str], gpu: str | None, dry: bool) -> tuple[int, float]:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    # Fresh port per command: shards run concurrently on one node.
    s = socket.socket()
    s.bind(("", 0))
    env["MASTER_PORT"] = str(s.getsockname()[1])
    s.close()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    shown = " ".join(a.split("=")[0] + "=***" if ".token=" in a else a for a in cmd)
    print(f"\n[sweep] CUDA_VISIBLE_DEVICES={gpu} {shown}\n", flush=True)
    if dry:
        return 0, 0.0
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    return result.returncode, time.time() - t0


SCORE_RE = re.compile(
    r"scores\s+\S+\s+min=(?P<min>[-\d.]+)\s+max=(?P<max>[-\d.]+)\s+"
    r"mean=(?P<mean>[-\d.]+)\s+std=(?P<std>[-\d.]+)\s+selected=\s*(?P<sel>[-\d.]+)%"
)
LOSS_RE = re.compile(
    r"forget=(?P<forget>[-\d.]+)\s+f_entropy=(?P<ent>[-\d.]+)\s+"
    r"f_budget=(?P<budget>[-\d.]+)\s+l2=(?P<l2>[-\d.]+)"
)


def parse_scorer_log(log_path: Path) -> dict:
    """Realised token-selection statistics from ATWU.log.

    `selected` is the fraction of forget tokens scored > 0.5 — for the rho sweep
    this is what shows whether the population constraint still binds. Tail means
    are over the last 20% of logged scorer updates.
    """
    if not log_path.exists():
        return {"available": False}
    text = log_path.read_text(errors="replace")
    sel = [float(m.group("sel")) for m in SCORE_RE.finditer(text)]
    mean = [float(m.group("mean")) for m in SCORE_RE.finditer(text)]
    std = [float(m.group("std")) for m in SCORE_RE.finditer(text)]
    budget = [float(m.group("budget")) for m in LOSS_RE.finditer(text)]
    if not sel:
        return {"available": False}
    tail = max(1, len(sel) // 5)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "available": True,
        "n_logged_updates": len(sel),
        "selected_pct_tail": avg(sel[-tail:]),
        "score_mean_tail": avg(mean[-tail:]),
        "score_std_tail": avg(std[-tail:]),
        "budget_penalty_tail": avg(budget[-tail:]) if budget else None,
        "selected_pct_series": sel,
        "score_mean_series": mean,
    }


def read_metrics(run: dict) -> dict:
    out = Path(REPO_ROOT / run["out_dir"])
    metrics: dict[str, float | None] = {}
    tofu = out / "TOFU_SUMMARY.json"
    if tofu.exists():
        d = json.loads(tofu.read_text())
        for k in ("extraction_strength", "retain_extraction_strength", "model_utility"):
            metrics[k] = d.get(k)
    para = out / "paraphrase_evals" / "Paraphrase_SUMMARY.json"
    if para.exists():
        d = json.loads(para.read_text())
        metrics["repetitiveness"] = d.get("repetitiveness")
        metrics["mmlu"] = d.get("mmlu")
    return metrics


def prune_weights(run: dict) -> int:
    """Delete model weights once metrics are recorded (2.4 GB/run). scorer.pt,
    logs and metric JSONs stay — scorer.pt is what the agreement analysis reads."""
    out = Path(REPO_ROOT / run["out_dir"])
    freed = 0
    for pattern in ("model.safetensors", "model-*.safetensors", "*.bin"):
        for f in out.glob(pattern):
            if f.name == "scorer.pt":
                continue
            freed += f.stat().st_size
            f.unlink()
    return freed


def check_reference(metrics: dict) -> list[str]:
    lines = []
    for key, expected in ANCHOR.items():
        got = metrics.get(key)
        if got is None:
            lines.append(f"    {key}: MISSING (anchor {expected:.6f})")
            continue
        delta = got - expected
        rel = abs(delta) / max(abs(expected), 1e-12)
        flag = "OK" if rel < 1e-6 else ("close" if rel < 5e-3 else "DIVERGED")
        lines.append(f"    {key}: {got:.6f} vs anchor {expected:.6f} "
                     f"(delta {delta:+.2e}, {flag})")
    return lines


def scorer_health(stats: dict) -> str | None:
    """Acceptance check for a fold scorer (and a sanity check for every run)."""
    if not stats.get("available"):
        return "no scorer statistics found in ATWU.log"
    sel, mean = stats["selected_pct_tail"], stats["score_mean_tail"]
    problems = []
    if not (REF_SELECTED_PCT[0] <= sel <= REF_SELECTED_PCT[1]):
        problems.append(f"selected={sel:.1f}% outside reference band "
                        f"{REF_SELECTED_PCT[0]}-{REF_SELECTED_PCT[1]}%")
    if not (REF_SCORE_MEAN[0] <= mean <= REF_SCORE_MEAN[1]):
        problems.append(f"score mean={mean:.3f} outside reference band "
                        f"{REF_SCORE_MEAN[0]}-{REF_SCORE_MEAN[1]}")
    return "; ".join(problems) or None


def stage_state(run: dict) -> dict:
    """Which stages of this run are already done on disk.

    Lets a resubmitted shard do only the missing work instead of retraining. The
    motivating case: the MMLU stage dying (see --repair) leaves a perfectly good
    checkpoint plus a finished repetitiveness result, so all that is owed is one
    eval — the paraphrase evaluator itself skips metrics already in
    Paraphrase_EVAL.json, so re-running it costs MMLU alone.
    """
    out = Path(REPO_ROOT / run["out_dir"])
    weights = (out / "model.safetensors").exists() or any(out.glob("model-*.safetensors"))
    trained = (out / "trainer_state.json").exists() and weights
    tofu = out / "TOFU_SUMMARY.json"
    es_done = False
    if tofu.exists():
        d = json.loads(tofu.read_text())
        es_done = all(d.get(k) is not None for k in
                      ("extraction_strength", "retain_extraction_strength"))
    para = out / "paraphrase_evals" / "Paraphrase_SUMMARY.json"
    para_done = False
    if para.exists():
        d = json.loads(para.read_text())
        para_done = all(d.get(k) is not None for k in ("mmlu", "repetitiveness"))
    return {"weights": weights, "trained": trained,
            "eval_tofu": es_done, "eval_paraphrase": para_done}


def execute(run: dict, gpu: str | None, dry: bool, keep_weights: bool) -> bool:
    record_path = RECORDS_DIR / f"{run['run_id']}.json"
    prior = None
    if record_path.exists():
        try:
            prior = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            prior = None
    if prior and prior.get("status") == "ok" and not dry:
        print(f"[sweep] {run['run_id']}: complete record exists, skipping")
        return True

    # A prior incomplete record means a stage died (or the job was preempted).
    # Resume from whatever survived on disk rather than redoing the whole run.
    done = {} if dry else stage_state(run)
    if prior and not dry:
        print(f"[sweep] {run['run_id']}: retrying incomplete run "
              f"(prior status {prior.get('status')}); on disk: "
              + ", ".join(k for k, v in done.items() if v) or "nothing")
        if not done.get("weights") and done.get("trained") is False:
            print(f"[sweep] {run['run_id']}: weights are gone — retraining from scratch")

    print(f"\n{'=' * 70}\n[sweep] {run['run_id']}  "
          f"(rho={run['rho']}, beta={run['beta']:g}"
          + (f", fold={run['fold']} rows {run['fold_slice']}" if run.get("fold") else "")
          + f")\n{'=' * 70}", flush=True)

    record: dict = {
        "run_id": run["run_id"], "kind": run["kind"],
        "params": {k: run[k] for k in ("rho", "beta", "lr", "alpha", "gamma", "scorer_lr")},
        "out_dir": run["out_dir"], "seed": SEED,
        "forget_split": FORGET_SPLIT, "model": MODEL,
        "timings_s": {}, "status": "running",
    }
    if run.get("fold"):
        record["params"]["fold"] = run["fold"]
        record["params"]["fold_slice"] = run["fold_slice"]
        record["params"]["scorer_pretrain_steps"] = SCORER_PRETRAIN_STEPS
    if run.get("uniform"):
        record["params"]["uniform_control"] = True
    if run.get("train_split_expr"):
        # The AUROC analysis needs to know which rows this run actually saw.
        record["params"]["fold_scheme"] = run["fold_scheme"]
        record["params"]["held_out_fold"] = run["held_out_fold"]
        record["params"]["train_split_expr"] = run["train_split_expr"]
        record["held_out_indices"] = run["held_out_indices"]
        record["trained_indices"] = run["trained_indices"]

    # ── stage A (folds only): scorer-only training on the fold ──
    if run.get("fold"):
        scorer_pt = Path(REPO_ROOT / run["scorer_dir"]) / "scorer.pt"
        if scorer_pt.exists() and not dry:
            print(f"[sweep] {run['run_id']}: fold scorer exists, skipping stage A")
        else:
            rc, dt = run_cmd(scorer_stage_cmd(run), gpu, dry)
            record["timings_s"]["scorer_stage"] = round(dt, 1)
            if rc != 0:
                record["status"] = "failed:scorer_stage"
                _write_record(record_path, record, dry)
                return False
        stats_a = parse_scorer_log(Path(REPO_ROOT / run["scorer_dir"]) / "ATWU.log")
        record["fold_scorer_stats"] = stats_a
        warn = scorer_health(stats_a)
        record["fold_scorer_warning"] = warn
        if warn:
            print(f"[sweep] WARNING {run['run_id']} fold scorer: {warn}\n"
                  f"        -> not converged to the reference regime; consider "
                  f"raising SCORER_PRETRAIN_STEPS for ALL folds together.", flush=True)

    # Carry forward timings measured before an interruption.
    if prior:
        record["timings_s"].update(prior.get("timings_s", {}))
        record["resumed_from"] = prior.get("status")

    # ── unlearn ──
    if done.get("trained"):
        print(f"[sweep] {run['run_id']}: checkpoint present, skipping training")
    else:
        rc, dt = run_cmd(unlearn_cmd(run), gpu, dry)
        record["timings_s"]["train"] = round(dt, 1)
        if rc != 0:
            record["status"] = "failed:train"
            _write_record(record_path, record, dry)
            return False

    # ── evals ──
    if done.get("eval_tofu"):
        print(f"[sweep] {run['run_id']}: extraction strengths present, skipping TOFU eval")
    else:
        rc, dt = run_cmd(tofu_eval_cmd(run), gpu, dry)
        record["timings_s"]["eval_tofu"] = round(dt, 1)
        if rc != 0:
            record["status"] = "failed:eval_tofu"
            _write_record(record_path, record, dry)
            return False

    if done.get("eval_paraphrase"):
        print(f"[sweep] {run['run_id']}: repetitiveness + MMLU present, "
              f"skipping paraphrase eval")
    else:
        # The evaluator skips metrics already in Paraphrase_EVAL.json, so after an
        # MMLU-only failure this re-runs MMLU alone (~8 min), not repetitiveness.
        rc, dt = run_cmd(para_eval_cmd(run), gpu, dry)
        record["timings_s"]["eval_paraphrase"] = round(dt, 1)
        if rc != 0:
            record["status"] = "failed:eval_paraphrase"
            _write_record(record_path, record, dry)
            return False

    if dry:
        return True

    record["metrics"] = read_metrics(run)
    record["scorer_stats"] = parse_scorer_log(Path(REPO_ROOT / run["out_dir"]) / "ATWU.log")
    # A frozen scorer never runs a scorer update, so the unlearning log carries no
    # score statistics by design — the fold scorer's own stats come from stage A.
    frozen = bool(run.get("fold") or run.get("uniform"))
    if not frozen:
        record["scorer_warning"] = scorer_health(record["scorer_stats"])
    missing = [k for k in ("extraction_strength", "retain_extraction_strength",
                           "repetitiveness", "mmlu") if record["metrics"].get(k) is None]
    record["status"] = "ok" if not missing else f"incomplete:{','.join(missing)}"

    if run.get("is_reference"):
        record["anchor_check"] = check_reference(record["metrics"])
        print(f"\n[sweep] reference vs on-disk anchor ({ANCHOR_RUN}):")
        for line in record["anchor_check"]:
            print(line, flush=True)

    if not missing and not keep_weights:
        freed = prune_weights(run)
        record["weights_pruned_bytes"] = freed
        print(f"[sweep] pruned {freed / 1e9:.2f} GB of weights from {run['out_dir']}")

    _write_record(record_path, record, dry)
    m = record["metrics"]

    def show(key: str, prec: int = 4) -> str:
        v = m.get(key)
        return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "n/a"

    stats = record["scorer_stats"] if record["scorer_stats"].get("available") \
        else record.get("fold_scorer_stats", {})
    sel = stats.get("selected_pct_tail")
    print(f"[sweep] {run['run_id']} done ({record['status']}): "
          f"ES_f={show('extraction_strength')} "
          f"ES_r={show('retain_extraction_strength')} "
          f"MMLU={show('mmlu')} Rep={show('repetitiveness', 2)} "
          f"selected={f'{sel:.1f}%' if isinstance(sel, float) else 'n/a'}",
          flush=True)
    return record["status"] == "ok"


def _write_record(path: Path, record: dict, dry: bool) -> None:
    if dry:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))


def reference_model(hf_id: str, gpu: str | None, dry: bool,
                    eval_set: str | None = None) -> bool:
    """Evaluate an untouched reference model with the SAME two eval stages the
    sweep applies, so the plots' reference lines are measured by the same code
    path (same generation config, same repetitiveness sample count) rather than by
    the community baseline script, which drops MMLU.

    Default target is the retain90 model: the gold standard for forget10 — it never
    learned the forget authors, so its ES_forget is the floor a perfect unlearner
    would reach, and its MMLU/Rep are the capability ceiling.
    """
    tag = hf_id.rstrip("/").split("/")[-1]
    out_dir = SWEEP_ROOT / "reference_model" / tag
    record_path = out_dir / "record.json"
    if record_path.exists() and not dry:
        print(f"[sweep] reference model {tag}: record exists, skipping")
        return True

    run = {
        "run_id": f"refmodel_{tag}", "kind": "ref",
        "out_dir": str(out_dir), "model_path": hf_id,
    }
    if eval_set:
        run["eval_set"] = eval_set
    print(f"\n{'=' * 70}\n[sweep] reference model: {hf_id}\n{'=' * 70}", flush=True)

    record = {"run_id": run["run_id"], "kind": "ref", "model_path": hf_id,
              "out_dir": str(out_dir), "timings_s": {}, "status": "running"}
    for stage, cmd in (("eval_tofu", tofu_eval_cmd(run)),
                       ("eval_paraphrase", para_eval_cmd(run))):
        rc, dt = run_cmd(cmd, gpu, dry)
        record["timings_s"][stage] = round(dt, 1)
        if rc != 0:
            record["status"] = f"failed:{stage}"
            _write_record(record_path, record, dry)
            return False
    if dry:
        return True

    record["metrics"] = read_metrics(run)
    record["status"] = "ok"
    _write_record(record_path, record, dry)
    print(f"[sweep] reference model {tag}: {json.dumps(record['metrics'])}")
    return True


def collect() -> None:
    rows = []
    for f in sorted(RECORDS_DIR.glob("*.json")):
        rows.append(json.loads(f.read_text()))
    out = SWEEP_ROOT / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"{len(rows)} records -> {out}\n")
    hdr = f"{'run':<12}{'rho':>5}{'beta':>6}{'ES_f':>9}{'ES_r':>9}{'MMLU':>8}{'Rep':>9}{'sel%':>7}  status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        m, p = r.get("metrics", {}), r["params"]
        # Frozen-scorer runs log no score stats during unlearning; report the
        # scorer's own statistics from the fold (stage A) instead.
        s = r.get("scorer_stats") or {}
        if not s.get("available"):
            s = r.get("fold_scorer_stats") or {}

        def fmt(v, w, prec):
            return f"{v:>{w}.{prec}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"

        print(f"{r['run_id']:<12}{p['rho']:>5}{p['beta']:>6g}"
              f"{fmt(m.get('extraction_strength'), 9, 4)}"
              f"{fmt(m.get('retain_extraction_strength'), 9, 4)}"
              f"{fmt(m.get('mmlu'), 8, 4)}"
              f"{fmt(m.get('repetitiveness'), 9, 2)}"
              f"{fmt(s.get('selected_pct_tail'), 7, 1)}  {r.get('status')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for this shard")
    ap.add_argument("--only", default=None,
                    help="Comma-separated run_ids to run instead of a shard slice")
    ap.add_argument("--include-uniform", action="store_true",
                    help="Add the frozen zero-init (uniform 0.5) control as a 26th run")
    ap.add_argument("--kfold", choices=["author", "question", "both"], default=None,
                    help="Run the leave-one-fold-out manifest instead of the sensitivity "
                         "sweep: unlearn on 4 of the 5 folds, evaluate all 400 rows, so "
                         "per-example scores split into unlearned vs held-out (AUROC). "
                         "'author' holds out 4 whole authors; 'question' holds out 4 QA "
                         "rows of every author; 'both' runs the two schemes (10 runs)")
    ap.add_argument("--keep-weights", action="store_true",
                    help="Do not delete model.safetensors after a run's evals (2.4 GB/run)")
    ap.add_argument("--repair", action="store_true",
                    help="Visit only the runs of this shard whose record is missing or "
                         "incomplete, resuming each from whatever is already on disk "
                         "(an MMLU-only failure costs one eval, not a retrain)")
    ap.add_argument("--list", action="store_true", help="Print the manifest and exit")
    ap.add_argument("--collect", action="store_true",
                    help="Merge records into saves/sweep_atwu/summary.json and print a table")
    ap.add_argument("--reference-model", nargs="?", const="AUTO", default=None,
                    help="Instead of a shard: evaluate an untouched reference model with "
                         "the same two eval stages (default: the retain90 gold model), "
                         "giving the plots' reference lines")
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = ap.parse_args()

    rt.load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN for the subprocesses

    if args.collect:
        collect()
        return

    if args.reference_model:
        hf_id = (f"open-unlearning/tofu_{MODEL}_{RETAIN_SPLIT}"
                 if args.reference_model == "AUTO" else args.reference_model)
        # With --kfold, use the same eval set as the fold runs so the baseline
        # carries the per-example scores the AUROC analysis needs.
        eval_set = "tofu_kfold" if args.kfold else None
        sys.exit(0 if reference_model(hf_id, args.gpu, args.dry_run, eval_set) else 1)

    if args.kfold:
        schemes = ["author", "question"] if args.kfold == "both" else [args.kfold]
        manifest = [r for s in schemes for r in kfold_holdout_manifest(s)]
    else:
        manifest = build_manifest()
        if args.include_uniform:
            manifest.append(uniform_control_run())

    if args.list:
        kinds = {}
        for r in manifest:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{len(manifest)} runs "
              f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})\n")
        for i, r in enumerate(manifest):
            shard = i % max(args.num_shards, 1)
            extra = f" fold rows {r['fold_slice']}" if r.get("fold") else ""
            print(f"  [{i:2d}] shard {shard}  {r['run_id']:<12} "
                  f"rho={r['rho']:<4g} beta={r['beta']:<4g}{extra}")
        return

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        todo = [r for r in manifest if r["run_id"] in wanted]
        unknown = wanted - {r["run_id"] for r in todo}
        if unknown:
            sys.exit(f"unknown run_ids: {sorted(unknown)}")
    else:
        todo = manifest[args.shard::args.num_shards]

    if args.repair:
        # Visit only runs that are not complete. Every stage is skipped if its
        # output is already on disk, so a repair of an MMLU-only failure costs one
        # eval (~8 min) rather than a full retrain.
        broken = []
        for r in todo:
            rec = RECORDS_DIR / f"{r['run_id']}.json"
            if not rec.exists():
                broken.append(r)
                continue
            try:
                if json.loads(rec.read_text()).get("status") != "ok":
                    broken.append(r)
            except json.JSONDecodeError:
                broken.append(r)
        print(f"[sweep] --repair: {len(broken)}/{len(todo)} runs in this shard "
              f"are incomplete: {[r['run_id'] for r in broken]}")
        todo = broken

    print(f"[sweep] shard {args.shard}/{args.num_shards} on GPU {args.gpu}: "
          f"{len(todo)} runs -> {[r['run_id'] for r in todo]}", flush=True)

    failures = []
    for run in todo:
        if not execute(run, args.gpu, args.dry_run, args.keep_weights):
            failures.append(run["run_id"])

    print(f"\n[sweep] shard {args.shard} finished: "
          f"{len(todo) - len(failures)}/{len(todo)} ok"
          + (f", failed: {failures}" if failures else ""))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
