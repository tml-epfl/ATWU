#!/usr/bin/env python
"""ATWU HP search + 8B lr transfer.

Phase 1: full Bayesian search over (lr, gamma, alpha, beta) on 1B
         (scorer_lr is fixed to SCORER_LR).
Phase 2: lr-only Bayesian search on 8B using phase-1 best non-lr params.

Both phases share one Optuna DB (scripts/hpsearch_atwu_pop15.db) under separate
study names. Each phase writes its best to its own bayesian_summary.json.

Usage:
    python scripts/hpsearch_atwu.py --gpus 0
    python scripts/hpsearch_atwu.py --gpus 0 --skip-phase1   # reuse existing 1B summary
    python scripts/hpsearch_atwu.py --gpus 0 --skip-phase2   # only do 1B search
"""

import argparse
import fcntl
import json
import logging
import os
import subprocess
from pathlib import Path

import optuna

FORGET_SPLIT = "forget10"
NUM_EPOCHS = 10
ARTIFACT_ROOT = Path("hyperparam/TOFU_pop15")
SUMMARY_PATH = ARTIFACT_ROOT / "bayesian_summary.json"
LR_RELATIVE_RANGE = 5.0
SCORER_LR = 0.05

DB = "sqlite:///scripts/hpsearch_atwu_pop15.db"

# Phase 1 (1B full search)
MODEL_1B = "Llama-3.2-1B-Instruct"
PRETRAINED_1B = f"open-unlearning/tofu_{MODEL_1B}_full"
STUDY_1B = "bayesian_ATWU"
SUMMARY_KEY_1B = "ATWU"
INPUT_DIM_1B = 2048

# Phase 2 (8B lr transfer)
MODEL_8B = "Llama-3.1-8B-Instruct"
PRETRAINED_8B = f"open-unlearning/tofu_{MODEL_8B}_full"
STUDY_8B = "lr_transfer_ATWU"
SUMMARY_KEY_8B = "ATWU_8B"
INPUT_DIM_8B = 4096

# Tuning objective only needs the two extraction strengths — drop the rest.
EVAL_OVERRIDES = [
    "eval=[tofu_repro]",
    "~eval.tofu.metrics.model_utility",
]

PARAM_TO_HYDRA = {
    "lr": "trainer.args.learning_rate",
    "gamma": "trainer.method_args.gamma",
    "alpha": "trainer.method_args.alpha",
    "beta": "trainer.method_args.beta",
}

INITIAL_1B = [
    {"lr": 2e-5,   "gamma": 3.0, "alpha": 1.0,  "beta": 5.0},   # γ/α = 3
    {"lr": 2e-5,   "gamma": 4.0, "alpha": 1.0,  "beta": 7.0},   # γ/α = 4
    {"lr": 2e-5,   "gamma": 5.0, "alpha": 1.0,  "beta": 8.0},   # γ/α = 5
    {"lr": 2e-5,   "gamma": 3.0, "alpha": 0.5,  "beta": 7.0},   # γ/α = 6
    {"lr": 1.5e-5, "gamma": 2.0, "alpha": 0.5,  "beta": 7.0},   # γ/α = 4
    {"lr": 1e-5,   "gamma": 1.0, "alpha": 0.25, "beta": 5.0},   # γ/α = 4
    {"lr": 3e-5,   "gamma": 3.0, "alpha": 1.0,  "beta": 6.0},   # γ/α = 3
    {"lr": 1e-5,   "gamma": 2.0, "alpha": 0.5,  "beta": 8.0},   # γ/α = 4
    {"lr": 1e-5,   "gamma": 3.0, "alpha": 0.5,  "beta": 5.0},   # γ/α = 6
    {"lr": 1.5e-5, "gamma": 5.0, "alpha": 1.0,  "beta": 6.0},   # γ/α = 5
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hpsearch_atwu")


def fixed(input_dim: int) -> list[str]:
    return [
        f"trainer.method_args.scorer.cfg.input_dimension={input_dim}",
        f"trainer.method_args.scorer_trainer.optim_cfg.lr={SCORER_LR}",
        "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5",
        "+trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear",
        "trainer.method_args.scorer_trainer.lambda_entropy=1",
        "trainer.method_args.scorer_trainer.lambda_population=15",
        "trainer.method_args.scorer_trainer.budget=0.2",
        "trainer.method_args.scorer_trainer.lambda_l2=1",
        "trainer.args.per_device_train_batch_size=8",
        "trainer.args.gradient_accumulation_steps=4",
    ]


def run_trial(
    model: str, pretrained: str, input_dim: int,
    trial_number: int, params: dict, gpus: str | None, dry: bool,
) -> float | None:
    tag = "_".join(f"{k}{v:.3g}" for k, v in sorted(params.items()))
    out = str(ARTIFACT_ROOT / model / "ATWU" / f"trial_{trial_number}_{tag}")
    overrides = [f"{PARAM_TO_HYDRA[k]}={v}" for k, v in params.items()]

    train_cmd = [
        "python", "src/train.py", "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default", "trainer=ATWU",
        f"model={model}",
        f"model.model_args.pretrained_model_name_or_path={pretrained}",
        "++model.model_args.device_map='auto'",
        "task_name=_unused_", f"paths.output_dir={out}",
        f"trainer.args.num_train_epochs={NUM_EPOCHS}",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        *overrides, *fixed(input_dim),
    ]
    eval_cmd = [
        "python", "src/eval.py", "--config-name=eval.yaml",
        "experiment=eval/tofu/default", f"model={model}",
        f"model.model_args.pretrained_model_name_or_path={out}",
        "task_name=_unused_", f"paths.output_dir={out}",
        "eval.tofu.overwrite=true", *EVAL_OVERRIDES,
    ]

    log.info(f"trial {trial_number}: {params}")
    if dry:
        log.info("DRY: " + " ".join(train_cmd))
        return 0.0
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    for stage, cmd in [("train", train_cmd), ("eval", eval_cmd)]:
        if subprocess.run(cmd, env=env).returncode != 0:
            log.error(f"  {stage} failed")
            return None
    summary = Path(out) / "TOFU_SUMMARY.json"
    if not summary.exists():
        log.warning(f"  no summary at {summary}")
        return None
    r = json.loads(summary.read_text())
    e, re_ = r.get("extraction_strength"), r.get("retain_extraction_strength")
    if e is None or re_ is None:
        return None
    obj = re_ - e
    log.info(f"  objective={obj:.4f} (re={re_:.4f}, e={e:.4f})")
    return obj


def write_summary(key: str, payload: dict) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = SUMMARY_PATH.with_suffix(".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            existing = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
            existing[key] = payload
            SUMMARY_PATH.write_text(json.dumps(existing, indent=2, default=str))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    log.info(f"summary → {SUMMARY_PATH} [{key}]")
    return SUMMARY_PATH


def _replay_for_resume(sampler: optuna.samplers.BaseSampler, completed: list) -> None:
    """Replay completed trials on a throwaway in-memory study sharing `sampler`.

    Brings the sampler's RNG to the exact state it had at preemption, so the
    next study.ask() returns the same params the preempted (or next) trial
    would have received.

    Enqueued trials (e.g. the user-provided initial) bypass `_sample_relative`
    and don't advance the sampler RNG. Sampled trials DO advance it. To match
    the original run step-for-step, we re-enqueue exactly the trials that were
    enqueued in the original (detected via system_attrs["fixed_params"]) and
    let the sampler propose params for the rest — the sampler will reproduce
    the recorded params byte-for-byte because it's at the same state.
    """
    if not completed:
        return
    burn = optuna.create_study(direction="maximize", sampler=sampler)
    for t in sorted(completed, key=lambda t: t.number):
        if "fixed_params" in t.system_attrs:
            burn.enqueue_trial(t.system_attrs["fixed_params"])
        trial = burn.ask()
        for name, dist in t.distributions.items():
            if isinstance(dist, optuna.distributions.FloatDistribution):
                trial.suggest_float(name, dist.low, dist.high,
                                    log=dist.log, step=dist.step)
            elif isinstance(dist, optuna.distributions.IntDistribution):
                trial.suggest_int(name, dist.low, dist.high,
                                  step=dist.step, log=dist.log)
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                trial.suggest_categorical(name, dist.choices)
        burn.tell(trial, t.value if t.value is not None else 0.0)
    log.info(f"replayed {len(completed)} completed trials to restore RNG state")


def phase1_full_search(n_trials: int, gpus: str | None, dry: bool) -> dict | None:
    log.info("=" * 60 + "\nPhase 1: full HP search on 1B\n" + "=" * 60)
    sampler = optuna.samplers.TPESampler(seed=8)
    study = optuna.create_study(
        study_name=STUDY_1B, direction="maximize",
        sampler=sampler,
        storage=None if dry else DB, load_if_exists=True,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    _replay_for_resume(sampler, completed)
    if not completed:
        for p in INITIAL_1B:
            study.enqueue_trial(p)

    for _ in range(max(n_trials - len(completed), 0)):
        trial = study.ask()
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
            "gamma": trial.suggest_float("gamma", 0.5, 8.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.2, 2.0, log=True),
            "beta": trial.suggest_float("beta", 3.0, 10.0),
        }
        obj = run_trial(
            MODEL_1B, PRETRAINED_1B, INPUT_DIM_1B,
            trial.number, params, gpus, dry,
        )
        if obj is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, obj)

    if dry or study.best_trial is None:
        return None
    b = study.best_trial
    write_summary(SUMMARY_KEY_1B, {
        "method": "ATWU",
        "best_params": {**b.params, "scorer_lr": SCORER_LR},
        "best_objective": b.value,
        "best_trial": b.number, "n_trials": len(study.trials),
    })
    log.info(f"Phase 1 best: {b.params} obj={b.value:.4f}")
    return b.params


def phase2_lr_transfer(source: dict, n_trials: int, gpus: str | None, dry: bool) -> None:
    log.info("=" * 60 + "\nPhase 2: lr-only transfer to 8B\n" + "=" * 60)
    source_lr = source["lr"]
    lr_min, lr_max = source_lr / LR_RELATIVE_RANGE, source_lr * LR_RELATIVE_RANGE
    fixed_non_lr = {k: v for k, v in source.items() if k != "lr" and k in PARAM_TO_HYDRA}
    log.info(f"Source lr={source_lr:.2e}; range [{lr_min:.2e}, {lr_max:.2e}]; "
             f"frozen={fixed_non_lr}")

    sampler = optuna.samplers.TPESampler(seed=8, n_startup_trials=5)
    study = optuna.create_study(
        study_name=STUDY_8B, direction="maximize",
        sampler=sampler,
        storage=None if dry else DB, load_if_exists=True,
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    _replay_for_resume(sampler, completed)
    seeds = [
        source_lr,
        source_lr + 1e-5,
        source_lr - 1e-5,
        source_lr + 5e-6,
        source_lr - 5e-6,
    ]
    for lr in seeds[len(completed):]:
        study.enqueue_trial({"lr": lr})

    for _ in range(max(n_trials - len(completed), 0)):
        trial = study.ask()
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        params = {"lr": lr, **fixed_non_lr}
        obj = run_trial(
            MODEL_8B, PRETRAINED_8B, INPUT_DIM_8B,
            trial.number, params, gpus, dry,
        )
        if obj is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, obj)

    if dry or study.best_trial is None:
        return
    b = study.best_trial
    write_summary(SUMMARY_KEY_8B, {
        "method": "ATWU",
        "best_params": {"lr": b.params["lr"], "scorer_lr": SCORER_LR, **fixed_non_lr},
        "best_objective": b.value, "best_trial": b.number,
        "n_trials": len(study.trials), "source_params": source,
    })
    log.info(f"Phase 2 best: lr={b.params['lr']:.2e} obj={b.value:.4f}")


def load_source_from_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())[SUMMARY_KEY_1B]["best_params"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=50, help="Phase 1 trials (default 50)")
    p.add_argument("--n-trials-transfer", type=int, default=5, help="Phase 2 trials (default 5)")
    p.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-phase1", action="store_true",
                   help="Skip 1B search; reuse existing 1B summary as source")
    p.add_argument("--skip-phase2", action="store_true", help="Skip 8B lr transfer")
    args = p.parse_args()

    if args.skip_phase1:
        source = load_source_from_summary()
        log.info(f"Loaded 1B source params: {source}")
    else:
        source = phase1_full_search(args.n_trials, args.gpus, args.dry_run)
        if source is None and not args.dry_run:
            log.error("Phase 1 produced no result; skipping Phase 2")
            return

    if not args.skip_phase2:
        phase2_lr_transfer(source or INITIAL_1B[0], args.n_trials_transfer,
                           args.gpus, args.dry_run)


if __name__ == "__main__":
    main()
