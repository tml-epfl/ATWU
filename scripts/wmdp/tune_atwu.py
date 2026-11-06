#!/usr/bin/env python
"""Run the ATWU Optuna search for WMDP and select its top trial.

The search runs 20 trials at seed 0 by default. The first trial uses initial
parameters from preliminary sweeps; the initial values and search ranges are
defined below. Once all trials have completed, the trial with the highest
objective is written to ``$WMDP_STATE/selected.json``. Running the command
again on a finished study only rewrites that file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402


EXPERIMENT = "gen_corpora"
STATE = common.STATE
DB = STATE / "optuna_singlestep.db"
TRIALS = STATE / "optuna_trials.jsonl"
STUDY = "singlestep_hp"
TRAIN_TIMEOUT = 3 * 3600
EVAL_TIMEOUT = 3 * 3600
EVAL_ATTEMPTS = int(os.environ.get("EVAL_MAX_ATTEMPTS", "5"))
EVAL_RETRY_SLEEP_S = int(os.environ.get("EVAL_RETRY_SLEEP_S", "60"))

FIXED_HP = {"alternate_every": 1, "retain_replacement": False}

# Enqueued first on a new study. These values came from preliminary sweeps.
INITIAL_PARAMS = {
    "lr": 4.7619925960174024e-06,
    "alpha": 0.1,
    "gamma": 10.0,
    "beta": 2.386643331698242,
    "scorer_lr": 0.008162511373314002,
    "budget": 0.4,
    "lambda_population": 10.0,
    "lambda_entropy": 1.0,
    "lambda_l2": 1.0,
}


def suggest(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.05, 0.18),
        "gamma": trial.suggest_float("gamma", 8.0, 12.0),
        "budget": trial.suggest_float("budget", 0.30, 0.50),
        "lambda_population": trial.suggest_float("lambda_population", 6.0, 12.0),
        "lr": trial.suggest_float("lr", 2.0e-6, 1.2e-5, log=True),
        "beta": trial.suggest_float("beta", 1.0, 5.0),
        "scorer_lr": trial.suggest_float("scorer_lr", 3.0e-3, 3.0e-2, log=True),
        "lambda_entropy": trial.suggest_float("lambda_entropy", 0.3, 2.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 2.0, log=True),
    }


def score_from(mmlu, bio, cyber):
    wmdp_floored = (max(0.25, bio) + max(0.25, cyber)) / 2
    return mmlu - (wmdp_floored - 0.25)


def _complete_count(study):
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def _reset_interrupted(study):
    """Return interrupted RUNNING trials to WAITING before resuming one worker."""
    running = optuna.trial.TrialState.RUNNING
    waiting = optuna.trial.TrialState.WAITING
    interrupted = [trial for trial in study.trials if trial.state == running]
    for trial in interrupted:
        study._storage.set_trial_state_values(trial._trial_id, waiting)
    if interrupted:
        common.log(
            "[optuna] reset interrupted trials to WAITING: "
            f"{[trial.number for trial in interrupted]}"
        )


def _append_trial(record):
    with TRIALS.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run(name, hp, seed, gpus, corpus_dir):
    """Train and evaluate one configuration at one seed."""
    out_dir = STATE / "optuna" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_index = int(str(gpus).split(",")[0]) if gpus is not None else 0
    if not common.wait_for_gpu(gpu_index=gpu_index):
        raise RuntimeError(f"{name}: GPU {gpu_index} never became free")
    env = {"CUDA_VISIBLE_DEVICES": str(gpus)} if gpus is not None else None

    if any(out_dir.glob("*.safetensors")):
        common.log(f"[optuna] {name}: checkpoint present; retrying evaluation")
    else:
        command = common.atwu_train_args(
            out_dir,
            hp=hp,
            seed=seed,
            experiment=EXPERIMENT,
            corpus_dir=corpus_dir,
        )
        (out_dir / "train_cmd.txt").write_text(" ".join(command) + "\n")
        rc = common.run_cmd(
            command, out_dir / "train.log", timeout=TRAIN_TIMEOUT, env=env
        )
        if rc != 0 or not any(out_dir.glob("*.safetensors")):
            common.cleanup_weights(out_dir)
            common.log(f"[optuna] {name}: training failed")
            os._exit(75)

    summary = out_dir / "LMEval_SUMMARY.json"
    for attempt in range(1, EVAL_ATTEMPTS + 1):
        if summary.exists():
            summary.unlink()
        rc = common.run_cmd(
            common.eval_args(out_dir),
            out_dir / "eval.log",
            timeout=EVAL_TIMEOUT,
            env=env,
        )
        if rc == 0 and summary.exists():
            break
        if attempt < EVAL_ATTEMPTS:
            delay = min(EVAL_RETRY_SLEEP_S * (2 ** (attempt - 1)), 300)
            common.log(
                f"[optuna] {name}: evaluation failed ({attempt}/{EVAL_ATTEMPTS}); "
                f"retrying in {delay}s"
            )
            time.sleep(delay)
    else:
        common.log(f"[optuna] {name}: evaluation failed {EVAL_ATTEMPTS} times")
        os._exit(75)

    metrics = json.loads(summary.read_text())
    common.cleanup_weights(out_dir)
    return (
        metrics["mmlu/acc"],
        metrics["wmdp_bio/acc"],
        metrics["wmdp_cyber/acc"],
    )


def objective(trial, gpus, corpus_dir):
    search_params = suggest(trial)
    hp = dict(FIXED_HP, **search_params)
    name = f"opt_t{trial.number}"
    common.log(f"[optuna] trial {trial.number} start hp={json.dumps(search_params)}")
    mmlu, bio, cyber = _run(name, hp, seed=0, gpus=gpus, corpus_dir=corpus_dir)
    wmdp = (bio + cyber) / 2.0
    score = score_from(mmlu, bio, cyber)
    passed = mmlu >= common.MMLU_TARGET and wmdp <= common.WMDP_TARGET
    for key, value in (
        ("mmlu", mmlu),
        ("wmdp_bio", bio),
        ("wmdp_cyber", cyber),
        ("wmdp_mean", wmdp),
        ("passed", passed),
    ):
        trial.set_user_attr(key, value)
    _append_trial(
        {
            "trial": trial.number,
            "params": search_params,
            "mmlu": mmlu,
            "wmdp_bio": bio,
            "wmdp_cyber": cyber,
            "wmdp_mean": wmdp,
            "score": score,
            "passed": passed,
            "ts": common.now(),
        }
    )
    common.log(
        f"[optuna] trial {trial.number}: mmlu={mmlu:.4f} "
        f"wmdp={wmdp:.4f} score={score:+.4f} pass={passed}"
    )
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=common.DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    common.ensure_corpus(args.corpus_dir, quiet=True)
    if args.dry_run:
        command = common.atwu_train_args(
            STATE / "optuna" / "opt_t0",
            hp=dict(FIXED_HP, **INITIAL_PARAMS),
            corpus_dir=args.corpus_dir,
        )
        print(" ".join(command))
        return

    STATE.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY,
        storage=f"sqlite:///{DB}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=5),
    )
    if not study.trials:
        study.enqueue_trial(INITIAL_PARAMS)
        common.log("[optuna] enqueued initial parameters")

    _reset_interrupted(study)
    initial_complete = any(
        trial.state == optuna.trial.TrialState.COMPLETE
        and trial.params == INITIAL_PARAMS
        for trial in study.trials
    )
    initial_pending = any(
        trial.state in (optuna.trial.TrialState.WAITING, optuna.trial.TrialState.RUNNING)
        and (
            trial.params == INITIAL_PARAMS
            or trial.system_attrs.get("fixed_params") == INITIAL_PARAMS
        )
        for trial in study.trials
    )
    if not initial_complete and not initial_pending:
        study.enqueue_trial(INITIAL_PARAMS)
        initial_pending = True

    complete = _complete_count(study)
    remaining = max(0, args.n_trials - complete)
    if not initial_complete and initial_pending:
        remaining = max(1, remaining)
    if remaining:
        study.optimize(
            lambda trial: objective(trial, args.gpus, args.corpus_dir),
            n_trials=remaining,
        )
    top = common.top_trial(study.trials, args.n_trials)
    if top is None:
        raise SystemExit(
            f"only {_complete_count(study)} of {args.n_trials} trials completed; "
            "no trial selected"
        )
    path = common.write_selected("atwu", top)
    common.log(f"[optuna] top trial {top.number}: score={top.value:+.4f} -> {path}")


if __name__ == "__main__":
    main()
