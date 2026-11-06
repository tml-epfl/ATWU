#!/usr/bin/env python
"""Run a baseline Optuna search for WMDP and select its top trial.

The search runs 20 trials at seed 0 by default. The first trial uses the
method's initial parameters from preliminary sweeps. Once all trials have
completed, the trial with the highest objective is written to
``$WMDP_STATE/selected.json``. Running the command again on a finished study
only rewrites that method's entry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hpsearch_baselines")

STATE = common.STATE
TRAIN_TIMEOUT = 3 * 3600
EVAL_TIMEOUT = 3 * 3600
EVAL_ATTEMPTS = int(os.environ.get("EVAL_MAX_ATTEMPTS", "5"))
EVAL_RETRY_SLEEP_S = int(os.environ.get("EVAL_RETRY_SLEEP_S", "60"))

# Resume and retry helpers for interrupted or failed trials.

MAX_TRIAL_ATTEMPTS = int(os.environ.get("MAX_TRIAL_ATTEMPTS", "3"))

COMPLETE = optuna.trial.TrialState.COMPLETE
FAIL = optuna.trial.TrialState.FAIL
WAITING = optuna.trial.TrialState.WAITING
RUNNING = optuna.trial.TrialState.RUNNING


def count_complete(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == COMPLETE)


def count_failed(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == FAIL)


def under_fail_budget(study: optuna.Study, n_trials: int) -> bool:
    """Safety valve: stop the while-loop if FAILED trials exceed n_trials, which
    signals a systematic problem rather than transient noise."""
    nf = count_failed(study)
    if nf > n_trials:
        log.error(
            f"{nf} FAILED trials exceeds budget (n_trials={n_trials}); stopping to "
            f"avoid a runaway loop. Inspect the failures before re-running."
        )
        return False
    return True


def _reset_interrupted(study: optuna.Study) -> int:
    """Move interrupted RUNNING trials back to WAITING."""
    interrupted = [t for t in study.trials if t.state == RUNNING]
    for t in interrupted:
        study._storage.set_trial_state_values(t._trial_id, WAITING)
    if interrupted:
        log.info(
            f"reset {len(interrupted)} RUNNING trial(s) → WAITING: "
            f"{[t.number for t in interrupted]}"
        )
    return len(interrupted)


def _replay_for_resume(sampler: optuna.samplers.BaseSampler, trials_to_replay: list) -> None:
    """Replay trials on a throwaway study sharing `sampler` to restore its RNG.

    Enqueued initial trials carry `fixed_params` → re-enqueued in the burn
    study (no RNG draw, matching the original). TPE-proper trials re-issue their
    suggest_* calls (consuming RNG exactly as before). Interrupted WAITING trials
    have value None → a 0.0 dummy is used (the sampler RNG is independent of the
    told value; the real KDE is rebuilt in the main study)."""
    if not trials_to_replay:
        return
    burn = optuna.create_study(direction="maximize", sampler=sampler)
    for t in sorted(trials_to_replay, key=lambda t: t.number):
        if "fixed_params" in t.system_attrs:
            burn.enqueue_trial(t.system_attrs["fixed_params"])
        trial = burn.ask()
        for name, dist in t.distributions.items():
            if isinstance(dist, optuna.distributions.FloatDistribution):
                trial.suggest_float(name, dist.low, dist.high, log=dist.log, step=dist.step)
            elif isinstance(dist, optuna.distributions.IntDistribution):
                trial.suggest_int(name, dist.low, dist.high, step=dist.step, log=dist.log)
            elif isinstance(dist, optuna.distributions.CategoricalDistribution):
                trial.suggest_categorical(name, dist.choices)
        burn.tell(trial, t.value if t.value is not None else 0.0)
    log.info(f"replayed {len(trials_to_replay)} trial(s) to restore sampler RNG state")


def resume(study: optuna.Study, sampler: optuna.samplers.BaseSampler, dry: bool = False) -> None:
    """Reset interrupted trials and restore the sampler RNG before resuming."""
    if not dry:
        _reset_interrupted(study)
    completed = [t for t in study.trials if t.state == COMPLETE]
    waiting = [t for t in study.trials if t.state == WAITING and t.params]
    _replay_for_resume(sampler, completed + waiting)


def tell_or_retry(study: optuna.Study, trial, obj, label: str = "") -> bool:
    """obj is not None → tell COMPLETE (return True). Else bump the trial's attempt
    counter and, if under MAX_TRIAL_ATTEMPTS, reset it to WAITING to rerun with the
    same params; otherwise mark FAIL. Returns whether the trial completed."""
    if obj is not None:
        study.tell(trial, obj)
        return True
    n = trial.user_attrs.get("attempts", 0) + 1
    trial.set_user_attr("attempts", n)
    if n < MAX_TRIAL_ATTEMPTS:
        log.warning(
            f"{label}trial {trial.number} failed (attempt {n}/{MAX_TRIAL_ATTEMPTS}); "
            f"retrying with the SAME params"
        )
        study._storage.set_trial_state_values(trial._trial_id, WAITING)
    else:
        log.error(f"{label}trial {trial.number} failed {n}x; marking FAIL")
        study.tell(trial, state=FAIL)
    return False


def score_from(mmlu, bio, cyber):
    return mmlu - ((max(0.25, bio) + max(0.25, cyber)) / 2 - 0.25)


def _alpha_gamma(trial):
    return {
        "alpha": trial.suggest_float("alpha", 0.1, 3.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.1, 5.0, log=True),
    }


def _graddiff(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1.5e-5, log=True),
        **_alpha_gamma(trial),
    }


def _jensun(trial):
    return _graddiff(trial)


def _wga(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        **_alpha_gamma(trial),
        "beta": trial.suggest_float("beta", 0.1, 5.0, log=True),
    }


def _satimp(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-6, 1e-5, log=True),
        **_alpha_gamma(trial),
        "beta1": trial.suggest_float("beta1", 1.0, 10.0, log=True),
        "beta2": trial.suggest_float("beta2", 0.1, 5.0, log=True),
    }


def _npo(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        **_alpha_gamma(trial),
        "beta": trial.suggest_float("beta", 0.05, 0.2, log=True),
    }


def _simnpo(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        **_alpha_gamma(trial),
        "beta": trial.suggest_float("beta", 2.0, 3.0, log=True),
        "delta": trial.suggest_float("delta", 0.0, 2.0),
    }


def _rmu(trial):
    return {
        "lr": trial.suggest_float("lr", 5e-6, 5e-4, log=True),
        **_alpha_gamma(trial),
        "steering_coeff": trial.suggest_float("steering_coeff", 0.5, 10.0, log=True),
        "layer": trial.suggest_int("layer", 4, 12),
    }


METHOD_SPECS = {
    "graddiff": {
        "search": _graddiff,
        "initial_params": {"lr": 9.409e-6, "alpha": 2.0, "gamma": 4.80},
    },
    "jensun": {
        "search": _jensun,
        "initial_params": {"lr": 9.011e-6, "alpha": 0.6745, "gamma": 2.498},
    },
    "wga": {
        "search": _wga,
        "initial_params": {
            "lr": 6.258e-6,
            "alpha": 1.548,
            "gamma": 1.125,
            "beta": 0.2141,
        },
    },
    "satimp": {
        "search": _satimp,
        "initial_params": {
            "lr": 6.353e-6,
            "alpha": 0.5,
            "gamma": 1.5,
            "beta1": 1.829,
            "beta2": 0.5,
        },
    },
    "npo": {
        "search": _npo,
        "initial_params": {
            "lr": 2.60e-5,
            "alpha": 0.6337,
            "gamma": 4.760,
            "beta": 0.07055,
        },
    },
    "simnpo": {
        "search": _simnpo,
        "initial_params": {
            "lr": 2.082e-5,
            "alpha": 0.4577,
            "gamma": 2.426,
            "beta": 2.591,
            "delta": 1.501,
        },
    },
    "rmu": {
        "search": _rmu,
        "initial_params": {
            "lr": 8.259e-6,
            "alpha": 1.5,
            "gamma": 0.4668,
            "steering_coeff": 9.862,
            "layer": 5,
        },
    },
}


def _run_config(method, params, name, seed, gpus, corpus_dir, dry=False):
    out = STATE / "baselines" / method / name
    command = common.baseline_train_args(method, params, out, seed, corpus_dir)
    if dry:
        print(" ".join(command))
        return None
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_cmd.txt").write_text(" ".join(command) + "\n")

    gpu_index = int(str(gpus).split(",")[0]) if gpus is not None else 0
    env = {"CUDA_VISIBLE_DEVICES": str(gpus)} if gpus is not None else None
    if not common.wait_for_gpu(gpu_index=gpu_index):
        raise RuntimeError(f"GPU {gpu_index} never became free")
    if not any(out.glob("*.safetensors")):
        rc = common.run_cmd(command, out / "train.log", timeout=TRAIN_TIMEOUT, env=env)
        if rc != 0 or not any(out.glob("*.safetensors")):
            common.cleanup_weights(out)
            return None

    summary = out / "LMEval_SUMMARY.json"
    for attempt in range(1, EVAL_ATTEMPTS + 1):
        if summary.exists():
            summary.unlink()
        rc = common.run_cmd(
            common.eval_args(out), out / "eval.log", timeout=EVAL_TIMEOUT, env=env
        )
        if rc == 0 and summary.exists():
            break
        if attempt < EVAL_ATTEMPTS:
            delay = min(EVAL_RETRY_SLEEP_S * (2 ** (attempt - 1)), 300)
            log.warning("%s: evaluation failed; retrying in %ds", name, delay)
            time.sleep(delay)
    else:
        os._exit(75)

    metrics = json.loads(summary.read_text())
    common.cleanup_weights(out)
    if any(
        metrics.get(key) is None
        for key in ("mmlu/acc", "wmdp_bio/acc", "wmdp_cyber/acc")
    ):
        return None
    return (
        metrics["mmlu/acc"],
        metrics["wmdp_bio/acc"],
        metrics["wmdp_cyber/acc"],
    )


def _study_paths(method):
    suffix = f"_{method}_ms{common.MAX_STEPS}_aev1_ftcyber"
    return (
        STATE / f"hpsearch_baselines{suffix}.db",
        f"baselines_gencorp{suffix}",
        STATE / f"baselines_trials_{method}.jsonl",
    )


def _append_trial(path, record):
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def search(method, n_trials, gpus, corpus_dir, dry=False):
    db, study_name, trials_path = _study_paths(method)
    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=5)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        storage=None if dry else f"sqlite:///{db}",
        load_if_exists=True,
    )
    resume(study, sampler, dry=dry)
    if not study.trials:
        study.enqueue_trial(METHOD_SPECS[method]["initial_params"])
        log.info("enqueued initial parameters for %s", method)

    spec = METHOD_SPECS[method]
    while count_complete(study) < n_trials and under_fail_budget(
        study, n_trials
    ):
        trial = study.ask()
        params = spec["search"](trial)
        result = _run_config(
            method,
            params,
            f"trial_{trial.number}",
            seed=0,
            gpus=gpus,
            corpus_dir=corpus_dir,
            dry=dry,
        )
        if dry:
            study.tell(trial, 0.0)
            return
        if result is None:
            tell_or_retry(study, trial, None)
            continue
        mmlu, bio, cyber = result
        wmdp = (bio + cyber) / 2
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
            trials_path,
            {
                "trial": trial.number,
                "method": method,
                "params": params,
                "mmlu": mmlu,
                "wmdp_bio": bio,
                "wmdp_cyber": cyber,
                "wmdp_mean": wmdp,
                "score": score,
                "passed": passed,
                "ts": common.now(),
            },
        )
        tell_or_retry(study, trial, score)

    if dry:
        return
    top = common.top_trial(study.trials, n_trials)
    if top is None:
        raise SystemExit(
            f"{method}: only {count_complete(study)} of {n_trials} trials "
            "completed; no trial selected"
        )
    path = common.write_selected(method, top)
    log.info("%s top trial %s: score=%+.4f -> %s", method, top.number, top.value, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPECS))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=common.DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    common.ensure_corpus(args.corpus_dir, quiet=True)
    if not args.dry_run:
        STATE.mkdir(parents=True, exist_ok=True)
    search(args.method, args.n_trials, args.gpus, args.corpus_dir, args.dry_run)


if __name__ == "__main__":
    main()
