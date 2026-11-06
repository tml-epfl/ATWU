#!/usr/bin/env python
"""Shared command builders and process helpers for the WMDP experiments."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJ = Path(__file__).resolve().parents[2]
PY = os.environ.get("ATWU_PYTHON", sys.executable)
STATE = Path(os.environ.get("WMDP_STATE", "saves/wmdp"))
if not STATE.is_absolute():
    STATE = PROJ / STATE

MMLU_TARGET = 0.57
WMDP_TARGET = 0.27

ATWU_RUN_DEFAULTS = {
    "max_steps": 150,
    "alternate_every": 1,
    "first_topic": "cyber",
    "bio_max_length": 768,
    "cyber_max_length": 512,
    "per_device_batch": 8,
    "grad_accum": 4,
    "retain_replacement": False,
}
ATWU_REQUIRED = {
    "lr",
    "alpha",
    "beta",
    "gamma",
    "scorer_lr",
    "budget",
    "lambda_population",
    "lambda_entropy",
    "lambda_l2",
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


SEARCH_METRICS = ("mmlu", "wmdp_bio", "wmdp_cyber", "wmdp_mean")


def top_trial(trials, n_trials):
    """Return the best of the first ``n_trials`` completed trials.

    Ties go to the lower trial number. Returns None until ``n_trials`` trials
    have completed.
    """
    completed = sorted(
        (t for t in trials if t.state.name == "COMPLETE" and t.value is not None),
        key=lambda t: t.number,
    )[:n_trials]
    if len(completed) < n_trials:
        return None
    return max(completed, key=lambda t: (t.value, -t.number))


def write_selected(method, trial, state=None) -> Path:
    """Record one method's top trial in ``<state>/selected.json``.

    Uses the ``selected_trial``/``selected_params`` keys of
    ``results/wmdp.json``, so the file can be passed to
    ``evaluate.py --params``. Several searches may write concurrently.
    """
    state = Path(state or STATE)
    path = state / "selected.json"
    with (state / ".selected.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {"methods": {}}
        data["methods"][method] = {
            "selected_trial": trial.number,
            "selected_params": dict(trial.params),
            "search": {
                "seed": 0,
                "score": trial.value,
                "metrics": {key: trial.user_attrs.get(key) for key in SEARCH_METRICS},
            },
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, path)
    return path


def run_cmd(cmd, logfile, timeout=None, env=None) -> int:
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    process_env = dict(os.environ)
    process_env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "FLASH_ATTENTION_DETERMINISTIC": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TRITON_CACHE_DIR": "/tmp/triton_cache_wmdp",
        }
    )
    if env:
        process_env.update(env)
    with logfile.open("a") as handle:
        handle.write(f"\n\n===== {now()} :: {' '.join(map(str, cmd))}\n")
        handle.flush()
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=PROJ,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=process_env,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return 124


def wait_for_gpu(min_free_mib=60000, timeout=7200, gpu_index=0) -> bool:
    started = time.time()
    while time.time() - started < timeout:
        try:
            rows = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().splitlines()
            used, total = [int(value) for value in rows[gpu_index].split(",")]
            if total - used >= min_free_mib:
                return True
        except Exception as error:
            log(f"[gpu] query failed: {error}")
        log(f"[gpu {gpu_index}] busy; checking again in 120 seconds")
        time.sleep(120)
    return False


def corpus_overrides(corpus_dir: str | Path | None) -> list[str]:
    if corpus_dir is None:
        return []
    root = Path(corpus_dir).resolve()
    return [
        f"gen_bio_file={root / 'bio-forget.text.jsonl'}",
        f"gen_cyber_file={root / 'cyber-forget.text.jsonl'}",
        f"gen_retain_file={root / 'general-retain.text.jsonl'}",
    ]


def atwu_train_args(
    out_dir,
    hp,
    seed=0,
    experiment="gen_corpora",
    corpus_dir=None,
):
    """Build the ATWU training command for one configuration and seed."""
    missing = sorted(ATWU_REQUIRED - hp.keys())
    if missing:
        raise ValueError(f"missing ATWU parameters: {', '.join(missing)}")
    config = dict(ATWU_RUN_DEFAULTS)
    config.update(hp)
    block_size = (
        config["alternate_every"]
        * config["per_device_batch"]
        * config["grad_accum"]
    )
    consumed = (
        config["max_steps"]
        * config["per_device_batch"]
        * config["grad_accum"]
    )
    args = [
        PY,
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment=unlearn/wmdp/{experiment}",
        "trainer=ATWU",
        "model=zephyr-7b-beta",
        "model.model_args.pretrained_model_name_or_path=HuggingFaceH4/zephyr-7b-beta",
        "++model.model_args.device_map=auto",
        "data_split=cyber",
        "task_name=_unused_",
        f"paths.output_dir={out_dir}",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        f"trainer.args.learning_rate={config['lr']}",
        f"trainer.method_args.gamma={config['gamma']}",
        f"trainer.method_args.alpha={config['alpha']}",
        f"trainer.method_args.beta={config['beta']}",
        f"trainer.method_args.scorer_trainer.optim_cfg.lr={config['scorer_lr']}",
        "~trainer.method_args.steering_coeff",
        "~trainer.method_args.retain_loss_type",
        "~trainer.method_args.module_regex",
        "~trainer.method_args.trainable_params_regex",
        "trainer.method_args.scorer.cfg.input_dimension=4096",
        "trainer.method_args.canonical_ce=true",
        "trainer.method_args.scorer_trainer.optim_cfg.update_every_n_steps=5",
        "+trainer.method_args.scorer_trainer.optim_cfg.scheduler=linear",
        f"trainer.method_args.scorer_trainer.lambda_entropy={config['lambda_entropy']}",
        f"trainer.method_args.scorer_trainer.lambda_population={config['lambda_population']}",
        f"trainer.method_args.scorer_trainer.budget={config['budget']}",
        f"trainer.method_args.scorer_trainer.lambda_l2={config['lambda_l2']}",
        f"trainer.args.max_steps={config['max_steps']}",
        f"trainer.args.per_device_train_batch_size={config['per_device_batch']}",
        f"trainer.args.gradient_accumulation_steps={config['grad_accum']}",
        "trainer.args.gradient_checkpointing=True",
        "trainer.args.optim=paged_adamw_8bit",
        f"trainer.args.seed={seed}",
        f"data.alternating_topics.block_size={block_size}",
        f"data.alternating_topics.seed={seed}",
        f"data.alternating_topics.first_topic={config['first_topic']}",
        f"data.forget.bio.args.max_length={config['bio_max_length']}",
        f"data.forget.cyber.args.max_length={config['cyber_max_length']}",
        "++data.alternating_topics.subset_seed=0",
        f"++data.alternating_topics.subset_consumed={consumed}",
    ]
    if not config["retain_replacement"]:
        args.extend(
            [
                "++data.retain_replacement=false",
                f"++data.retain_sample_seed={seed}",
            ]
        )
    args.extend(corpus_overrides(corpus_dir))
    return args


def eval_args(out_dir):
    return [
        PY,
        "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/wmdp/default",
        "model=zephyr-7b-beta",
        f"model.model_args.pretrained_model_name_or_path={out_dir}",
        "data_split=cyber",
        "task_name=_unused_",
        f"paths.output_dir={out_dir}",
        "eval.lm_eval.tasks=[wmdp_cyber,wmdp_bio,mmlu]",
        "eval.lm_eval.overwrite=true",
    ]


def cleanup_weights(out_dir) -> int:
    freed = 0
    for path in Path(out_dir).glob("*"):
        if path.suffix in (".safetensors", ".bin") or path.name in (
            "model.safetensors.index.json",
            "LMEval_EVAL.json",
            "optimizer.pt",
            "scheduler.pt",
        ):
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:
                pass
    for path in Path(out_dir).glob("checkpoint-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    return freed


# --------------------------------------------------------------------------
# Generated corpus

DEFAULT_CORPUS = PROJ / "data" / "wmdp" / "generated"
CORPUS_ROWS = {
    "bio-forget.text.jsonl": 3200,
    "cyber-forget.text.jsonl": 3200,
    "general-retain.text.jsonl": 4000,
}


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def ensure_corpus(output_dir: Path = DEFAULT_CORPUS, quiet: bool = False) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    paths = {name: output_dir / name for name in CORPUS_ROWS}
    problems = []
    counts = {}
    for name, minimum in CORPUS_ROWS.items():
        path = paths[name]
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        counts[name] = _line_count(path)
        if counts[name] < minimum:
            problems.append(f"{path}: {counts[name]} rows; expected at least {minimum}")

    if problems:
        details = "\n".join(f"- {problem}" for problem in problems)
        raise SystemExit(
            f"WMDP generated corpora are not ready:\n{details}\n"
            "Run: python scripts/wmdp/generate_corpus.py"
        )
    if not quiet:
        summary = ", ".join(f"{name}={counts[name]}" for name in CORPUS_ROWS)
        print(f"WMDP generated corpora ready: {summary}")
    return paths


# --------------------------------------------------------------------------
# Baseline training commands (shared by tune_baselines.py and evaluate.py)

EXPERIMENT = "gen_corpora"
MODEL = "zephyr-7b-beta"
PRETRAINED = "HuggingFaceH4/zephyr-7b-beta"
MAX_STEPS = 150
PER_DEVICE_BATCH = 8
GRAD_ACCUM = 4
BLOCK_SIZE = PER_DEVICE_BATCH * GRAD_ACCUM
SUBSET_CONSUMED = MAX_STEPS * PER_DEVICE_BATCH * GRAD_ACCUM

HYDRA_PATHS = {
    "lr": "trainer.args.learning_rate",
    "gamma": "trainer.method_args.gamma",
    "alpha": "trainer.method_args.alpha",
    "beta": "trainer.method_args.beta",
    "delta": "trainer.method_args.delta",
    "beta1": "trainer.method_args.beta1",
    "beta2": "trainer.method_args.beta2",
    "steering_coeff": "trainer.method_args.steering_coeff",
}
DELETE_RMU_KEYS = [
    "~trainer.method_args.steering_coeff",
    "~trainer.method_args.module_regex",
    "~trainer.method_args.trainable_params_regex",
]

BASELINES = {
    "graddiff": {
        "trainer": "GradDiff",
        "fixed": ["trainer.method_args.retain_loss_type=NLL"],
        "delete": DELETE_RMU_KEYS,
    },
    "jensun": {
        "trainer": "JensUn",
        "fixed": ["trainer.method_args.retain_loss_type=JensUn"],
        "delete": DELETE_RMU_KEYS,
    },
    "wga": {
        "trainer": "WGA",
        "fixed": ["trainer.method_args.retain_loss_type=NLL"],
        "delete": DELETE_RMU_KEYS,
    },
    "satimp": {
        "trainer": "SatImp",
        "fixed": ["trainer.method_args.retain_loss_type=NLL"],
        "delete": DELETE_RMU_KEYS,
    },
    "npo": {
        "trainer": "NPO",
        "fixed": ["trainer.method_args.retain_loss_type=NLL"],
        "delete": DELETE_RMU_KEYS,
    },
    "simnpo": {
        "trainer": "SimNPO",
        "fixed": ["trainer.method_args.retain_loss_type=NLL"],
        "delete": DELETE_RMU_KEYS,
    },
    "rmu": {
        "trainer": "RMU",
        "fixed": ["trainer.method_args.retain_loss_type=EMBED_DIFF"],
        "delete": [],
    },
}


def baseline_train_args(method, params, out_dir, seed, corpus_dir=None):
    """Build the training command for one baseline configuration and seed."""
    spec = BASELINES[method]
    consumed = set()
    overrides = []
    if method == "rmu":
        layer = int(params["layer"])
        consumed.add("layer")
        overrides.extend(
            [
                rf"trainer.method_args.module_regex=model\.layers\.{layer}",
                r"trainer.method_args.trainable_params_regex="
                rf'["model\.layers\.({layer - 2}|{layer - 1}|{layer})\.mlp\.down_proj\.weight"]',
            ]
        )
    overrides.extend(
        f"{HYDRA_PATHS[key]}={value}"
        for key, value in params.items()
        if key not in consumed
    )
    return [
        PY,
        "src/train.py",
        "--config-name=unlearn.yaml",
        f"experiment=unlearn/wmdp/{EXPERIMENT}",
        f"trainer={spec['trainer']}",
        f"model={MODEL}",
        f"model.model_args.pretrained_model_name_or_path={PRETRAINED}",
        "++model.model_args.device_map=auto",
        "data_split=cyber",
        "task_name=_unused_",
        f"paths.output_dir={out_dir}",
        "trainer.args.eval_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        f"trainer.args.max_steps={MAX_STEPS}",
        f"trainer.args.per_device_train_batch_size={PER_DEVICE_BATCH}",
        f"trainer.args.gradient_accumulation_steps={GRAD_ACCUM}",
        "trainer.args.gradient_checkpointing=True",
        "trainer.args.optim=paged_adamw_8bit",
        f"trainer.args.seed={seed}",
        f"data.alternating_topics.block_size={BLOCK_SIZE}",
        f"data.alternating_topics.seed={seed}",
        "data.alternating_topics.first_topic=cyber",
        "data.forget.bio.args.max_length=768",
        "data.forget.cyber.args.max_length=512",
        "++data.alternating_topics.subset_seed=0",
        f"++data.alternating_topics.subset_consumed={SUBSET_CONSUMED}",
        "++data.retain_replacement=false",
        f"++data.retain_sample_seed={seed}",
        *overrides,
        *spec["fixed"],
        *spec["delete"],
        *corpus_overrides(corpus_dir),
    ]
