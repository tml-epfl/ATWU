#!/usr/bin/env python3
"""Consolidate final unlearning runs into a single short-form JSON.

Walks a runs directory laid out as <model>/<split>/<method> and, per run,
collects tofu_summary + paraphrase_summary + paraphrase_eval + training_stats
+ log_history + training_config. "Short" means:

  - tofu_eval (full TOFU_EVAL.json) is dropped.
  - mia* keys are stripped from paraphrase_summary and paraphrase_eval.

Usage:
    python scripts/consolidate_final_runs_short.py
    python scripts/consolidate_final_runs_short.py --runs-dir saves/unlearn/ATWU_TOFU --output my.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent


def load_json_safe(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_yaml_safe(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def strip_secrets(obj):
    """Recursively strip keys containing 'token', 'secret', 'password' from dicts."""
    if isinstance(obj, dict):
        return {
            k: strip_secrets(v)
            for k, v in obj.items()
            if not any(s in k.lower() for s in ("token", "secret", "password"))
        }
    elif isinstance(obj, list):
        return [strip_secrets(v) for v in obj]
    return obj


def extract_training_stats(trainer_state: dict) -> dict:
    """Extract summary training statistics from trainer_state.json."""
    log_history = trainer_state.get("log_history", [])

    # The final entry with 'train_loss' is the summary entry
    summary_entry = None
    for entry in reversed(log_history):
        if "train_loss" in entry:
            summary_entry = entry
            break

    # Last step entry (with 'loss' key) for final loss
    last_step = None
    for entry in reversed(log_history):
        if "loss" in entry:
            last_step = entry
            break

    stats = {
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "epoch": trainer_state.get("epoch"),
        "train_batch_size": trainer_state.get("train_batch_size"),
    }
    if summary_entry:
        stats["train_loss"] = summary_entry.get("train_loss")
        stats["train_runtime"] = summary_entry.get("train_runtime")
        stats["train_samples_per_second"] = summary_entry.get("train_samples_per_second")
    if last_step:
        stats["final_step_loss"] = last_step.get("loss")

    return stats


def extract_training_config(hydra_config: dict) -> dict:
    """Extract training config from .hydra/config.yaml."""
    trainer_cfg = hydra_config.get("trainer", {})
    args = trainer_cfg.get("args", {})

    config: dict = {
        "handler": trainer_cfg.get("handler"),
        "learning_rate": args.get("learning_rate"),
        "num_train_epochs": args.get("num_train_epochs"),
        "per_device_train_batch_size": args.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": args.get("gradient_accumulation_steps"),
        "lr_scheduler_type": args.get("lr_scheduler_type"),
        "warmup_ratio": args.get("warmup_ratio"),
    }

    method_args = trainer_cfg.get("method_args", {})
    regular_args = {}
    scorer_config = {}

    for k, v in method_args.items():
        if k == "scorer" and isinstance(v, dict):
            cfg = v.get("cfg", {})
            scorer_config["input_dimension"] = cfg.get("input_dimension")
        elif k == "scorer_trainer" and isinstance(v, dict):
            scorer_config["lambda_entropy"] = v.get("lambda_entropy")
            scorer_config["lambda_population"] = v.get("lambda_population")
            scorer_config["budget"] = v.get("budget")
            scorer_config["lambda_l2"] = v.get("lambda_l2")
            optim = v.get("optim_cfg", {})
            scorer_config["scorer_lr"] = optim.get("lr")
            scorer_config["update_every_n_steps"] = optim.get("update_every_n_steps")
            scorer_config["scheduler"] = optim.get("scheduler")
        elif k == "scorer_pretrain" and isinstance(v, dict):
            scorer_config["pretrain"] = v
        elif isinstance(v, (int, float, str, bool)):
            regular_args[k] = v

    config["method_args"] = regular_args
    if scorer_config:
        config["scorer_config"] = scorer_config

    return config


def drop_mia(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if not k.startswith("mia")}


def collect_one_run_short(run_dir: Path) -> dict:
    run_data: dict = {}

    para_eval_dir = run_dir / "paraphrase_evals"
    run_data["paraphrase_summary"] = drop_mia(
        load_json_safe(para_eval_dir / "Paraphrase_SUMMARY.json")
    )
    run_data["paraphrase_eval"] = drop_mia(
        load_json_safe(para_eval_dir / "Paraphrase_EVAL.json")
    )

    run_data["tofu_summary"] = None
    for ckpt_dir in sorted(run_dir.glob("checkpoint-*")):
        tofu_sum = load_json_safe(ckpt_dir / "evals" / "TOFU_SUMMARY.json")
        if tofu_sum is not None:
            run_data["tofu_summary"] = tofu_sum
            break
    # Fallback for runs that didn't save intermediate checkpoints (e.g. runs
    # with eval_strategy=no where the post-train TOFU eval is run separately
    # with paths.output_dir=<run>, landing TOFU_SUMMARY.json at the run root).
    if run_data["tofu_summary"] is None:
        run_data["tofu_summary"] = load_json_safe(run_dir / "TOFU_SUMMARY.json")

    trainer_state = load_json_safe(run_dir / "trainer_state.json")
    if trainer_state:
        run_data["training_stats"] = extract_training_stats(trainer_state)
        run_data["log_history"] = trainer_state.get("log_history", [])

    unlearn_hydra = run_dir / ".hydra.unlearn" / "config.yaml"
    hydra_config = load_yaml_safe(
        unlearn_hydra if unlearn_hydra.exists() else run_dir / ".hydra" / "config.yaml"
    )
    if hydra_config:
        run_data["training_config"] = strip_secrets(
            extract_training_config(hydra_config)
        )

    return run_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir",
        default="saves/unlearn/ATWU_TOFU",
        help="Root dir to walk (model/split/method).",
    )
    p.add_argument(
        "--output",
        default="results_archive/final_runs_short.json",
    )
    args = p.parse_args()

    runs_dir = ROOT_DIR / args.runs_dir
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")

    log.info(f"Collecting final runs from {runs_dir}")

    result: dict = {
        "metadata": {
            "description": "Final unlearning runs (short: no tofu_eval, no mia)",
            "source_dir": str(runs_dir.relative_to(ROOT_DIR)),
            "consolidated_at": datetime.now().isoformat(),
        },
        "runs": {},
    }

    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        result["runs"][model_name] = {}

        for split_dir in sorted(model_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            result["runs"][model_name][split_name] = {}

            for method_dir in sorted(split_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method_name = method_dir.name

                result["runs"][model_name][split_name][method_name] = (
                    collect_one_run_short(method_dir)
                )
                log.info(f"  {model_name}/{split_name}/{method_name}")

    out_path = ROOT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
