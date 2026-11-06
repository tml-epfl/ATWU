#!/usr/bin/env python
"""Retrain and evaluate the selected WMDP configurations on seeds 1, 2, and 3.

Each method's selected trial is read from ``--params``: ``results/wmdp.json``
(the default) for the published configurations, or
``$WMDP_STATE/selected.json`` written by the search scripts after a new
tuning run. Seed 0 is the search seed and is not used here.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402


METHODS = ("atwu", "graddiff", "npo", "simnpo", "jensun", "rmu", "wga", "satimp")
MODEL = "zephyr-7b-beta"
PRETRAINED = "HuggingFaceH4/zephyr-7b-beta"
TRAIN_TIMEOUT = 3 * 3600
EVAL_TIMEOUT = 3 * 3600
NUM_SAMPLES = 1000
MAX_NEW_TOKENS = 128
LM_METRICS = ("mmlu", "wmdp_bio", "wmdp_cyber", "wmdp_mean")


def _load_selected(path: Path, methods):
    source = json.loads(path.read_text())
    selected = {}
    for method in methods:
        item = source["methods"].get(method)
        if item is None:
            raise SystemExit(f"{path} has no selected configuration for {method}")
        params = item.get("selected_params")
        if params is None:  # layout written by export_results.py
            params = item["selected"]["search_params"]
        selected[method] = {"trial": item.get("selected_trial"), "params": params}
    return selected, source.get("base_model", {})


def _recorded_params(output_path: Path):
    """Return the hyperparameters already recorded per method in results.json."""
    if not output_path.exists():
        return {}
    output = json.loads(output_path.read_text())
    return {
        method: record.get("params")
        for method, record in output.get("methods", {}).items()
    }


def _check_params(out_dir: Path, params, recorded):
    """Refuse to reuse a run directory trained with other hyperparameters.

    New runs store ``params.json``. Older run directories without it are
    accepted only when results.json records the same hyperparameters.
    """
    path = out_dir / "params.json"
    if path.exists():
        if json.loads(path.read_text()) != params:
            raise RuntimeError(
                f"{out_dir} was trained with different hyperparameters; "
                "use a new --state directory"
            )
        return
    has_outputs = any(out_dir.glob("*.safetensors")) or any(
        (out_dir / name).exists()
        for name in ("LMEval_SUMMARY.json", "Paraphrase_SUMMARY.json")
    )
    if has_outputs and recorded != params:
        raise RuntimeError(
            f"{out_dir} has outputs but no matching params.json; cannot confirm "
            "which hyperparameters produced them. Use a new --state directory."
        )
    path.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")


def _train_args(method, params, out_dir, seed, corpus_dir):
    if method == "atwu":
        hp = dict(params, alternate_every=1, retain_replacement=False)
        return common.atwu_train_args(
            out_dir, hp=hp, seed=seed, corpus_dir=corpus_dir
        )
    return common.baseline_train_args(method, params, out_dir, seed, corpus_dir)


def _paraphrase_args(model_path, out_dir, baseline_path=None, baseline_only=False):
    command = [
        common.PY,
        "src/eval.py",
        "experiment=eval/paraphrase/default.yaml",
        "eval=[paraphrase]",
        f"model={MODEL}",
        f"task_name=wmdp_selected_{Path(out_dir).name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        "++model.model_args.device_map=auto",
        f"paths.output_dir={out_dir}",
        "~eval.paraphrase.metrics.mmlu",
        f"++eval.paraphrase.metrics.repetitiveness.num_samples={NUM_SAMPLES}",
        "++eval.paraphrase.metrics.repetitiveness.generation="
        f"{{max_new_tokens:{MAX_NEW_TOKENS},do_sample:false,temperature:0.0}}",
    ]
    if baseline_only:
        command.append("~eval.paraphrase.metrics.winrate")
    else:
        command.append(f"+eval.paraphrase.metrics.winrate.baseline_path={baseline_path}")
    return command


def _metrics(out_dir):
    record = {}
    lm_path = out_dir / "LMEval_SUMMARY.json"
    if lm_path.exists():
        lm = json.loads(lm_path.read_text())
        bio = lm.get("wmdp_bio/acc")
        cyber = lm.get("wmdp_cyber/acc")
        record.update(
            {
                "mmlu": lm.get("mmlu/acc"),
                "wmdp_bio": bio,
                "wmdp_cyber": cyber,
                "wmdp_mean": (bio + cyber) / 2
                if bio is not None and cyber is not None
                else None,
            }
        )
    para_path = out_dir / "Paraphrase_SUMMARY.json"
    if para_path.exists():
        para = json.loads(para_path.read_text())
        record["repetitiveness"] = para.get("repetitiveness")
        record["winrate"] = para.get("winrate")
    return record


def _lm_complete(metrics):
    """Return true only when the complete WMDP/MMLU suite is present."""
    return all(metrics.get(metric) is not None for metric in LM_METRICS)


def _clear_paraphrase_outputs(out_dir):
    """Discard prior utility outputs when a checkpoint must be reconstructed."""
    for name in ("Paraphrase_EVAL.json", "Paraphrase_SUMMARY.json"):
        (out_dir / name).unlink(missing_ok=True)
    for name in ("repetitiveness", "winrate"):
        shutil.rmtree(out_dir / name, ignore_errors=True)


def _ensure_baseline(root, gpu, dry_run=False):
    out_dir = root / "base"
    generations = out_dir / "repetitiveness" / "model.jsonl"
    if generations.exists():
        return generations
    command = _paraphrase_args(PRETRAINED, out_dir, baseline_only=True)
    if dry_run:
        print(" ".join(command))
        return generations
    out_dir.mkdir(parents=True, exist_ok=True)
    with (root / ".baseline.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if generations.exists():
            return generations
        gpu_index = int(str(gpu).split(",")[0]) if gpu is not None else 0
        if not common.wait_for_gpu(gpu_index=gpu_index):
            raise RuntimeError("GPU never became free for base-model generation")
        env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
        rc = common.run_cmd(
            command, out_dir / "paraphrase.log", timeout=EVAL_TIMEOUT, env=env
        )
        if rc != 0 or not generations.exists():
            raise RuntimeError(
                f"base-model generation failed; see {out_dir / 'paraphrase.log'}"
            )
    return generations


def _store_result(
    output_path,
    params_file,
    corpus_dir,
    seeds,
    base_model,
    method,
    item,
    seed,
    metrics,
):
    """Merge one completed seed into the shared result file atomically."""
    with (output_path.parent / ".results.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output_path.exists():
            output = json.loads(output_path.read_text())
        else:
            output = {
                "params_file": str(params_file.resolve()),
                "corpus_dir": str(corpus_dir.resolve()),
                "seeds": seeds,
                "base_model": base_model,
                "methods": {},
            }
        method_record = output["methods"].setdefault(
            method,
            {"source_trial": item["trial"], "params": item["params"], "seeds": {}},
        )
        if method_record["params"] != item["params"]:
            raise RuntimeError(
                f"{output_path} already holds {method} results for different "
                "hyperparameters; use a new --state directory"
            )
        method_record["seeds"][str(seed)] = metrics
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(output, indent=2) + "\n")
        os.replace(temporary, output_path)


def _run_one(
    method,
    params,
    seed,
    root,
    corpus_dir,
    gpu,
    baseline_path,
    include_paraphrase,
    dry_run,
    recorded=None,
):
    out_dir = root / method / f"seed_{seed}"
    train_command = _train_args(method, params, out_dir, seed, corpus_dir)
    if dry_run:
        print(" ".join(train_command))
        print(" ".join(common.eval_args(out_dir)))
        if include_paraphrase:
            print(" ".join(_paraphrase_args(out_dir, out_dir, baseline_path)))
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    _check_params(out_dir, params, recorded)
    env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else None
    gpu_index = int(str(gpu).split(",")[0]) if gpu is not None else 0
    current = _metrics(out_dir)
    lm_done = _lm_complete(current)
    para_done = (
        current.get("repetitiveness") is not None
        and current.get("winrate") is not None
    )
    need_checkpoint = not lm_done or (include_paraphrase and not para_done)
    retrained = False

    if need_checkpoint and not any(out_dir.glob("*.safetensors")):
        if not common.wait_for_gpu(gpu_index=gpu_index):
            raise RuntimeError(f"GPU {gpu_index} never became free")
        (out_dir / "train_cmd.txt").write_text(" ".join(train_command) + "\n")
        rc = common.run_cmd(
            train_command, out_dir / "train.log", timeout=TRAIN_TIMEOUT, env=env
        )
        if rc != 0 or not any(out_dir.glob("*.safetensors")):
            raise RuntimeError(f"{method} seed {seed}: training failed")
        retrained = True
        if include_paraphrase:
            _clear_paraphrase_outputs(out_dir)
            para_done = False

    if not lm_done:
        eval_env = dict(env or {})
        # Parallel workers that pre-download WMDP and MMLU set this so their
        # evaluations read only the shared cache and cannot race on the Hub.
        if os.environ.get("WMDP_OFFLINE_LM_EVAL") == "1":
            eval_env["HF_DATASETS_OFFLINE"] = "1"
        rc = common.run_cmd(
            common.eval_args(out_dir),
            out_dir / "eval.log",
            timeout=EVAL_TIMEOUT,
            env=eval_env,
        )
        current = _metrics(out_dir)
        if rc != 0 or not _lm_complete(current):
            raise RuntimeError(f"{method} seed {seed}: WMDP/MMLU evaluation failed")

    if include_paraphrase and not para_done:
        rc = common.run_cmd(
            _paraphrase_args(out_dir, out_dir, baseline_path),
            out_dir / "paraphrase.log",
            timeout=EVAL_TIMEOUT,
            env=env,
        )
        current = _metrics(out_dir)
        if rc != 0 or current.get("repetitiveness") is None or current.get("winrate") is None:
            raise RuntimeError(f"{method} seed {seed}: utility evaluation failed")

    common.cleanup_weights(out_dir)
    return _metrics(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--corpus-dir", type=Path, default=common.DEFAULT_CORPUS)
    parser.add_argument(
        "--params",
        type=Path,
        default=common.PROJ / "results" / "wmdp.json",
        help="results/wmdp.json, or $WMDP_STATE/selected.json after a new search",
    )
    parser.add_argument("--state", type=Path, default=common.STATE)
    parser.add_argument("--no-paraphrase", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Generate the shared base-model generations and exit",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = args.state.resolve()
    common.ensure_corpus(args.corpus_dir, quiet=True)
    selected, base_model = _load_selected(args.params, args.method)
    root = state / "final"
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    include_paraphrase = not args.no_paraphrase
    baseline_path = None
    if include_paraphrase:
        baseline_path = _ensure_baseline(root, args.gpus, args.dry_run)
    if args.baseline_only:
        if not include_paraphrase:
            parser.error("--baseline-only cannot be combined with --no-paraphrase")
        if not args.dry_run:
            print(f"baseline ready: {baseline_path}")
        return

    output_path = root / "results.json"
    recorded = _recorded_params(output_path)
    for method in args.method:
        item = selected[method]
        for seed in args.seeds:
            common.log(f"[selected] {method} seed {seed}")
            metrics = _run_one(
                method,
                item["params"],
                seed,
                root,
                args.corpus_dir,
                args.gpus,
                baseline_path,
                include_paraphrase,
                args.dry_run,
                recorded.get(method),
            )
            if not args.dry_run:
                _store_result(
                    output_path,
                    args.params,
                    args.corpus_dir,
                    args.seeds,
                    base_model,
                    method,
                    item,
                    seed,
                    metrics,
                )
    if not args.dry_run:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
