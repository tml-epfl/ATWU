#!/usr/bin/env python
"""Export compact TOFU sensitivity and scorer-transfer result summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text())


def _compact_run(run: dict) -> dict:
    scorer = run.get("scorer_stats") or run.get("fold_scorer_stats") or {}
    return {
        "run_id": run["run_id"],
        "status": run["status"],
        "params": run.get("params", {}),
        "seed": run.get("seed"),
        "metrics": run.get("metrics", {}),
        "scorer": {
            key: scorer[key]
            for key in (
                "selected_pct_tail",
                "score_mean_tail",
                "score_std_tail",
                "budget_penalty_tail",
            )
            if key in scorer
        },
    }


def _aggregate_transfer(results: dict, prefix: str) -> dict:
    folds = [value for key, value in sorted(results.items()) if key.startswith(prefix)]
    seen = [row["auroc_seen_fold"] for row in folds]
    unseen = [row["auroc_unseen_folds"] for row in folds]
    gaps = [a - b for a, b in zip(seen, unseen)]
    return {
        "seen_mean": statistics.mean(seen),
        "seen_std": statistics.stdev(seen),
        "unseen_mean": statistics.mean(unseen),
        "unseen_std": statistics.stdev(unseen),
        "gap_mean": statistics.mean(gaps),
        "gap_std": statistics.stdev(gaps),
    }


def _reference_model(root: Path, name: str) -> dict:
    record = _load(root / "reference_model" / name / "record.json")
    return {"model": record["model_path"], "status": record["status"], "metrics": record["metrics"]}


def export(root: Path, output_json: Path, output_md: Path) -> None:
    runs = _load(root / "summary.json")
    by_id = {run["run_id"]: run for run in runs}
    reference = by_id["reference"]

    rho_runs = [reference] + [run for run in runs if run["run_id"].startswith("rho")]
    rho_runs.sort(key=lambda run: float(run["params"]["rho"]))
    beta_runs = [reference] + [run for run in runs if run["run_id"].startswith("beta")]
    beta_runs.sort(key=lambda run: float(run["params"]["beta"]))

    scorer_transfer = _load(root / "scorer_auroc_kfold.json")
    transfer_results = scorer_transfer["results"]
    reference_scorer = _load(root / "scorer_auroc.json")
    end_to_end = _load(root / "kfold_auroc.json")

    result = {
        "schema_version": 1,
        "benchmark": "TOFU forget10",
        "model": reference["model"],
        "seed": reference["seed"],
        "selected_hyperparameters": reference["params"],
        "ground_truth_forget_token_ratio": 0.22,
        "reference_models": {
            "full": _reference_model(root, "tofu_Llama-3.2-1B-Instruct_full"),
            "retain90": _reference_model(root, "tofu_Llama-3.2-1B-Instruct_retain90"),
        },
        "sensitivity": {
            "rho": [_compact_run(run) for run in rho_runs],
            "beta": [_compact_run(run) for run in beta_runs],
        },
        "frozen_scorer_folds": [
            _compact_run(run) for run in runs if run["kind"] == "exp2"
        ],
        "scorer_transfer": {
            "schemes": {
                "author": "hold out unseen authors",
                "fact": "hold out unseen facts/questions for seen authors",
            },
            "per_fold": transfer_results,
            "aggregate": {
                "author": _aggregate_transfer(transfer_results, "author_holdout"),
                "fact": _aggregate_transfer(transfer_results, "question_holdout"),
            },
            "reference_scorer": reference_scorer["results"]["reference"],
            "end_to_end_fold_metrics": end_to_end,
        },
        "holdout_runs": [
            _compact_run(run) for run in runs if run["kind"] == "exp3"
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# TOFU sensitivity and scorer-transfer ablations",
        "",
        "All runs use TOFU forget10 and Llama-3.2-1B-Instruct. The selected operating",
        "point is ρ = 0.2 and β = 7; the selected ρ closely matches the labelled",
        "forget-token ratio of 0.22.",
        "",
        "## ρ sensitivity (β = 7)",
        "",
        "| ρ | Forget ES ↓ | Retain ES ↑ | MMLU ↑ | Rep. ↑ | Selected % |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in rho_runs:
        metrics = run["metrics"]
        selected = (run.get("scorer_stats") or {}).get("selected_pct_tail")
        lines.append(
            f"| {run['params']['rho']:.1f} | {metrics['extraction_strength']:.4f} | "
            f"{metrics['retain_extraction_strength']:.4f} | {100 * metrics['mmlu']:.1f} | "
            f"{metrics['repetitiveness']:.0f} | {selected:.1f} |"
        )
    lines.extend([
        "",
        "## β sensitivity (ρ = 0.2)",
        "",
        "| β | Forget ES ↓ | Retain ES ↑ | MMLU ↑ | Rep. ↑ | Selected % |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for run in beta_runs:
        metrics = run["metrics"]
        selected = (run.get("scorer_stats") or {}).get("selected_pct_tail")
        lines.append(
            f"| {run['params']['beta']:.0f} | {metrics['extraction_strength']:.4f} | "
            f"{metrics['retain_extraction_strength']:.4f} | {100 * metrics['mmlu']:.1f} | "
            f"{metrics['repetitiveness']:.0f} | {selected:.1f} |"
        )
    lines.extend([
        "",
        "Utility and forgetting are stable around the selected operating point. For β,",
        "there is a broad plateau from approximately 6–10; disabling saturation at β = 0",
        "destroys retention. For ρ, approximately 0.2–0.5 gives similar forgetting, while",
        "both very small and all-token budgets degrade the result.",
        "",
        "## Five-fold scorer transfer",
        "",
        "| Run | Scheme | Seen AUROC ↑ | Unseen AUROC ↑ | Gap |",
        "|---|---|---:|---:|---:|",
    ])
    for key, row in transfer_results.items():
        scheme = "author" if key.startswith("author") else "fact"
        fold = key.rsplit("holdout", 1)[1]
        gap = row["auroc_seen_fold"] - row["auroc_unseen_folds"]
        lines.append(
            f"| {scheme[0].upper()}{fold} | {scheme} | {row['auroc_seen_fold']:.4f} | "
            f"{row['auroc_unseen_folds']:.4f} | {gap:+.4f} |"
        )
    for scheme in ("author", "fact"):
        agg = result["scorer_transfer"]["aggregate"][scheme]
        lines.append(
            f"| **{scheme} mean** | **{scheme}** | **{agg['seen_mean']:.4f}** | "
            f"**{agg['unseen_mean']:.4f}** | **{agg['gap_mean']:+.4f}** |"
        )
    lines.extend([
        "",
        "The small seen–unseen gaps show that the learned scorer transfers both to unseen",
        "authors and to unseen facts about authors observed during training. The JSON",
        "companion contains all sweep points, fold-level results, aggregate statistics,",
        "and end-to-end fold metrics.",
        "",
    ])
    output_md.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("saves/sweep_atwu"))
    parser.add_argument("--output-json", type=Path, default=Path("results/tofu_ablations.json"))
    parser.add_argument("--output-md", type=Path, default=Path("results/tofu_ablations.md"))
    args = parser.parse_args()
    export(args.root, args.output_json, args.output_md)


if __name__ == "__main__":
    main()
