#!/usr/bin/env python
"""Figures for the ATWU rho/beta sensitivity sweep and the 5-fold scorer transfer.

Reads the per-run records written by scripts/sweep_atwu_sensitivity.py.

Experiment 1 — a 3x2 grid, rho down the left column, beta down the right:
    row 1  ES_forget (down = more forgetting) and ES_retain (up = more retention)
    row 2  MMLU and repetitiveness, with retain-model reference lines when a
           baseline is available
    row 3  realised token-selection %, against the rho target (dashed identity).
           This row is what says whether the population constraint still binds —
           where realised selection decouples from the target, metric movement is
           not attributable to rho.

Experiment 2 — the 5 folds against the in-domain co-trained reference, per metric,
with the frozen-uniform control drawn as a floor line when it was run.

Usage:
    python scripts/plot_atwu_sensitivity.py
    python scripts/plot_atwu_sensitivity.py --out-dir saves/sweep_atwu/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = Path(os.environ.get("ATWU_SWEEP_ROOT", "saves/sweep_atwu"))
if not SWEEP_ROOT.is_absolute():
    SWEEP_ROOT = REPO_ROOT / SWEEP_ROOT
RECORDS_DIR = SWEEP_ROOT / "records"
MODEL = "Llama-3.2-1B-Instruct"

METRICS = {
    "extraction_strength": ("ES$_{forget}$", "lower = more forgetting"),
    "retain_extraction_strength": ("ES$_{retain}$", "higher = more retention"),
    "mmlu": ("MMLU", "higher = capability kept"),
    "repetitiveness": ("Rep. (entropy)", "lower = more degenerate"),
}


def load_records() -> list[dict]:
    if not RECORDS_DIR.exists():
        raise SystemExit(f"no records at {RECORDS_DIR} — run the sweep first")
    return [json.loads(f.read_text()) for f in sorted(RECORDS_DIR.glob("*.json"))]


def retain_baseline() -> dict:
    """Retain-model (gold) reference lines.

    Preferred source is the sweep's own reference-model run
    (`--reference-model`), which measures all four metrics through the identical
    eval path as the sweep runs. Falls back to whatever pre-existing evals are on
    disk: the retain90 TOFU eval (ES only) and the community paraphrase baseline
    (repetitiveness only, no MMLU)."""
    ref = SWEEP_ROOT / f"reference_model/tofu_{MODEL}_retain90/record.json"
    if ref.exists():
        rec = json.loads(ref.read_text())
        if rec.get("metrics"):
            return rec["metrics"]
    out: dict = {}
    tofu = REPO_ROOT / f"saves/eval/tofu_{MODEL}_retain90/TOFU_SUMMARY.json"
    if tofu.exists():
        out.update({k: v for k, v in json.loads(tofu.read_text()).items()
                    if k in ("extraction_strength", "retain_extraction_strength")})
    para = REPO_ROOT / ("saves/eval/paraphrase_baselines/" + MODEL +
                        "/retain90_forget10/paraphrase_evals/Paraphrase_SUMMARY.json")
    if para.exists():
        out.update(json.loads(para.read_text()))
    return out


def series(records: list[dict], knob: str, other_fixed: float) -> list[tuple[float, dict]]:
    """(knob value, record) pairs for the 1-D sweep in `knob`. The reference run
    belongs to both grids, so it is included in each."""
    other = "beta" if knob == "rho" else "rho"
    out = []
    for r in records:
        if r["kind"] != "exp1" or r.get("status") != "ok":
            continue
        p = r["params"]
        if abs(p[other] - other_fixed) < 1e-9:
            out.append((p[knob], r))
    return sorted(out, key=lambda kv: kv[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(SWEEP_ROOT / "figures"))
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed in this environment.\n"
            "  .venv/bin/pip install matplotlib\n"
            "The numbers themselves are already in the records — "
            "`sweep_atwu_sensitivity.py --collect` prints them without matplotlib."
        )

    records = load_records()
    ok = [r for r in records if r.get("status") == "ok"]
    print(f"{len(ok)}/{len(records)} records usable")
    baseline = retain_baseline()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = next((r for r in ok if r["run_id"] == "reference"), None)
    ref_rho = ref["params"]["rho"] if ref else 0.2
    ref_beta = ref["params"]["beta"] if ref else 7.0

    # ══ Experiment 1 ══
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for col, (knob, fixed, label) in enumerate([
        ("rho", ref_beta, rf"$\rho$ (budget), $\beta$={ref_beta:g}"),
        ("beta", ref_rho, rf"$\beta$, $\rho$={ref_rho:g}"),
    ]):
        pts = series(ok, knob, fixed)
        if not pts:
            for row in range(3):
                axes[row][col].text(0.5, 0.5, f"no {knob} runs yet", ha="center")
            continue
        xs = [x for x, _ in pts]

        def vals(key):
            return [r["metrics"].get(key) for _, r in pts]

        # row 1 — the forget/retain trade-off
        ax = axes[0][col]
        ax.plot(xs, vals("extraction_strength"), "o-", color="#c0392b",
                label=METRICS["extraction_strength"][0])
        ax.set_ylabel(METRICS["extraction_strength"][0] + " (lower = forgotten)",
                      color="#c0392b")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        # Gold line: the retain model never learned the forget authors, so its
        # ES_forget is the floor a perfect unlearner would reach.
        if baseline.get("extraction_strength"):
            ax.axhline(baseline["extraction_strength"], ls=":", color="#c0392b", alpha=0.6)
        ax2 = ax.twinx()
        ax2.plot(xs, vals("retain_extraction_strength"), "s--", color="#2471a3",
                 label=METRICS["retain_extraction_strength"][0])
        ax2.set_ylabel(METRICS["retain_extraction_strength"][0] + " (higher = kept)",
                       color="#2471a3")
        ax2.tick_params(axis="y", labelcolor="#2471a3")
        if baseline.get("retain_extraction_strength"):
            ax2.axhline(baseline["retain_extraction_strength"], ls=":",
                        color="#2471a3", alpha=0.6)
        ax.set_title(f"Forget / retain vs {label}  (dotted = retain-model gold)")

        # row 2 — capability and degeneration
        ax = axes[1][col]
        ax.plot(xs, vals("mmlu"), "o-", color="#1e8449", label="MMLU")
        ax.set_ylabel("MMLU", color="#1e8449")
        ax.tick_params(axis="y", labelcolor="#1e8449")
        if baseline.get("mmlu"):
            ax.axhline(baseline["mmlu"], ls=":", color="#1e8449", alpha=0.6)
        ax2 = ax.twinx()
        ax2.plot(xs, vals("repetitiveness"), "s--", color="#8e44ad", label="Rep.")
        ax2.set_ylabel("Rep. (entropy; lower = degenerate)", color="#8e44ad")
        ax2.tick_params(axis="y", labelcolor="#8e44ad")
        if baseline.get("repetitiveness"):
            ax2.axhline(baseline["repetitiveness"], ls=":", color="#8e44ad", alpha=0.6)
        ax.set_title(f"Capability / degeneration vs {label}")

        # row 3 — does the knob do what it claims
        ax = axes[2][col]
        sel = [r.get("scorer_stats", {}).get("selected_pct_tail") for _, r in pts]
        ax.plot(xs, sel, "o-", color="#d68910", label="realised selected %")
        if knob == "rho":
            ax.plot(xs, [100 * x for x in xs], "k--", alpha=0.5, label=r"target $\rho$")
        ax.set_ylabel("forget tokens scored > 0.5 (%)")
        ax.set_title(f"Realised token selection vs {label}")
        ax.legend(fontsize=8)

        for row in range(3):
            axes[row][col].set_xlabel(label)
            axes[row][col].grid(alpha=0.25)
            axes[row][col].axvline(ref_rho if knob == "rho" else ref_beta,
                                   color="grey", ls="-", alpha=0.25)

    fig.suptitle(f"ATWU sensitivity — TOFU forget10, {MODEL}, seed 0 "
                 "(grey line = paper operating point)", y=0.995)
    fig.tight_layout()
    p = out_dir / "exp1_sensitivity.png"
    fig.savefig(p, dpi=150)
    print(f"-> {p}")

    # ══ Experiment 2 ══
    folds = sorted([r for r in ok if r["kind"] == "exp2" and r["params"].get("fold")],
                   key=lambda r: r["params"]["fold"])
    if not folds:
        print("no fold runs yet — skipping experiment 2 figure")
        return

    uniform = next((r for r in ok if r["run_id"] == "uniform"), None)
    # In-domain co-trained references: the sweep reference, and the paper repro run.
    repro_summary = REPO_ROOT / f"saves/unlearn/TOFU_repro/{MODEL}/forget10/ATWU/TOFU_SUMMARY.json"
    repro_para = REPO_ROOT / f"saves/unlearn/TOFU_repro/{MODEL}/forget10/ATWU/paraphrase_evals/Paraphrase_SUMMARY.json"
    in_domain: dict[str, float] = {}
    if ref:
        in_domain = {k: v for k, v in ref["metrics"].items() if v is not None}
    elif repro_summary.exists():
        in_domain = json.loads(repro_summary.read_text())
        if repro_para.exists():
            in_domain.update(json.loads(repro_para.read_text()))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    xs = [r["params"]["fold"] for r in folds]
    for ax, (key, (name, direction)) in zip(axes, METRICS.items()):
        ys = [r["metrics"].get(key) for r in folds]
        ax.bar([str(x) for x in xs], ys, color="#5499c7", label="fold scorer (frozen)")
        if key in in_domain:
            ax.axhline(in_domain[key], color="#c0392b", ls="-",
                       label="in-domain co-trained")
        if uniform and uniform["metrics"].get(key) is not None:
            ax.axhline(uniform["metrics"][key], color="grey", ls="--",
                       label="frozen uniform (0.5)")
        ax.set_title(f"{name}\n({direction})", fontsize=10)
        ax.set_xlabel("fold used to fit the scorer")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("5-fold scorer transfer — scorer fit on 20% of forget10 (model frozen), "
                 "then frozen while unlearning all of forget10", y=1.02)
    fig.tight_layout()
    p = out_dir / "exp2_kfold_transfer.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"-> {p}")

    # Spread table: the number that answers "does it need relearning".
    print(f"\n{'metric':<32}{'folds mean':>12}{'min':>10}{'max':>10}{'in-domain':>12}")
    print("-" * 76)
    for key, (name, _) in METRICS.items():
        ys = [r["metrics"].get(key) for r in folds if r["metrics"].get(key) is not None]
        if not ys:
            continue
        ref_v = in_domain.get(key)
        ref_s = f"{ref_v:>12.4f}" if ref_v is not None else f"{'-':>12}"
        print(f"{key:<32}{sum(ys) / len(ys):>12.4f}{min(ys):>10.4f}{max(ys):>10.4f}{ref_s}")


if __name__ == "__main__":
    main()
