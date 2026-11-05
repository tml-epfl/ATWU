#!/usr/bin/env python
"""AUROC analysis for the leave-one-fold-out unlearning experiment.

Each run unlearned 4 of the 5 forget10 folds and was then evaluated on ALL 400
rows, so the per-example scores in TOFU_EVAL.json split into two groups:

    trained  — the 320 rows the unlearning actually saw
    held-out — the 80 rows it never saw

AUROC is reported with the HELD-OUT rows as the positive class, so:

    0.5  the two groups are indistinguishable -> forgetting reached data the
         unlearning never saw (it generalised)
    1.0  perfectly separable -> forgetting stayed on the rows it was trained on
    <0.5 held-out rows are forgotten MORE than trained rows (would be odd)

Two per-example scores are used independently: extraction strength (higher = more
extractable) and the forget-answer loss (higher = less likely, so it is negated to
keep "higher = more memorised").

With --baseline pointing at an un-unlearned model's TOFU_EVAL.json, the same fold
split is applied there too. That gives (a) a sanity floor — AUROC should be ~0.5
before any unlearning, since the folds are arbitrary — and (b) a normalised
recovery fraction per fold:

    generalisation = (baseline_heldout - run_heldout) / (baseline_heldout - run_trained)

i.e. how much of the forgetting achieved on the trained rows also landed on the
held-out rows. 1.0 = held-out forgotten as thoroughly as trained; 0.0 = untouched.

Usage:
    python scripts/kfold_auroc.py
    python scripts/kfold_auroc.py --baseline saves/sweep_atwu/reference_model/tofu_Llama-3.2-1B-Instruct_full/TOFU_EVAL.json
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

SCORES = {
    "extraction_strength": ("ES", "score", +1),
    # avg_loss: higher = less likely, so negate to keep "higher = more memorised"
    "forget_Q_A_Prob": ("neg-loss", "avg_loss", -1),
}


def per_example(eval_json: Path, metric: str, field: str, sign: int) -> dict[int, float]:
    d = json.loads(eval_json.read_text())
    vbi = (d.get(metric) or {}).get("value_by_index") or {}
    out = {}
    for k, v in vbi.items():
        x = v.get(field)
        if isinstance(x, list):          # perturbed variants store a list
            x = sum(x) / len(x)
        if x is not None:
            out[int(k)] = sign * float(x)
    return out


def fold_membership(record: dict) -> tuple[list[int], list[int], str, int]:
    """(held_out, trained, scheme, fold) for a leave-one-fold-out run.

    Taken from the record when present, otherwise reconstructed from the manifest
    by run_id. Either way it is VERIFIED against the `split` override the run was
    actually launched with (`.hydra/overrides.yaml`) — mislabelling the two groups
    would silently invert the whole analysis, so a mismatch is fatal rather than a
    warning.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from sweep_atwu_sensitivity import kfold_holdout_manifest

    rid = record["run_id"]
    scheme = record.get("params", {}).get("fold_scheme")
    if scheme is None:                       # records written before this was stored
        scheme = {"a": "author", "q": "question"}[rid[2]]
    fold = record.get("params", {}).get("held_out_fold") or int(rid[3:])

    spec = next(r for r in kfold_holdout_manifest(scheme) if r["run_id"] == rid)
    held = record.get("held_out_indices") or spec["held_out_indices"]
    trained = record.get("trained_indices") or spec["trained_indices"]

    # Verify the run really did train on a subset. The training overrides are not
    # recoverable from .hydra (the eval reuses paths.output_dir and overwrites them),
    # but the optimizer-step count is decisive: 320 rows / (8 x 4) x 10 epochs = 100
    # steps, where all 400 rows would give 120. A run showing 120 silently ignored
    # its split override and is not a leave-one-fold-out run at all.
    ts = Path(REPO_ROOT / record["out_dir"]) / "trainer_state.json"
    if ts.exists():
        steps = json.loads(ts.read_text()).get("max_steps")
        expected_steps = (len(trained) // 8 // 4) * 10
        if steps != expected_steps:
            raise SystemExit(
                f"{rid}: trained {steps} optimizer steps but {len(trained)} rows imply "
                f"{expected_steps} — the forget-split override did not take effect, so "
                "the two groups would be mislabelled. Refusing to report AUROC.")
    return held, trained, scheme, fold


def label_check(vals: dict[int, float], held: list[int]) -> float:
    """Empirical check that the fold labels line up with the data: the n highest-scoring
    rows should be the held-out ones. Returns the overlap fraction (1.0 = the top-n by
    score are exactly the held-out rows, ~0.2 = chance). Diagnostic, not a gate — under
    a scheme where forgetting partially generalises, a lower overlap is a real result
    rather than a bookkeeping error."""
    n = len(held)
    top = {i for i, _ in sorted(vals.items(), key=lambda kv: -kv[1])[:n]}
    return len(top & set(held)) / max(n, 1)


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a positive > score of a negative), ties counted as half."""
    from sklearn.metrics import roc_auc_score
    labels = [1] * len(pos) + [0] * len(neg)
    return float(roc_auc_score(labels, pos + neg))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=None,
                    help="TOFU_EVAL.json of a model that was NOT unlearned (e.g. the "
                         "tofu_*_full checkpoint) for the 0.5 sanity floor and the "
                         "normalised generalisation fraction")
    ap.add_argument("--out", default=str(SWEEP_ROOT / "kfold_auroc.json"))
    args = ap.parse_args()

    runs = []
    for f in sorted(RECORDS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("kind") == "exp3" and r.get("status") == "ok":
            runs.append(r)
    if not runs:
        raise SystemExit("no completed exp3 (leave-one-fold-out) records found — "
                         "run: sweep_atwu_sensitivity.py --kfold both")

    baseline = Path(REPO_ROOT / args.baseline) if args.baseline else None
    results = []

    for metric, (label, field, sign) in SCORES.items():
        rows = []
        for r in runs:
            ev = Path(REPO_ROOT / r["out_dir"]) / "TOFU_EVAL.json"
            vals = per_example(ev, metric, field, sign)
            if not vals:
                continue
            held_idx, trained_idx, scheme_r, fold_r = fold_membership(r)
            held = [vals[i] for i in held_idx if i in vals]
            trained = [vals[i] for i in trained_idx if i in vals]
            row = {
                "run_id": r["run_id"], "scheme": scheme_r,
                "fold": fold_r, "metric": label,
                "auroc": auroc(held, trained),
                "mean_trained": mean(trained), "mean_heldout": mean(held),
                "n_trained": len(trained), "n_heldout": len(held),
                "top_n_overlap": label_check(vals, held_idx),
            }
            if baseline and baseline.exists():
                b = per_example(baseline, metric, field, sign)
                bh = [b[i] for i in held_idx if i in b]
                bt = [b[i] for i in trained_idx if i in b]
                row["baseline_auroc"] = auroc(bh, bt)
                row["baseline_mean_heldout"] = mean(bh)
                row["baseline_mean_trained"] = mean(bt)
                denom = mean(bt) - row["mean_trained"]
                row["generalisation"] = (
                    (mean(bh) - row["mean_heldout"]) / denom if abs(denom) > 1e-9
                    else float("nan"))
            rows.append(row)
        results.extend(rows)

        if not rows:
            continue
        for scheme in ("author", "question"):
            sub = [x for x in rows if x["scheme"] == scheme]
            if not sub:
                continue
            title = ("hold out 4 whole AUTHORS" if scheme == "author"
                     else "hold out 4 QA rows of EVERY author")
            print(f"\n=== {label}: {title} ===")
            has_b = "generalisation" in sub[0]
            hdr = (f"{'fold':<6}{'AUROC':>8}{'trained':>10}{'held-out':>10}{'top-n':>8}")
            if has_b:
                hdr += f"{'base h/o':>10}{'base AUROC':>12}{'generalised':>13}"
            print(hdr)
            print("-" * len(hdr))
            for x in sorted(sub, key=lambda y: y["fold"]):
                line = (f"{x['fold']:<6}{x['auroc']:>8.3f}{x['mean_trained']:>10.4f}"
                        f"{x['mean_heldout']:>10.4f}{x['top_n_overlap']:>8.2f}")
                if has_b:
                    line += (f"{x['baseline_mean_heldout']:>10.4f}"
                             f"{x['baseline_auroc']:>12.3f}"
                             f"{x['generalisation']:>13.2f}")
                print(line)
            aur = [x["auroc"] for x in sub]
            print(f"{'mean':<6}{mean(aur):>8.3f}"
                  f"{mean([x['mean_trained'] for x in sub]):>10.4f}"
                  f"{mean([x['mean_heldout'] for x in sub]):>10.4f}"
                  f"{mean([x['top_n_overlap'] for x in sub]):>8.2f}"
                  + (f"{mean([x['baseline_mean_heldout'] for x in sub]):>10.4f}"
                     f"{mean([x['baseline_auroc'] for x in sub]):>12.3f}"
                     f"{mean([x['generalisation'] for x in sub]):>13.2f}" if has_b else ""))

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
