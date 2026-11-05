#!/usr/bin/env python
"""Token-level AUROC of ATWU scorers against the GPT importance labels.

This is the metric used for method comparison in the importance-scoring repo
(notebooks/1.17-method-comparison-extended-tofu-forget10.ipynb, src/eval/auroc.py):
for each sample, rank the answer tokens by the scorer's weight and measure how well
that ranking recovers the tokens GPT labelled as important, then average the
per-sample AUROCs. Samples whose ground truth is all-0 or all-1 are skipped, as
there AUROC is undefined.

Ported here — not imported — because atwu's QADataset has no `importance_set`
argument; the span -> token mapping and the gt/score alignment are reproduced
exactly (token t of the score vector corresponds to input token t+1, masked to
positions where labels[1:] != -100).

What it is for: the k-fold scorer question. Each fold scorer was fit on 80 rows of
forget10, so its AUROC on the OTHER 320 rows says whether the token importance it
learned transfers to forget data it never saw. Reported per scorer, split into the
fold it was trained on ("seen") and the rest ("unseen").

Usage:
    python scripts/scorer_auroc.py                      # auto-discover sweep scorers
    python scripts/scorer_auroc.py --scorers a=path/a.pt b=path/b.pt
    python scripts/scorer_auroc.py --show-examples 3    # print highlighted tokens
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
SWEEP_ROOT = Path(os.environ.get("ATWU_SWEEP_ROOT", "saves/sweep_atwu"))
if not SWEEP_ROOT.is_absolute():
    SWEEP_ROOT = REPO_ROOT / SWEEP_ROOT

MODEL = "Llama-3.2-1B-Instruct"
FORGET_SPLIT = "forget10"
RETAIN_SPLIT = "retain90"
INPUT_DIM = 2048
IGNORE_INDEX = -100
N_FOLDS = 5

# GPT-labelled important tokens for TOFU forget10 (char spans over the answer).
GPT_LABELS = (
    REPO_ROOT
    / "data/gpt-selected-tokens-tofu/forget10_with_common_words_gpt"
)


def load_importance_set(dataset_path: Path) -> dict[int, list[tuple[int, int]]]:
    """{row index: [(start_char, end_char), ...]} — as in importance-scoring's
    src/utils/data.py."""
    from datasets import load_from_disk
    gpt = load_from_disk(str(dataset_path))
    return {
        idx: [(p["start"], p["end"]) for p in json.loads(tw)]
        for idx, tw in enumerate(gpt["target_words"])
    }


def important_token_indices(tokenizer, answer: str, answer_start_idx: int,
                            spans: list[tuple[int, int]]) -> set[int]:
    """Absolute input_ids positions whose characters overlap a labelled span."""
    if not spans:
        return set()
    offsets = tokenizer(answer, add_special_tokens=False,
                        return_offsets_mapping=True)["offset_mapping"]
    out = set()
    for local_idx, (cs, ce) in enumerate(offsets):
        if cs == ce:
            continue
        for rs, re_ in spans:
            if cs < re_ and (ce - 1) >= rs:
                out.add(answer_start_idx + local_idx)
                break
    return out


def discover_scorers() -> dict[str, str]:
    found: dict[str, str] = {}
    for p in sorted((SWEEP_ROOT / "exp2").glob("fold*/scorer/scorer.pt")):
        found[p.parent.parent.name] = str(p)
    ref = SWEEP_ROOT / "exp1/reference/scorer.pt"
    if ref.exists():
        found["reference"] = str(ref)
    return found


def kfold_scorers() -> dict[str, str]:
    """Scorers co-trained during the leave-one-fold-out unlearning runs. These are the
    only properly-trained scorers with rows they never saw, so they give the
    out-of-sample AUROC — the reference scorer was co-trained on all 400."""
    found = {}
    for p in sorted((SWEEP_ROOT / "exp3").glob("*/scorer.pt")):
        found[p.parent.name] = str(p)
    return found


def heldout_indices_for(label: str) -> set[int] | None:
    """Rows the scorer named `label` never saw, from the run's record."""
    rec_by_dir = {}
    for f in (SWEEP_ROOT / "records").glob("*.json"):
        r = json.loads(f.read_text())
        if r.get("kind") == "exp3":
            rec_by_dir[Path(r["out_dir"]).name] = r
    r = rec_by_dir.get(label)
    if r is None:
        return None
    from kfold_auroc import fold_membership
    held, _, _, _ = fold_membership(r)
    return set(held)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="*", default=None, help="label=path pairs")
    ap.add_argument("--kfold", action="store_true",
                    help="Use the scorers co-trained by the leave-one-fold-out runs, and "
                         "split seen/unseen by each run's actual held-out rows")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--labels", default=str(GPT_LABELS))
    ap.add_argument("--show-examples", type=int, default=0,
                    help="Print this many samples with the labelled tokens marked, as a "
                         "check that the span->token mapping lines up")
    ap.add_argument("--out", default=str(SWEEP_ROOT / "scorer_auroc.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    import reproduce_tofu as rt
    from data import get_collators, get_data
    from model import get_model
    from scorers import LearnedScorer, ScorerConfig

    rt.load_dotenv(REPO_ROOT / ".env")

    scorers_paths = {}
    if args.scorers:
        for spec in args.scorers:
            label, path = spec.split("=", 1)
            scorers_paths[label] = path
    elif args.kfold:
        scorers_paths = kfold_scorers()
    else:
        scorers_paths = discover_scorers()
    if not scorers_paths:
        sys.exit("no scorers found — pass --scorers label=path")
    missing = [p for p in scorers_paths.values() if not Path(p).exists()]
    if missing:
        sys.exit(f"missing scorer checkpoints: {missing}")
    print(f"scorers: {json.dumps(scorers_paths, indent=2)}")

    importance_set = load_importance_set(Path(args.labels))
    print(f"GPT labels: {len(importance_set)} rows, "
          f"{sum(1 for v in importance_set.values() if v)} with >=1 span")

    overrides = [
        "experiment=unlearn/tofu/default", "trainer=ATWU", f"model={MODEL}",
        "task_name=scorer_auroc",
        f"model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{MODEL}_full",
        "++model.model_args.device_map=auto",
        f"forget_split={FORGET_SPLIT}", f"retain_split={RETAIN_SPLIT}",
        *rt.hf_token_overrides(),
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="unlearn.yaml", overrides=overrides)

    model, tokenizer = get_model(cfg.model)
    model.eval()
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer,
                    template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator)

    from datasets import load_dataset
    raw = load_dataset("locuslab/TOFU", FORGET_SPLIT, split="train")
    answers = raw["answer"]

    device = next(model.parameters()).device
    scfg = ScorerConfig(input_dimension=INPUT_DIM, layers=1, scale_factor=1.0,
                        use_sigmoid=True, zero_init=True, use_bias=True)
    scorers = {lbl: LearnedScorer(scfg, pretrained_path=Path(p)).float().to(device).eval()
               for lbl, p in scorers_paths.items()}
    any_scorer = next(iter(scorers.values()))

    per_sample: dict[str, list[tuple[int, float]]] = {lbl: [] for lbl in scorers}
    n_skipped = 0
    row = 0
    shown = 0

    with torch.no_grad():
        for batch in loader:
            f = batch["forget"]
            inputs = {k: f[k].to(device) for k in ("input_ids", "attention_mask", "labels")}
            cache = any_scorer.cache(model, inputs)
            weights = {lbl: s.score(cache)[0].float().cpu() for lbl, s in scorers.items()}
            mask_all = inputs["labels"][:, 1:] != IGNORE_INDEX

            for b in range(inputs["input_ids"].size(0)):
                if row >= len(answers):
                    break
                labels_b = inputs["labels"][b]
                nz = (labels_b != IGNORE_INDEX).nonzero()
                answer_start = int(nz[0].item()) if len(nz) else len(labels_b)
                imp = important_token_indices(tokenizer, answers[row], answer_start,
                                              importance_set.get(row, []))
                m = mask_all[b].cpu()
                gt = np.array([1 if (t + 1) in imp else 0 for t in range(m.numel())])[
                    m.numpy()]

                if shown < args.show_examples and gt.sum():
                    ids = inputs["input_ids"][b].cpu().tolist()
                    toks = [tokenizer.decode([i]) for i in ids]
                    marked = "".join(f"[{t}]" if j in imp else t
                                     for j, t in enumerate(toks[answer_start:],
                                                           start=answer_start))
                    print(f"\n--- sample {row} (labelled tokens in [brackets]) ---\n{marked}")
                    shown += 1

                if gt.sum() == 0 or gt.sum() == len(gt):
                    n_skipped += 1
                    row += 1
                    continue

                for lbl in scorers:
                    s = weights[lbl][b][m].numpy()
                    per_sample[lbl].append((row, float(roc_auc_score(gt, s))))
                row += 1
            del cache

    fold_of = lambda idx: idx // (len(answers) // N_FOLDS) + 1   # noqa: E731
    print(f"\nscored {row} samples, skipped {n_skipped} with degenerate ground truth")

    results = {}
    print(f"\n{'scorer':<12}{'AUROC (all)':>13}{'± stderr':>10}"
          f"{'seen fold':>11}{'unseen folds':>14}{'n':>6}")
    print("-" * 66)
    for lbl, vals in per_sample.items():
        arr = np.array([v for _, v in vals])
        stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        row_out = {"auroc_mean": float(arr.mean()), "stderr": stderr, "n": len(arr)}
        seen_txt = unseen_txt = "-"
        held = heldout_indices_for(lbl)
        if held is not None:
            # "seen" = rows this scorer was co-trained on, "unseen" = its held-out fold
            seen = np.array([v for i, v in vals if i not in held])
            unseen = np.array([v for i, v in vals if i in held])
        elif lbl.startswith("fold"):
            k = int(lbl[4:])
            seen = np.array([v for i, v in vals if fold_of(i) == k])
            unseen = np.array([v for i, v in vals if fold_of(i) != k])
        else:
            seen = unseen = None
        if seen is not None:
            row_out["auroc_seen_fold"] = float(seen.mean()) if len(seen) else float("nan")  # noqa: E501
            row_out["auroc_unseen_folds"] = (float(unseen.mean()) if len(unseen)
                                             else float("nan"))
            row_out["n_seen"], row_out["n_unseen"] = len(seen), len(unseen)
            seen_txt = f"{row_out['auroc_seen_fold']:.4f}"
            unseen_txt = f"{row_out['auroc_unseen_folds']:.4f}"
        results[lbl] = row_out
        print(f"{lbl:<12}{row_out['auroc_mean']:>13.4f}{stderr:>10.4f}"
              f"{seen_txt:>11}{unseen_txt:>14}{len(arr):>6}")

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "labels": str(args.labels), "model": MODEL, "forget_split": FORGET_SPLIT,
        "n_scored": row, "n_skipped": n_skipped,
        "scorers": scorers_paths, "results": results,
        "per_sample": {k: v for k, v in per_sample.items()},
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
