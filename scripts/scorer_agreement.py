#!/usr/bin/env python
"""Pairwise agreement between ATWU token scorers, on identical inputs.

Companion analysis to the 5-fold scorer-transfer experiment: it asks whether the
scorers the folds learn are the SAME FUNCTION, independently of whether the
end-to-end unlearning metrics happen to agree. Every scorer is evaluated on the
same forget10 batches under the same base model (one forward pass, N scorers), so
differences are purely in the learned map from hidden states to token importance.

Reported per pair:
    spearman / pearson   correlation over all valid forget tokens (pooled)
    jaccard@top20        overlap of each scorer's own top-20% of tokens
    jaccard@selected     overlap of the tokens each scorer scores > 0.5, which is
                         the set the population budget rho actually controls

Reported per scorer: score mean/std, selected fraction, and the token strings it
scores highest (what it actually learned to target).

No training, no gradients — one model forward pass over forget10.

Usage:
    python scripts/scorer_agreement.py                       # auto-discover sweep scorers
    python scripts/scorer_agreement.py --scorers a=path/a.pt b=path/b.pt
    python scripts/scorer_agreement.py --gpu 0 --max-batches 10
"""

from __future__ import annotations

import argparse
import itertools
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


def discover_scorers() -> dict[str, str]:
    """Default scorer set: the 5 fold scorers, the sweep reference, and the
    in-domain co-trained scorer from the paper repro run."""
    found: dict[str, str] = {}
    for p in sorted((SWEEP_ROOT / "exp2").glob("fold*/scorer/scorer.pt")):
        found[p.parent.parent.name] = str(p)
    ref = SWEEP_ROOT / "exp1/reference/scorer.pt"
    if ref.exists():
        found["reference"] = str(ref)
    repro = REPO_ROOT / f"saves/unlearn/TOFU_repro/{MODEL}/{FORGET_SPLIT}/ATWU/scorer.pt"
    if repro.exists():
        found["repro_forget10"] = str(repro)
    return found


def rankdata(x):
    """Average-tie ranks; scipy if available, numpy fallback."""
    try:
        from scipy.stats import rankdata as _rd  # type: ignore
        return _rd(x)
    except Exception:
        import numpy as np
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x), dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        return ranks


def corr(a, b) -> float:
    import numpy as np
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorers", nargs="*", default=None,
                    help="label=path pairs; default auto-discovers the sweep scorers")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Limit batches (default: all of forget10)")
    ap.add_argument("--top-tokens", type=int, default=15)
    ap.add_argument("--out", default=str(SWEEP_ROOT / "scorer_agreement.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import reproduce_tofu as rt
    rt.load_dotenv(REPO_ROOT / ".env")

    if args.scorers:
        scorers_paths = {}
        for spec in args.scorers:
            if "=" not in spec:
                sys.exit(f"--scorers expects label=path, got {spec!r}")
            label, path = spec.split("=", 1)
            scorers_paths[label] = path
    else:
        scorers_paths = discover_scorers()
    if len(scorers_paths) < 1:
        sys.exit("no scorers found — run the sweep first, or pass --scorers label=path")
    missing = [p for p in scorers_paths.values() if not Path(p).exists()]
    if missing:
        sys.exit(f"missing scorer checkpoints: {missing}")

    print(f"scorers: {json.dumps(scorers_paths, indent=2)}")

    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from torch.utils.data import DataLoader

    from data import get_collators, get_data
    from model import get_model
    from scorers import LearnedScorer, ScorerConfig

    overrides = [
        "experiment=unlearn/tofu/default",
        "trainer=ATWU",
        f"model={MODEL}",
        "task_name=scorer_agreement",
        f"model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_{MODEL}_full",
        "++model.model_args.device_map=auto",
        f"forget_split={FORGET_SPLIT}",
        f"retain_split={RETAIN_SPLIT}",
        *rt.hf_token_overrides(),
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="unlearn.yaml", overrides=overrides)

    model, tokenizer = get_model(cfg.model)
    model.eval()
    template_args = cfg.model.template_args
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator)

    device = next(model.parameters()).device
    scorer_cfg = ScorerConfig(input_dimension=INPUT_DIM, layers=1, scale_factor=1.0,
                              use_sigmoid=True, zero_init=True, use_bias=True)
    scorers = {}
    for label, path in scorers_paths.items():
        s = LearnedScorer(scorer_cfg, pretrained_path=Path(path))
        scorers[label] = s.float().to(device).eval()
    any_scorer = next(iter(scorers.values()))

    pooled: dict[str, list] = {label: [] for label in scorers}
    token_ids: list = []
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["forget"]["input_ids"].to(device),
                "attention_mask": batch["forget"]["attention_mask"].to(device),
                "labels": batch["forget"]["labels"].to(device),
            }
            # One forward pass; every scorer reads the same cached hidden states.
            cache = any_scorer.cache(model, inputs)
            mask = None
            for label, scorer in scorers.items():
                weights, m = scorer.score(cache)
                mask = m if mask is None else mask
                pooled[label].append(weights[m].float().cpu().numpy())
            # Score t corresponds to token t+1 (see LearnedScorer.score).
            token_ids.append(inputs["input_ids"][:, 1:][mask].cpu().numpy())
            n_batches += 1
            if args.max_batches and n_batches >= args.max_batches:
                break
            del cache

    vecs = {label: np.concatenate(v) for label, v in pooled.items()}
    ids = np.concatenate(token_ids)
    n_tok = len(ids)
    print(f"\npooled {n_tok} valid forget tokens over {n_batches} batches\n")

    # ── Per-scorer distribution ──
    per_scorer = {}
    for label, v in vecs.items():
        thresh = np.quantile(v, 0.8)
        top_idx = np.argsort(-v)[:max(1, int(0.05 * n_tok))]
        top_tok_counts: dict[str, list[float]] = {}
        for i in top_idx:
            tok = tokenizer.decode([int(ids[i])])
            top_tok_counts.setdefault(tok, []).append(float(v[i]))
        top_tokens = sorted(top_tok_counts.items(), key=lambda kv: -len(kv[1]))
        per_scorer[label] = {
            "mean": float(v.mean()), "std": float(v.std()),
            "selected_frac": float((v > 0.5).mean()),
            "p80": float(thresh),
            "top_tokens": [
                {"token": t, "count": len(s), "mean_score": float(np.mean(s))}
                for t, s in top_tokens[:args.top_tokens]
            ],
        }

    print(f"{'scorer':<16}{'mean':>8}{'std':>8}{'selected':>10}")
    print("-" * 42)
    for label, st in per_scorer.items():
        print(f"{label:<16}{st['mean']:>8.4f}{st['std']:>8.4f}"
              f"{st['selected_frac'] * 100:>9.1f}%")

    # ── Pairwise agreement ──
    ranks = {label: rankdata(v) for label, v in vecs.items()}
    pairs = {}
    labels = list(vecs)
    for a, b in itertools.combinations(labels, 2):
        va, vb = vecs[a], vecs[b]
        top_a, top_b = va >= np.quantile(va, 0.8), vb >= np.quantile(vb, 0.8)
        sel_a, sel_b = va > 0.5, vb > 0.5
        union_sel = (sel_a | sel_b).sum()
        pairs[f"{a}|{b}"] = {
            "spearman": corr(ranks[a], ranks[b]),
            "pearson": corr(va, vb),
            "jaccard_top20": float((top_a & top_b).sum() / max((top_a | top_b).sum(), 1)),
            "jaccard_selected": (float((sel_a & sel_b).sum() / union_sel)
                                 if union_sel else float("nan")),
        }

    if len(labels) > 1:
        print(f"\n{'pair':<26}{'spearman':>10}{'pearson':>10}"
              f"{'J@top20':>10}{'J@sel':>10}")
        print("-" * 66)
        for pair, st in pairs.items():
            print(f"{pair:<26}{st['spearman']:>10.3f}{st['pearson']:>10.3f}"
                  f"{st['jaccard_top20']:>10.3f}{st['jaccard_selected']:>10.3f}")

        # Spearman matrix, easiest thing to read at a glance.
        print("\nspearman matrix")
        print("".ljust(16) + "".join(f"{l[:9]:>10}" for l in labels))
        for a in labels:
            row = ""
            for b in labels:
                if a == b:
                    row += f"{1.0:>10.3f}"
                else:
                    key = f"{a}|{b}" if f"{a}|{b}" in pairs else f"{b}|{a}"
                    row += f"{pairs[key]['spearman']:>10.3f}"
            print(f"{a[:15]:<16}{row}")

    out = Path(REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scorers": scorers_paths,
        "n_tokens": int(n_tok),
        "n_batches": n_batches,
        "forget_split": FORGET_SPLIT,
        "model": MODEL,
        "per_scorer": per_scorer,
        "pairs": pairs,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
