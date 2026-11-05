#!/usr/bin/env python
"""Verify the TOFU split structure the k-fold scorer-transfer experiment assumes.

Two assumptions are checked:

1. NESTING — forget01 ⊂ forget05 ⊂ forget10 (exact question-string containment).
   Determines whether a scorer trained on one standard split has already seen
   part of another; the k-fold design exists to avoid this, but the nesting is
   worth recording in the writeup.

2. AUTHOR-CONTIGUITY of forget10 — 20 authors × 20 consecutive QA rows, so that
   the contiguous slices used as folds (train[0:80], train[80:160], ...) are
   author-DISJOINT. If this fails, the folds are random 20% subsets instead, and
   the experiment tests generalisation to unseen tokens of seen authors rather
   than to unseen authors. The claim in the paper must then be weakened.

Author identity is not a dataset field, so contiguity is checked by extracting
capitalised name candidates from each question and testing whether one candidate
dominates each 20-row block.

Usage:
    python scripts/check_tofu_folds.py                      # forget10, 5 folds
    python scripts/check_tofu_folds.py --folds 4
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Capitalised multi-word sequences, e.g. "Hsiao Yun-Hwa" — TOFU questions name
# the author they are about. Hyphens/apostrophes are part of names.
NAME_RE = re.compile(r"\b[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)+")

# Sentence-initial and generic capitalised phrases that are not author names.
STOPPHRASES = {
    "What", "Who", "How", "Why", "When", "Where", "Which", "Can", "Could",
    "Did", "Do", "Does", "Has", "Have", "Is", "In", "The", "This", "That",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def name_candidates(text: str) -> list[str]:
    out = []
    for m in NAME_RE.finditer(text):
        phrase = m.group(0)
        head = phrase.split()[0]
        if head in STOPPHRASES:
            # Drop the leading interrogative/article, keep the rest if still 2+ words.
            rest = phrase.split()[1:]
            if len(rest) < 2:
                continue
            phrase = " ".join(rest)
        # "Hsiao Yun-Hwa's" and "Hsiao Yun-Hwa" are the same author.
        phrase = re.sub(r"['’]s$", "", phrase).strip()
        out.append(phrase)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="forget10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--qa-per-author", type=int, default=20)
    ap.add_argument("--min-coverage", type=float, default=0.4,
                    help="Minimum share of a block's questions that must name the "
                         "block's dominant author (default 0.4; the rest use pronouns)")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    from datasets import load_dataset  # imported late so --help works without deps

    splits = {}
    for name in ("forget01", "forget05", "forget10"):
        splits[name] = load_dataset("locuslab/TOFU", name, split="train")
        print(f"{name}: {len(splits[name])} rows, columns={splits[name].column_names}")

    # ── 1. Nesting ──
    print("\n[nesting] exact question-string containment")
    qsets = {k: set(v["question"]) for k, v in splits.items()}
    for small, big in (("forget01", "forget05"), ("forget05", "forget10"),
                       ("forget01", "forget10")):
        inter = len(qsets[small] & qsets[big])
        frac_small = inter / max(len(qsets[small]), 1)
        frac_big = inter / max(len(qsets[big]), 1)
        verdict = "SUBSET" if frac_small == 1.0 else "NOT a subset"
        print(f"  {small} ⊂ {big}? {verdict} "
              f"({inter} shared = {frac_small:.0%} of {small}, {frac_big:.0%} of {big})")

    # ── 2. Author-contiguity of the fold split ──
    data = splits[args.split]
    n = len(data)
    per_author = args.qa_per_author
    print(f"\n[contiguity] {args.split}: {n} rows, assuming {per_author} QA/author "
          f"→ {n // per_author} authors")

    # Many TOFU questions refer to the author by pronoun ("What is their writing
    # style?"), so a block's author is named in only ~half of its rows. What
    # establishes contiguity is not high coverage but that ONE name dominates each
    # block and that no name recurs in another block — random row subsets would
    # scatter the same names across blocks instead.
    block_names: list[str] = []
    block_coverage: list[float] = []
    all_block_name_sets: list[set[str]] = []
    for start in range(0, n, per_author):
        block = data.select(range(start, min(start + per_author, n)))
        counts: Counter[str] = Counter()
        for q in block["question"]:
            # Count each candidate at most once per question.
            for cand in set(name_candidates(q)):
                counts[cand] += 1
        all_block_name_sets.append(set(counts))
        if not counts:
            block_names.append("<none>")
            block_coverage.append(0.0)
            continue
        top, hits = counts.most_common(1)[0]
        block_names.append(top)
        block_coverage.append(hits / len(block))

    n_blocks = len(block_names)
    distinct = len(set(block_names))
    named = sum(c >= args.min_coverage for c in block_coverage)
    # Does each block's dominant name appear in any OTHER block at all?
    recurring = [
        name for i, name in enumerate(block_names)
        if any(name in s for j, s in enumerate(all_block_name_sets) if j != i)
    ]
    print(f"  {named}/{n_blocks} blocks have a dominant name in ≥{args.min_coverage:.0%} "
          f"of rows (mean coverage {sum(block_coverage) / max(n_blocks, 1):.0%})")
    print(f"  {distinct}/{n_blocks} distinct dominant names; "
          f"{len(recurring)} of them also appear in another block")
    for i, (name, cov) in enumerate(zip(block_names, block_coverage)):
        if cov < args.min_coverage:
            print(f"    rows {i * per_author:3d}-{i * per_author + per_author - 1:3d}: "
                  f"weak dominant name {name!r} ({cov:.0%})")
    contiguous = named == n_blocks and distinct == n_blocks and not recurring

    # ── 3. Fold report ──
    fold_size = n // args.folds
    authors_per_fold = fold_size // per_author
    print(f"\n[folds] {args.folds} folds of {fold_size} rows "
          f"(= {authors_per_fold} authors each)")
    for k in range(args.folds):
        lo, hi = k * fold_size, (k + 1) * fold_size
        names = block_names[lo // per_author:hi // per_author]
        print(f"  fold {k + 1}: train[{lo}:{hi}] -> {names}")

    overlap = set()
    for k in range(args.folds):
        lo, hi = k * fold_size, (k + 1) * fold_size
        names = set(block_names[lo // per_author:hi // per_author])
        overlap |= (names & set().union(*[
            set(block_names[j * fold_size // per_author:(j + 1) * fold_size // per_author])
            for j in range(args.folds) if j != k
        ]))

    print()
    if contiguous and not overlap:
        print("VERDICT: forget10 is author-ordered and the contiguous folds are "
              "author-disjoint. The k-fold design holds as planned.")
    else:
        print("VERDICT: could NOT confirm author-disjoint folds "
              f"(single-name blocks: {ok_blocks}/{n_blocks}, "
              f"names shared across folds: {sorted(overlap) or 'none'}).")
        print("The folds are still disjoint ROW subsets, so the experiment runs, but")
        print("the claim must be stated as generalisation across forget-set rows,")
        print("not across authors/entities.")


if __name__ == "__main__":
    main()
