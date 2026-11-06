#!/usr/bin/env python
"""Generate WMDP forget / retain corpora with the target model itself.

Two stages:
  A) expand each seed category into concrete subtopics (model-generated, so the
     subtopic distribution is also a readout of what the model knows);
  B) generate one long-form prose passage per (subtopic, framing) pair.

Everything is resumable: stage A writes <out>.subtopics.json, stage B appends
one JSON object per passage to <out> immediately (flush + fsync), keyed by a
unique `key`. On restart, keys already present are skipped.

Quality gates are applied at write time; rejected generations go to
<out>.rejected.jsonl with a reason so the reject rate is inspectable.

Usage:
  python benchmarks/wmdp-generated-v1/provenance/gen_corpus.py --domain bio   --n 3200 --out data/wmdp/generated/bio-forget.jsonl
  python benchmarks/wmdp-generated-v1/provenance/gen_corpus.py --domain cyber --n 3200 --out data/wmdp/generated/cyber-forget.jsonl
  python benchmarks/wmdp-generated-v1/provenance/gen_corpus.py --domain retain --n 4000 --out data/wmdp/generated/general-retain.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topics import (  # noqa: E402
    BIO_CATEGORIES,
    CYBER_CATEGORIES,
    FRAMINGS,
    MMLU_SUBJECTS,
    STYLE_SUFFIX,
    SUBTOPIC_PROMPT_HAZARD,
    SUBTOPIC_PROMPT_RETAIN,
    SYSTEM_PROMPT,
    subject_label,
)

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

REFUSAL_PATTERNS = [
    r"\bI (?:can(?:no|')t|cannot|won't|will not|am not able to|'m not able to)\b",
    r"\bI (?:do not|don't) (?:have|feel comfortable|think it)\b",
    r"\bas an AI\b",
    r"\bI'm sorry\b",
    r"\bI am sorry\b",
    r"\bI apologize\b",
    r"\bit (?:would be |is )?(?:not )?(?:in)?appropriate for me\b",
    r"\bagainst my (?:programming|guidelines|principles)\b",
    r"\bI must (?:decline|refuse)\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# Boilerplate openers to strip so the passage is on-topic from the first token.
PREAMBLE_RE = re.compile(
    r"^\s*(?:sure[,!.]?|certainly[,!.]?|of course[,!.]?|here(?:'s| is)[^\n.]*[.:]?"
    r"|below is[^\n.]*[.:]?|the following[^\n.]*[.:]?)\s*",
    re.IGNORECASE,
)


def strip_preamble(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = PREAMBLE_RE.sub("", text, count=1).lstrip()
    return text


def max_ngram_repeat_ratio(text: str, n: int = 8) -> float:
    """Fraction of n-grams that are duplicates. Degenerate loops score high."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def nonascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


class Gen:
    def __init__(self, model_id=MODEL_ID, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # LEFT padding is mandatory for batched decoder-only generation.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            device_map="cuda",
        )
        self.model.eval()

    def chat(self, user_msgs, max_new_tokens, temperature, top_p, batch_size, seed):
        """Batched chat generation. Returns list[str] aligned with user_msgs."""
        out = []
        for start in range(0, len(user_msgs), batch_size):
            chunk = user_msgs[start : start + batch_size]
            prompts = [
                self.tok.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": m},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for m in chunk
            ]
            enc = self.tok(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=1024, add_special_tokens=True,  # zephyr template has no BOS
            ).to(self.model.device)
            torch.manual_seed(seed + start)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    pad_token_id=self.tok.pad_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1] :]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
            done = min(start + batch_size, len(user_msgs))
            print(f"    [gen] {done}/{len(user_msgs)}", flush=True)
        return out


LIST_LINE_RE = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def parse_list(text, want):
    items, seen = [], set()
    for line in text.splitlines():
        m = LIST_LINE_RE.match(line)
        if not m:
            continue
        t = m.group(1).strip().strip("*_`").strip()
        t = re.sub(r"\s+", " ", t)
        if not (3 <= len(t.split()) <= 20):
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        items.append(t)
        if len(items) >= want:
            break
    return items


def stage_a(gen, domain, per_cat, out_path, seed, batch_size):
    """Expand seed categories into subtopics. Cached on disk."""
    cache = Path(str(out_path) + ".subtopics.json")
    if cache.exists():
        d = json.loads(cache.read_text())
        print(f"[stage A] cached: {sum(len(v) for v in d.values())} subtopics", flush=True)
        return d

    if domain == "bio":
        cats, tmpl = BIO_CATEGORIES, SUBTOPIC_PROMPT_HAZARD
    elif domain == "cyber":
        cats, tmpl = CYBER_CATEGORIES, SUBTOPIC_PROMPT_HAZARD
    else:
        cats = [subject_label(s) for s in MMLU_SUBJECTS]
        tmpl = SUBTOPIC_PROMPT_RETAIN

    # Two passes with different sampling seeds so a category that under-fills on
    # the first pass gets a second chance (and gets more diverse subtopics).
    result = {c: [] for c in cats}
    for rnd in range(2):
        need = [c for c in cats if len(result[c]) < per_cat]
        if not need:
            break
        prompts = [tmpl.format(n=per_cat, category=c) for c in need]
        print(f"[stage A] round {rnd}: {len(need)} categories", flush=True)
        texts = gen.chat(
            prompts, max_new_tokens=1100, temperature=0.9, top_p=0.95,
            batch_size=batch_size, seed=seed + 1000 * rnd,
        )
        for c, t in zip(need, texts):
            have = {x.lower() for x in result[c]}
            for item in parse_list(t, per_cat * 2):
                if item.lower() not in have:
                    result[c].append(item)
                    have.add(item.lower())
            result[c] = result[c][: per_cat]

    n = sum(len(v) for v in result.values())
    thin = [c for c, v in result.items() if len(v) < max(4, per_cat // 3)]
    print(f"[stage A] {n} subtopics over {len(cats)} categories; thin: {thin}", flush=True)
    cache.write_text(json.dumps(result, indent=1))
    return result


def build_jobs(subtopics, n_target):
    """(category, subtopic, framing) triples, round-robin over categories so a
    truncated run is still balanced across the taxonomy."""
    cats = sorted(subtopics)
    jobs, f_idx = [], 0
    max_t = max((len(subtopics[c]) for c in cats), default=0)
    for rep in range(len(FRAMINGS)):
        for ti in range(max_t):
            for c in cats:
                lst = subtopics[c]
                if ti >= len(lst):
                    continue
                topic = lst[ti]
                framing = FRAMINGS[(ti + rep + f_idx) % len(FRAMINGS)]
                jobs.append(
                    {
                        "key": f"{c}||{ti}||{rep}",
                        "category": c,
                        "topic": topic,
                        "prompt": framing.format(topic=topic) + STYLE_SUFFIX,
                    }
                )
                if len(jobs) >= n_target:
                    return jobs
            f_idx += 1
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["bio", "cyber", "retain"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3200)
    ap.add_argument("--per-cat", type=int, default=None,
                    help="subtopics per seed category (default: bio/cyber 40, retain 14)")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--min-words", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.per_cat is None:
        args.per_cat = 50 if args.domain in ("bio", "cyber") else 14
    if args.max_new_tokens is None:
        # Training truncates each doc to the FIRST max_length tokens (bio 768,
        # cyber/retain 512) and never sees the tail, so generating far past that
        # is pure wasted compute. Measured: every sample runs to the cap (p10 of
        # the first batch was 749/750), so cost is exactly proportional to this
        # number. Keep a small margin over the truncation length.
        args.max_new_tokens = 820 if args.domain == "bio" else 560

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej_path = Path(str(out) + ".rejected.jsonl")

    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    if rej_path.exists():
        with rej_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    print(f"[init] domain={args.domain} target={args.n} already={len(done)}", flush=True)

    gen = Gen()
    subs = stage_a(gen, args.domain, args.per_cat, out, args.seed, args.batch_size)

    # Over-provision jobs so rejects don't starve the target.
    jobs = [j for j in build_jobs(subs, int(args.n * 1.35) + 64) if j["key"] not in done]
    n_have = sum(1 for _ in out.open()) if out.exists() else 0
    print(f"[stage B] {len(jobs)} jobs pending, {n_have} passages already kept", flush=True)

    t0 = time.time()
    fh = out.open("a")
    rf = rej_path.open("a")
    kept = n_have
    for start in range(0, len(jobs), args.batch_size):
        if kept >= args.n:
            print(f"[stage B] target {args.n} reached", flush=True)
            break
        chunk = jobs[start : start + args.batch_size]
        texts = gen.chat(
            [j["prompt"] for j in chunk],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            seed=args.seed + 7919 * start,
        )
        for j, raw in zip(chunk, texts):
            text = strip_preamble(raw.strip())
            nw = len(text.split())
            reason = None
            if REFUSAL_RE.search(text[:600]):
                reason = "refusal"
            elif nw < args.min_words:
                reason = f"short:{nw}"
            elif max_ngram_repeat_ratio(text) > 0.15:
                reason = "repetitive"
            elif nonascii_ratio(text) > 0.05:
                reason = "nonascii"
            rec = {"key": j["key"], "category": j["category"], "topic": j["topic"],
                   "n_words": nw, "text": text}
            if reason:
                rec["reason"] = reason
                rf.write(json.dumps(rec) + "\n")
            else:
                fh.write(json.dumps(rec) + "\n")
                kept += 1
        for h in (fh, rf):
            h.flush()
            os.fsync(h.fileno())
        el = time.time() - t0
        print(f"[stage B] kept={kept}/{args.n} elapsed={el/60:.1f}m", flush=True)
    fh.close()
    rf.close()
    print(f"[done] {kept} passages -> {out}", flush=True)


if __name__ == "__main__":
    main()
