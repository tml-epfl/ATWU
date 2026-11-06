#!/usr/bin/env python
"""Generate the WMDP training corpora reproducibly, or check that they exist.

This is the executable wrapper around the archived generator under
``benchmarks/wmdp-generated-v1/provenance``. It pins the Zephyr revision,
sampling parameters, batch size and seeds; enables deterministic CUDA settings;
and writes the filenames consumed by ``experiment=unlearn/wmdp/gen_corpora``.
``--check`` only verifies that the three corpus files exist with enough rows.

Fresh runs in the same software/hardware environment are expected to be
byte-identical. Cross-hardware or cross-version identity is not promised.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

# Must be set before CUDA libraries are initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as common  # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
ARCHIVE =PROJ / "benchmarks" / "wmdp-generated-v1" / "provenance"
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
MODEL_REVISION = "892b3d7a7b1cf10c7a701c60881cd93df615734c"

# domain: (training filename, frozen subtopics, target rows, subtopics/category,
#          max new tokens)
SPECS = {
    "bio": ("bio-forget.text.jsonl", "bio-forget.subtopics.json", 3200, 50, 820),
    "cyber": ("cyber-forget.text.jsonl", "cyber-forget.subtopics.json", 3200, 50, 560),
    "retain": ("general-retain.text.jsonl", "general-retain.subtopics.json", 4000, 14, 560),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def load_archived_generator():
    source = ARCHIVE / "gen_corpus.py"
    spec = importlib.util.spec_from_file_location("wmdp_archived_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJ / "data" / "wmdp" / "generated",
        help="Destination for training-ready JSONL files",
    )
    parser.add_argument(
        "--domains", nargs="+", choices=tuple(SPECS), default=list(SPECS)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument(
        "--target-rows",
        type=int,
        help="Override row target for one domain (useful for smoke tests)",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--regenerate-subtopics",
        action="store_true",
        help="Sample Stage-A subtopics instead of using the frozen lists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing outputs; use fresh directories for exact repeat checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check that --output-dir holds complete corpora; generate nothing",
    )
    args = parser.parse_args()

    if args.check:
        common.ensure_corpus(args.output_dir)
        return
    if args.target_rows is not None and len(args.domains) != 1:
        parser.error("--target-rows requires exactly one --domains value")
    if args.target_rows is not None and args.target_rows < 1:
        parser.error("--target-rows must be positive")

    output_dir = args.output_dir.resolve()
    plans = []
    for domain in args.domains:
        filename, subtopics_name, target, per_cat, max_tokens = SPECS[domain]
        if args.target_rows is not None:
            target = args.target_rows
        out = output_dir / filename
        cache = Path(f"{out}.subtopics.json")
        rejected = Path(f"{out}.rejected.jsonl")
        existing = [path for path in (out, rejected) if path.exists()]
        if existing and not args.resume and not args.dry_run:
            names = ", ".join(str(path) for path in existing)
            raise SystemExit(
                f"existing generation output: {names}\n"
                "use a fresh --output-dir, or pass --resume"
            )
        if (
            args.regenerate_subtopics
            and cache.exists()
            and not args.resume
            and not args.dry_run
        ):
            raise SystemExit(f"existing Stage-A cache: {cache}; use a fresh output directory")
        plans.append((domain, out, cache, subtopics_name, target, per_cat, max_tokens))

    for domain, out, _cache, _subtopics, target, per_cat, max_tokens in plans:
        print(
            f"{domain}: {out} n>={target} per_cat={per_cat} "
            f"max_new_tokens={max_tokens} seed={args.seed} batch={args.batch_size} "
            f"temperature={args.temperature} top_p={args.top_p}"
        )
    print(f"model: {MODEL_ID}@{args.model_revision}")
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for Zephyr corpus generation")
    # Strict deterministic-algorithm mode is incompatible with CUDA top-p
    # sampling. Reproducibility here is defined for the pinned stack/GPU, with
    # the per-batch RNG seeds set by the archived generator.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False

    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=args.model_revision,
        allow_patterns=["*.json", "*.model", "*.safetensors"],
    )
    generator = load_archived_generator()
    generator.MODEL_ID = snapshot
    output_dir.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv[:]
    try:
        for domain, out, cache, subtopics_name, target, per_cat, max_tokens in plans:
            if not args.regenerate_subtopics and not cache.exists():
                shutil.copyfile(ARCHIVE / "subtopics" / subtopics_name, cache)
            sys.argv = [
                str(ARCHIVE / "gen_corpus.py"),
                "--domain", domain,
                "--out", str(out),
                "--n", str(target),
                "--per-cat", str(per_cat),
                "--max-new-tokens", str(max_tokens),
                "--batch-size", str(args.batch_size),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--seed", str(args.seed),
            ]
            generator.main()
            if not out.exists() or line_count(out) < target:
                raise RuntimeError(f"{domain}: generation did not reach {target} rows")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        sys.argv = old_argv

    files = {}
    for domain, out, cache, _subtopics, _target, _per_cat, _max_tokens in plans:
        files[domain] = {
            "file": out.name,
            "rows": line_count(out),
            "sha256": sha256(out),
            "subtopics_sha256": sha256(cache),
        }
    manifest = {
        "model": MODEL_ID,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": 1.05,
        "subtopics": "regenerated" if args.regenerate_subtopics else "frozen",
        "domains": {
            domain: {
                "target_rows": target,
                "subtopics_per_category": per_cat,
                "max_new_tokens": max_tokens,
            }
            for domain, _out, _cache, _name, target, per_cat, max_tokens in plans
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "flash_attn": package_version("flash-attn"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "files": files,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
