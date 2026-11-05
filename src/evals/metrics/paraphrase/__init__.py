"""Paraphrase-based metrics for unlearning evaluation.

- quality.py        — forget + retain quality using LLM judge
- repetitiveness.py — n-gram entropy for measuring text repetitiveness
- winrate.py        — win rate comparing unlearned vs pretrained model
- aggregation.py    — WorstEval (J_W) and AvgEval (J_avg) aggregators
- generation.py     — Evaluator class (output directory / file management)
- judges/           — LLM judge backends (local, OpenAI, Gemini)
- utils.py          — slim helpers (API key, logger setup)
"""
from evals.metrics.paraphrase.quality import forget_quality, retain_quality
from evals.metrics.paraphrase.repetitiveness import repetitiveness
from evals.metrics.paraphrase.winrate import winrate

__all__ = [
    "forget_quality",
    "retain_quality",
    "repetitiveness",
    "winrate",
]
