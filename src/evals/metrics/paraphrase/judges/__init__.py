"""Judge infrastructure for paraphrase evaluation.

Provides judge backends (local, OpenAI, Gemini) and orchestrators
for quality judging (YES/NO) and win-rate comparison.
"""
from evals.metrics.paraphrase.judges.local import LocalJudge
from evals.metrics.paraphrase.judges.quality import QualityJudge
from evals.metrics.paraphrase.judges.winrate import WinrateJudge

__all__ = [
    "LocalJudge",
    "QualityJudge",
    "WinrateJudge"
]
