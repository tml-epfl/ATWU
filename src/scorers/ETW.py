from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from data.Cache import SampleCacheWithLogits
from scorers.base import TokenImportanceScorer, WeightResult


class ETWScorer(TokenImportanceScorer[SampleCacheWithLogits]):
    """ETW: Entropy Token Weighting.

    ``w_i = n * H(y_i | y_<i, x; theta_hat) / sum_j H(y_j | y_<j, x; theta_hat)``

    where the entropy is the Shannon entropy of the conditional next-token
    distribution. Weights are normalized per sample so that each row sums to
    ``n_b`` (the number of valid answer tokens in that sample).
    """

    cache_cls = SampleCacheWithLogits

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def score(self, cache: SampleCacheWithLogits, **kwargs: Any) -> WeightResult:
        mask = self._build_valid_mask(cache)                       # (B, T-1)
        shifted_logits = cache.logits[:, :-1, :].detach().float()  # (B, T-1, V)

        log_probs = F.log_softmax(shifted_logits / self.temperature, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)                 # (B, T-1)

        masked_entropy = entropy.masked_fill(~mask, 0.0)
        n = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, 1)
        denom = masked_entropy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = n * masked_entropy / denom

        weights = self._apply_mask(weights, mask)
        return WeightResult(weights=weights, valid_mask=mask)
