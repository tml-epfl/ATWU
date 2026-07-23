from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar

import torch
from torch import nn
from transformers import PreTrainedModel

from data.Cache import *

CacheT = TypeVar("CacheT", bound=SampleCache)

__all__ = [
    'CacheT',
    'WeightResult',
    'TokenImportanceScorer',
]


class WeightResult(NamedTuple):
    weights: torch.Tensor # (B, T) per-token importance weights
    valid_mask: torch.Tensor # (B, T) bool mask; True = valid token


class TokenImportanceScorer(nn.Module, ABC, Generic[CacheT]):
    IGNORE_INDEX: int = -100
    cache_cls: type[CacheT]

    def prepare(self, *args: Any, **kwargs: Any) -> None:
        """Prepare prerequisites (init models, compute stats, etc.).

        Default implementation does nothing.
        """

    def prepare_forward(
        self, 
        model: PreTrainedModel, 
        inputs: dict[str, Any], 
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Pre-forward setup (hooks, embeddings).

        Returns (possibly modified) inputs dict for ``Cache.from_forward``.
        Default implementation returns inputs unchanged.
        """
        return inputs

    def after_forward(self, cache: CacheT, **kwargs: Any) -> Any:
        """Post-forward gradient extraction.

        Returns precomputed data to pass to ``score()``.  Also cleans up
        per-forward hooks and transient tensors.
        """
        return cache

    def cleanup_forward(self) -> None:
        """Safety-net cleanup if ``after_forward`` was not called."""

    def _build_valid_mask(self, cache: SampleCache) -> torch.Tensor:
        """bool mask (B, T-1): True where label != IGNORE_INDEX, on same device as token_loss."""
        labels = cache.inputs["labels"]
        mask = labels[:, 1:] != self.IGNORE_INDEX
        # mask = labels[:, :-1] != self.IGNORE_INDEX
        return mask.to(cache.token_loss.device)

    def _apply_mask(self, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return weights.masked_fill(~mask, 0.0)

    @abstractmethod
    def score(self, cache: CacheT, **kwargs: Any) -> WeightResult:
        """Compute per-token importance scores.

        Args:
            cache: `Cache` or inherited from `Cache`
            **kwargs: Scorer-specific extra arguments.

        Returns:
            WeightResult with weights (B, T) and valid_mask (B, T).
            Ignored tokens have weight 0.
        """
        raise NotImplementedError

    def cache(self, model: PreTrainedModel, inputs: dict[str, Any], **kwargs):
        prepared_sample = self.prepare_forward(model, inputs)
        try:
            cached_sample = self.cache_cls.from_forward(model, prepared_sample, **kwargs)
            self.after_forward(cached_sample)
            return cached_sample
        finally:
            self.cleanup_forward()

