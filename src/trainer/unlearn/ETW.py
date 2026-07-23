from __future__ import annotations

from typing import Any

import torch

from scorers.base import TokenImportanceScorer
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import reweighted_NLL


class ETWGradDiff(UnlearnTrainer):
    """GradDiff with entropy-weighted forget loss.

    Forget loss is `reweighted_NLL(cache, scorer, beta=None)` — per-token NLL
    scaled by the scorer's importance weights. Retain loss is standard NLL.
    Total loss:

        L = alpha * retain_NLL  -  gamma * weighted_forget_NLL

    The scorer is non-learnable (no optimizer, no scorer trainer).
    """

    def __init__(
        self,
        *,
        scorer: TokenImportanceScorer,
        alpha: float = 1.0,
        gamma: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.scorer = scorer
        self.alpha = alpha
        self.gamma = gamma

    def pack_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache = self.scorer.cache(model, forget_inputs)

        forget_loss, _, _ = reweighted_NLL(
            cache=forget_cache, scorer=self.scorer, beta=None,
        )
        forget_loss = -forget_loss

        retain_loss: torch.Tensor = model(**retain_inputs).loss

        loss = self.alpha * retain_loss + self.gamma * forget_loss
        return (loss, forget_cache.outputs) if return_outputs else loss
