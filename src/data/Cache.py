from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

__all__ = [
    'Cache',
    'SampleCache',
    'SampleCacheWithHidden',
    'SampleCacheWithLogits',
]

def _run_forward(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    *,
    output_hidden_states: bool = False,
    ignore_index: int = -100,
    canonical_ce: bool = False,
) -> tuple[CausalLMOutputWithPast, torch.Tensor]:
    inputs = dict(inputs)
    if output_hidden_states:
        inputs["output_hidden_states"] = True

    outputs: CausalLMOutputWithPast = model(**inputs)

    logits = outputs.logits # (B, T, |V|)
    labels = inputs["labels"] # (B, T)

    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
    if canonical_ce:
        # Flatten to (N, V) — the same layout HF uses for outputs.loss — so the per-token
        # CE reduces in the canonical fp order. The legacy .transpose(-1,-2) form gives CE a
        # strided (B, V, T-1) tensor whose reduction order differs, seeding ~1e-5 perturbations
        # that gradient-ascent unlearning amplifies to ~1.6% over 20 steps. Off by default so
        # existing (TOFU) runs keep their exact numerics.
        token_loss: torch.Tensor = loss_fn(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
        ).view(shifted_labels.shape)
    else:
        token_loss = loss_fn(
            shifted_logits.transpose(-1, -2), shifted_labels
        )

    return outputs, token_loss

@dataclass
class SampleCache:
    inputs: dict[str, Any]
    outputs: CausalLMOutputWithPast
    token_loss: torch.Tensor # (B, T-1); the t'th token correspond to the suprisal of seeing the (t+1)'th token

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any]) -> Self:
        outputs, token_loss = _run_forward(model, inputs)
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss)


@dataclass
class SampleCacheWithLogits(SampleCache):
    logits: torch.Tensor # (B, T, V) on model device, graph alive

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any]) -> Self:
        outputs, token_loss = _run_forward(model, inputs)
        logits = outputs.logits # (B, T, V)
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss, logits=logits)


@dataclass
class SampleCacheWithHidden(SampleCache):
    hidden_states: torch.Tensor # (B, T, D) on model device, graph alive

    @classmethod
    def from_forward(cls, model: PreTrainedModel, inputs: dict[str, Any], *, layer: int = -1,
                     canonical_ce: bool = False) -> Self:
        outputs, token_loss = _run_forward(model, inputs, output_hidden_states=True,
                                           canonical_ce=canonical_ce)
        hidden_states = outputs.hidden_states[layer] # type: ignore
        return cls(inputs=inputs, outputs=outputs, token_loss=token_loss, hidden_states=hidden_states)

Cache = SampleCache | SampleCacheWithLogits | SampleCacheWithHidden
