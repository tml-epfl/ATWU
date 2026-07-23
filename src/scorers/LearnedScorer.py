from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from transformers import PreTrainedModel

from data.Cache import *
from scorers.base import TokenImportanceScorer, WeightResult


class ScorerConfig(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    input_dimension: int = Field(2048, gt=0, description="Hidden dimension size (> 0)")
    layers: int = Field(1, ge=0, description="Number of layers (>= 0)")
    scale_factor: float = Field(1.0, gt=0, description="Loss scale factor (> 0)")
    use_sigmoid: bool = True
    zero_init: bool = True
    use_bias: bool = True
    normalize_hidden: bool = False

class Scorer(nn.Sequential):
    def __init__(self, cfg: ScorerConfig):
        self._config = cfg

        modules: list[nn.Module] = []

        in_dim = out_dim = cfg.input_dimension

        for i in range(cfg.layers - 1):
            in_dim = max(1, int(cfg.input_dimension / (cfg.scale_factor ** i)))
            out_dim = max(1, int(cfg.input_dimension / (cfg.scale_factor ** (i + 1))))

            modules.append(nn.Linear(in_dim, out_dim, bias=cfg.use_bias))
            modules.append(nn.GELU())

        projection = nn.Linear(out_dim, 1, bias=cfg.use_bias)
        if cfg.zero_init:
            nn.init.zeros_(projection.weight)
            if cfg.use_bias:
                nn.init.zeros_(projection.bias)

        modules.append(projection)

        if cfg.use_sigmoid:
            modules.append(nn.Sigmoid())

        # Initialize Sequential with the built modules
        super().__init__(*modules)

        self.dtype = next(self.parameters()).dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.to(self.dtype)).squeeze(-1)

class LearnedScorer(TokenImportanceScorer[SampleCacheWithHidden]):
    cache_cls = SampleCacheWithHidden

    def __init__(self, cfg: ScorerConfig, pretrained_path: Path | None) -> None:
        super().__init__()
        self.scorer = Scorer(cfg)

        # Lets the trainer skip scorer-pretraining when the scorer is already warm
        # (loaded from a checkpoint, e.g. reuse pass-2).
        self.loaded_from_checkpoint = pretrained_path is not None
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, weights_only=True, map_location="cpu")
            self.load_state_dict(state_dict)

    def score(self, cache: SampleCacheWithHidden, **kwargs: Any) -> WeightResult:
        mask = self._build_valid_mask(cache)

        hidden_states = cache.hidden_states # (B, T, D)

        # TODO: Should it be `1:` or `:-1`?
        hidden_states = hidden_states[:, 1:, :] # (B, T-1, D)
        # hidden_states = hidden_states[:, :-1, :] # (B, T-1, D)
        hidden_states = hidden_states.detach()

        if self.scorer._config.normalize_hidden:
            hidden_states = F.normalize(hidden_states, dim=-1)

        # The t'th score corresponds to the t+1'th token's importance
        weights = self.scorer(hidden_states) # (B, T-1)

        weights = self._apply_mask(weights, mask)

        return WeightResult(weights=weights, valid_mask=mask)
