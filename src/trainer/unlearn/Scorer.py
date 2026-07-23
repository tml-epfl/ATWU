from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import Literal

logger = logging.getLogger(__name__)

import torch
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import SGD  # type: ignore
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    SequentialLR
)
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.utils import reweighted_NLL

Model = nn.Module | PreTrainedModel

__all__ = [
    'ScorerOptimConfig',
    'ScorerTrainer',
    'ScorerTrainerATWU'
]


class ScorerOptimConfig(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
    )

    lr: float = Field(1e-2, ge=0, le=1, description="Learning Rate (in [0, 1]; 0 freezes the scorer)")
    weight_decay: float = Field(1e-4, gt=0, le=1, description="Weight Decay (in (0, 1])")
    momentum: float = Field(0.9, gt=0, le=1, description="Momentum (in (0, 1])")

    update_every_n_steps: int = Field(10, gt=0, description="Update the scorer every \"n\" model optimizer steps (> 0)")
    grad_clip: float | None = Field(1.0, gt=0, description="Clip scorer gradient (> 0). Set to None to disable.")
    loss_reduction: Literal["mean", "sum"] = Field("mean", description="Loss reduction method")

    scheduler: Literal["none", "linear", "cosine"] = "none"
    warmup_ratio: float = Field(0.0, ge=0.0, le=1.0)
    total_steps: int | None = Field(None, gt=0, description="Total scorer *update* steps (not model steps).")


class ScorerTrainer:
    def __init__(
        self,
        scorer: TokenImportanceScorer,
        optim_cfg: ScorerOptimConfig,
        max_steps: int,
        accumulation_steps: int = 1,
        backwards_every_step: bool = False,
        accelerator: Accelerator | None = None,
        embed_grad: bool = False
    ):
        self.scorer = scorer
        self.optim_cfg = optim_cfg
        self.accumulation_steps = accumulation_steps
        self.effective_batches = accumulation_steps
        if backwards_every_step:
            self.effective_batches *= optim_cfg.update_every_n_steps
        self.accelerator = accelerator
        self.backwards_every_step = backwards_every_step
        self.embed_grad = embed_grad
        self._call_count = 0

        self.total_steps = (
            (max_steps + self.optim_cfg.update_every_n_steps - 1) //
            self.optim_cfg.update_every_n_steps
        )

        self.optimizer: Optimizer = self.build_optimizer(optim_cfg)
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler: LRScheduler | None = self.build_scheduler(
            self.optimizer,
            optim_cfg,
            total_steps=self.total_steps
        )

        if self.accelerator is not None:
            # Scorer is a small MLP — skip DeepSpeed wrapping, keep fp32
            # for numerical stability. accelerator.backward() and
            # clip_grad_norm_() still work on unwrapped models.
            self.scorer = self.scorer.float().to(self.accelerator.device)

    def build_optimizer(self, cfg: ScorerOptimConfig) -> Optimizer:
        return SGD(
            self.scorer.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    def build_scheduler(
        self,
        optimizer: Optimizer,
        cfg: ScorerOptimConfig,
        *,
        total_steps: int | None = None,
    ) -> LRScheduler | None:
        if cfg.scheduler == "none":
            return None

        # Resolve total update steps (scheduler steps are scorer *updates*, not model steps)
        resolved_total = cfg.total_steps if cfg.total_steps is not None else total_steps
        if resolved_total is None:
            raise ValueError("`total_steps` must be provided (or `cfg.total_steps` set) when scheduler != 'none'.")
        if resolved_total <= 0:
            raise ValueError("total_steps must be > 0.")

        warmup = int(math.floor(cfg.warmup_ratio * resolved_total))
        warmup = max(0, min(warmup, resolved_total - 1))
        main_steps = resolved_total - warmup

        warmup_sched: LRScheduler | None = None
        if warmup > 0:
            warmup_sched = LambdaLR(
                optimizer,
                lr_lambda=lambda step: step / max(1, warmup),
            )

        if cfg.scheduler == "linear":
            main_sched: LRScheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=main_steps,
            )
        elif cfg.scheduler == "cosine":
            main_sched = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=0.0, # type: ignore # Pylance is indicating `eta_min` should be an int which is wrong.
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        if warmup_sched is not None:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup],
            )

        return main_sched

    def _backwards_step(self, *caches: Cache):
        self.scorer.requires_grad_(True)

        loss = self.compute_loss(*caches)

        # TODO: Is `retain_graph` really needed?
        if getattr(self, "accelerator", None) is not None:
            self.accelerator.backward(loss, retain_graph=True) # type: ignore
        else:
            loss.backward(retain_graph=True)

        self.scorer.requires_grad_(False)

    def _update_step(self):
        if self.optim_cfg.grad_clip is not None:
            base = self.accelerator or torch.nn.utils
            base.clip_grad_norm_(self.scorer.parameters(), self.optim_cfg.grad_clip)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.scheduler is not None:
            self.scheduler.step()

    def step(self, global_step: int, *caches: Cache):
        self._call_count += 1
        at_accum_boundary = self._call_count % self.accumulation_steps == 0

        wants_update = self.is_update_step(global_step)

        # Backward on all micro-batches during an update window (or always if backwards_every_step)
        do_backward = self.backwards_every_step or wants_update
        # Only optimizer.step() at the last micro-batch of the accumulation window
        do_update = wants_update and at_accum_boundary

        if do_backward:
            self._backwards_step(*caches)

        if do_update:
            self._update_step()

    def is_update_step(self, global_step: int):
        return (global_step + 1) % self.optim_cfg.update_every_n_steps == 0

    def _log_scores(self, scores: torch.Tensor, mask: torch.Tensor, prefix: str = ""):
        valid = scores[mask]
        selected_mask = (scores > 0.5) & mask  # (B, T)
        total_counts = mask.sum(dim=1).float().clamp(min=1)  # (B,)

        means, stds = [], []
        for i in range(scores.size(0)):
            sel = scores[i][mask[i]]
            if sel.numel() > 0:
                means.append(sel.mean())
                stds.append(sel.std() if sel.numel() > 1 else torch.tensor(0.0, device=sel.device))

        mean_val = torch.stack(means).mean().item() if means else 0.0
        std_val = torch.stack(stds).mean().item() if stds else 0.0
        pct_selected = (selected_mask.sum(dim=1).float() / total_counts).mean().item() * 100

        logger.info(
            "%sscores — min=%.4f max=%.4f mean=%.4f std=%.4f selected=%5.2f%%",
            prefix,
            valid.min().item(), valid.max().item(),
            mean_val, std_val, pct_selected,
        )

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class ScorerTrainerATWU(ScorerTrainer):
    """Scorer trainer for ATWU.

    Forget objective: -g * sat(g*nll) * nll via reweighted_NLL(beta), plus
    entropy, population-budget and L2 penalties on the scores. No retain term.
    """

    def __init__(
        self,
        lambda_entropy: float = 1.0,
        lambda_population: float = 1.0,
        budget: float = 0.3,
        lambda_l2: float = 0.01,
        beta: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.lambda_population = lambda_population
        self.budget = budget
        self.lambda_l2 = lambda_l2
        self.beta = beta

    def _entropy_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-6,
    ):
        scores = scores.clone()[mask]
        g = scores.clamp(epsilon, 1 - epsilon)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()

    def _population_loss(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        budget: float = 0.2,
    ):
        hard = (scores > 0.5).float()
        selected = hard.detach() - scores.detach() + scores
        selected = selected * mask.float()
        seq_counts = mask.sum(dim=1).clamp(min=1)
        seq_means = selected.sum(dim=1) / seq_counts
        return ((seq_means - budget) ** 2).mean()

    def _l2_loss(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.scorer.parameters())  # type: ignore

    def compute_loss(
        self,
        forget_cache: Cache,
        retain_cache: Cache,
    ):
        forget_loss, f_scores, f_mask = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            scorer_requires_grad=True,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        forget_entropy = self._entropy_loss(f_scores, f_mask)
        forget_population = self._population_loss(f_scores, f_mask, budget=self.budget)
        l2 = self._l2_loss() # explicit weight decay

        loss = (
            forget_loss
            + self.lambda_entropy * forget_entropy
            + self.lambda_population * forget_population
            + self.lambda_l2 * l2
        )

        logger.info(
            "forget=%.3f f_entropy=%.3f f_budget=%.3f l2=%.3f total=%.3f",
            forget_loss.item(), forget_entropy.item(), forget_population.item(),
            l2.item(), loss.item(),
        )

        self._log_scores(f_scores, f_mask, prefix="forget ")

        if self.optim_cfg.loss_reduction == "mean":
            loss = loss / self.effective_batches

        return loss
