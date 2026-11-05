from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import SGD  # type: ignore
from transformers import PreTrainedModel, TrainerCallback

from data.Cache import Cache
from scorers.base import TokenImportanceScorer
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import reweighted_NLL

logger = logging.getLogger(__name__)

Model = PreTrainedModel


@dataclass
class ScorerPretrainConfig:
    epochs: int = 0
    steps: int | None = None  # if set (>0), pretrain for this many optimizer steps and
                              # take precedence over `epochs` (useful when 1 epoch over a
                              # large forget set is too much warm-up)
    lr: float = 0.05
    freeze_after: bool = False
    scorer_save_every_steps: int | None = None


class ScorerCheckpointCallback(TrainerCallback):
    def __init__(self, scorer: nn.Module, save_every_steps: int):
        self.scorer = scorer
        self.save_every_steps = save_every_steps

    def _save(self, output_dir: str, tag: str):
        out = Path(output_dir) / "scorer_checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), out / f"step_{tag}.pt")

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if state.global_step > 0 and state.global_step % self.save_every_steps == 0:
            self._save(args.output_dir, str(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._save(args.output_dir, "final")


class BaseATWU(UnlearnTrainer):
    def __init__(
        self,
        scorer: TokenImportanceScorer,
        scorer_trainer: Callable[..., Any],
        scorer_pretrain: ScorerPretrainConfig | None = None,
        canonical_ce: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.scorer = scorer
        self.scorer_pretrain_cfg = scorer_pretrain or ScorerPretrainConfig()
        self._freeze_scorer = False
        # When True, the per-token CE in the cache is computed in HF's canonical (N,V)
        # layout (reproducible) instead of the legacy strided (B,V,T-1) layout. Off by
        # default to preserve existing TOFU numerics. Threads into BOTH the unlearning
        # forget loss and the scorer-update forget loss because both read the same
        # cache.token_loss built in prepare_caches.
        self.canonical_ce = canonical_ce

        self.scorer_trainer = scorer_trainer(
            scorer=self.scorer,
            accumulation_steps=getattr(self.args, "gradient_accumulation_steps", 1),
            max_steps=self.compute_max_steps(),
            accelerator=self.accelerator
        )

        save_every = self.scorer_pretrain_cfg.scorer_save_every_steps
        if save_every is not None and self.accelerator.is_main_process:
            out = Path(self.args.output_dir) / "scorer_checkpoints"
            out.mkdir(parents=True, exist_ok=True)
            torch.save(self.scorer.state_dict(), out / "step_0.pt")
            self.add_callback(ScorerCheckpointCallback(self.scorer, save_every))

    def compute_max_steps(self):
        train_loader = self.get_train_dataloader()

        num_update_steps_per_epoch = len(train_loader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

        return math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        output_dir_path = Path(output_dir or self.args.output_dir)  # type: ignore
        output_dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.scorer.state_dict(), output_dir_path / "scorer.pt")

    def pack_inputs(self, inputs: dict[str, Any]):
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }

    def prepare_caches(self, model: Model, *inputs: dict[str, Any]) -> tuple[Cache, ...]:
        return tuple(
            self.scorer.cache(model, _inputs, canonical_ce=self.canonical_ce)
            for _inputs in inputs
        )

    def train(self, **kwargs): # type: ignore
        cfg = self.scorer_pretrain_cfg
        want_pretrain = (cfg.steps or 0) > 0 or cfg.epochs > 0
        if want_pretrain and getattr(self.scorer, "loaded_from_checkpoint", False):
            logger.info(
                "Skipping scorer pretrain: scorer was loaded from a checkpoint "
                "(scorer.pretrained_path set)."
            )
        elif want_pretrain:
            self._pretrain_scorer()
        if cfg.freeze_after:
            self.scorer.requires_grad_(False)
            self._freeze_scorer = True
        return super().train(**kwargs)

    @torch.no_grad()
    def _pretrain_scorer(self):
        cfg = self.scorer_pretrain_cfg
        use_steps = cfg.steps is not None and cfg.steps > 0
        if use_steps:
            logger.info(
                "Pretraining scorer for %d step(s) (lr=%s, freeze_after=%s)",
                cfg.steps, cfg.lr, cfg.freeze_after,
            )
        else:
            logger.info(
                "Pretraining scorer for %d epoch(s) (lr=%s, freeze_after=%s)",
                cfg.epochs, cfg.lr, cfg.freeze_after,
            )

        # Freeze model — no model gradients needed
        was_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        pretrain_opt = SGD(
            self.scorer.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        dataloader = self.get_train_dataloader()

        def _scorer_step(inputs):
            inputs = self._prepare_inputs(inputs)
            forget_inputs = self.pack_inputs(inputs["forget"]) # type: ignore
            retain_inputs = self.pack_inputs(inputs["retain"]) # type: ignore

            forget_cache, retain_cache = self.prepare_caches(
                self.model, forget_inputs, retain_inputs # type: ignore
            )

            # Enable grad only for scorer params
            with torch.enable_grad():
                self.scorer.requires_grad_(True)
                loss = self.scorer_trainer.compute_loss(forget_cache, retain_cache)
                loss.backward()
                self.scorer.requires_grad_(False)

            if self.scorer_trainer.optim_cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.scorer.parameters(),
                    self.scorer_trainer.optim_cfg.grad_clip,
                )
            pretrain_opt.step()
            pretrain_opt.zero_grad(set_to_none=True)

        if use_steps:
            # Train exactly cfg.steps optimizer steps, cycling the loader if needed.
            step = 0
            while step < cfg.steps:
                for inputs in dataloader:
                    _scorer_step(inputs)
                    step += 1
                    if step >= cfg.steps:
                        break
                else:
                    continue  # loader exhausted before target → iterate again
                break
            logger.info("Scorer pretrain complete: %d step(s)", step)
        else:
            for epoch in range(cfg.epochs):
                for inputs in dataloader:
                    _scorer_step(inputs)
                logger.info("Scorer pretrain epoch %d/%d complete", epoch + 1, cfg.epochs)

        # Restore model
        for p in self.model.parameters():
            p.requires_grad_(True)
        if was_training:
            self.model.train()


class ATWU(BaseATWU):
    def __init__(
        self, *,
        alpha: float,
        gamma: float,
        beta: float = 5.0,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs = self.pack_inputs(inputs["forget"])
        retain_inputs = self.pack_inputs(inputs["retain"])

        forget_cache, retain_cache = self.prepare_caches(model, forget_inputs, retain_inputs)

        if not self._freeze_scorer:
            self.scorer_trainer.step(self.state.global_step, forget_cache, retain_cache)

        forget_loss, _, _ = reweighted_NLL(
            cache=forget_cache,
            scorer=self.scorer,
            beta=self.beta,
        )
        forget_loss = -forget_loss

        retain_loss: torch.Tensor = retain_cache.outputs.loss # type: ignore
        loss = self.alpha * retain_loss + self.gamma * forget_loss

        return (loss, forget_cache.outputs) if return_outputs else loss
