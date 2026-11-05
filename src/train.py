from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # Instantiate any _target_ nodes (e.g. ATWU/ETW scorer, scorer_trainer
    # partial) so trainers receive objects, not DictConfigs.
    cfg = instantiate(cfg)
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    # Scorer-only mode: run the ATWU scorer pretrain loop (model frozen, no
    # unlearning) and save just the scorer. Used to fit a scorer on a subset of
    # the forget data — e.g. the k-fold scorer-transfer experiment — without
    # touching the model weights.
    if cfg.get("scorer_only", False):
        if not hasattr(trainer, "_pretrain_scorer"):
            raise ValueError(
                "scorer_only=true requires a scorer-based trainer (e.g. trainer=ATWU), "
                f"got {type(trainer).__name__}"
            )
        pretrain_cfg = trainer.scorer_pretrain_cfg
        if not ((pretrain_cfg.steps or 0) > 0 or pretrain_cfg.epochs > 0):
            raise ValueError(
                "scorer_only=true needs trainer.method_args.scorer_pretrain.steps>0 "
                "(or .epochs>0); otherwise nothing would be trained."
            )
        trainer._pretrain_scorer()
        out_path = Path(trainer_args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.scorer.state_dict(), out_path / "scorer.pt")
        return

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
