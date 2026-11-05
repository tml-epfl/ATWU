import logging

import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model, get_tokenizer
from evals import get_evaluators

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."

    judge_only = cfg.get("judge_only", False)
    if judge_only:
        logger.info("judge_only=True — skipping model loading, loading tokenizer only")
        model = None
        tokenizer = get_tokenizer(model_cfg.tokenizer_args)
    else:
        model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
