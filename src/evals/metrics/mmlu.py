"""MMLU Metric using lm-evaluation-harness."""
import logging
from typing import Any, Dict

from lm_eval.models.hf_vlms import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate

from evals.metrics.base import unlearning_metric


logger = logging.getLogger(__name__)


@unlearning_metric(name="mmlu")
def mmlu(model, **kwargs) -> Dict[str, Any]:
    """Evaluate model on MMLU benchmark using lm-evaluation-harness.

    This metric wraps the lm-evaluation-harness MMLU evaluation to integrate
    with the paraphrase eval suite, allowing MMLU results to be saved
    alongside the other metrics in the same summary file.

    Args:
        model: The model to evaluate.
        **kwargs: Additional arguments including:
            - tokenizer: Model tokenizer
            - batch_size: Batch size for evaluation (default: 16)
            - system_instruction: Optional system instruction
            - apply_chat_template: Whether to apply chat template (default: False)

    Returns:
        Dict containing:
            - agg_value: MMLU accuracy (0-1, higher is better)
            - mmlu/acc: MMLU accuracy
            - Additional per-task metrics if available
    """
    tokenizer = kwargs.get("tokenizer")
    batch_size = kwargs.get("batch_size", 16)
    system_instruction = kwargs.get("system_instruction")
    apply_chat_template = kwargs.get("apply_chat_template", False)

    logger.info("Running MMLU evaluation via lm-evaluation-harness")

    # Prepare model for lm-eval
    model.eval()
    lm_eval_model = HFLM(model, tokenizer=tokenizer)

    # Run evaluation
    task_manager = TaskManager()
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=["mmlu"],
        task_manager=task_manager,
        batch_size=batch_size,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
    )

    # Extract MMLU accuracy from group results
    mmlu_metrics = results.get("groups", {}).get("mmlu", {})

    # Get accuracy, removing ',none' suffix if present
    acc_key = next((k for k in mmlu_metrics.keys() if k.startswith("acc")), None)
    mmlu_acc = mmlu_metrics.get(acc_key, 0.0) if acc_key else 0.0

    logger.info(f"MMLU accuracy: {mmlu_acc:.4f}")

    # Return in format expected by the paraphrase evaluator
    result = {
        "agg_value": mmlu_acc,
        "mmlu/acc": mmlu_acc,
    }

    # Include any additional metrics
    for metric_name, value in mmlu_metrics.items():
        if metric_name == "alias":
            continue
        clean_name = metric_name.split(",", 1)[0].strip()
        if clean_name != "acc":  # Already added above
            result[f"mmlu/{clean_name}"] = value

    return result
