"""Win rate metric comparing unlearned model against pretrained baseline."""
import logging
from typing import Any, Dict, Optional

from evals.metrics.base import unlearning_metric
from evals.metrics.paraphrase.judges.winrate import WinrateJudge

logger = logging.getLogger(__name__)


@unlearning_metric(name="winrate")
def winrate(model, **kwargs) -> Dict[str, Any]:
    """Evaluate win rate of unlearned model vs pretrained baseline using LLM judge.

    This metric compares the quality of responses from an unlearned model against
    a pretrained baseline by:
    1. Loading repetitiveness evaluation results for both models
    2. Using an LLM judge to assess which model's response is better
    3. Computing win rate (where ties count as 0.5 wins)

    Note: This metric requires that repetitiveness has been run for both the
    pretrained baseline and the unlearned model.

    Args:
        model: The model to evaluate (not directly used, serves as context).
        **kwargs: Additional arguments including:
            - gemini_api_key: Optional Gemini API key (only for gemini judge)
            - eval_task: Task identifier
            - judge: Judge configuration dict
            - output_base: Output paths configuration dict
            - model_name: Name of the model being evaluated
            - task_name: Name of the evaluation task

    Returns:
        Dict containing:
            - agg_value: Win rate (0-1, higher is better)
            - wins: Number of wins for unlearned model
            - losses: Number of losses
            - ties: Number of ties
            - total_comparisons: Total comparisons made
    """
    gemini_api_key: Optional[str] = kwargs.get("gemini_api_key")

    # Extract configuration from kwargs
    judge_cfg: Dict[str, Any] = kwargs.get("judge", {})
    output_base_cfg: Dict[str, str] = kwargs.get("output_base", {})
    model_name: str = kwargs.get("model_name", "model")
    task_name: str = kwargs.get("task_name", "model")

    # Extract judge configuration
    judge_type: str = judge_cfg.get("type", "local")
    judge_model: Optional[str] = judge_cfg.get("model")
    judge_instance: Optional[Any] = kwargs.get("judge_instance")
    seed: Optional[int] = judge_cfg.get("seed", 42)
    baseline_path: Optional[str] = kwargs.get("baseline_path")

    logger.info(f"Starting win-rate evaluation with {judge_type} judge")

    # Initialize evaluator (file existence checks happen in WinrateJudge.__init__)
    evaluator = WinrateJudge(
        model_name=model_name,
        task_name=task_name,
        output_base=output_base_cfg,
        api_key=gemini_api_key,
        judge_type=judge_type,
        judge_model=judge_model,
        judge_instance=judge_instance,
        seed=seed,
        baseline_path=baseline_path,
    )

    # Run win-rate evaluation (loads cached results if available)
    evaluator.win_rate_evaluation()

    # The win_rate_evaluation() method doesn't return values, so we need to
    # load the results from the output file
    import json
    with open(evaluator.out_file_path, 'r') as f:
        results = json.load(f)

    win_rate = results["winrate"]
    counts = results["counts"]

    logger.info(f"Win-rate evaluation complete: {win_rate:.4f}")

    return {
        "agg_value": win_rate,
        "win_rate": win_rate,
        "wins": counts["wins"],
        "losses": counts["losses"],
        "ties": counts["ties"],
        "total_comparisons": len(results["results"]),
    }
