"""Paraphrase Evaluator."""
import logging

import torch

from evals.base import Evaluator
from evals.metrics.paraphrase.judges import QualityJudge

logger = logging.getLogger(__name__)

# Metrics that need an LLM judge. If none of them are active (e.g. a
# repetitiveness+mmlu-only run), no judge is created and no API key is needed.
_JUDGE_METRICS = {"forget_quality", "retain_quality", "winrate"}


class ParaphraseEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("Paraphrase", eval_cfg, **kwargs)

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        """Override to create a single shared judge instance for all Paraphrase metrics.

        The judge model (e.g. Llama-3.3-70B-Instruct quantised to 4-bit) is
        expensive to load.  Rather than letting each metric (forget_quality,
        retain_quality) instantiate its own copy, we build it once here, inject
        it into the kwargs passed to each metric, and clean up at the end.
        """
        # --- Create shared judge instance (only if a judge-based metric runs) ---
        judge_cfg = getattr(self.eval_cfg, "judge", None)
        judge_instance = None

        if judge_cfg is not None and any(m in _JUDGE_METRICS for m in self.metrics):
            judge_instance = QualityJudge.create_judge_instance(dict(judge_cfg))
            if judge_instance is not None:
                logger.info("Shared judge instance created — will be reused across all paraphrase metrics")

        try:
            # --- Evaluate loop (mirrors Evaluator.evaluate, adds judge_instance) ---
            overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite
            if model is not None:
                model = self.prepare_model(model)

            output_dir = output_dir if output_dir else self.eval_cfg.output_dir
            logs_file_path = self.get_logs_file_path(output_dir)
            summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

            logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

            logger.info(f"***** Running {self.name} evaluation suite *****")
            logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
            logger.info(f"Aggregated evaluations will be summarised in: {summary_file_path}")

            for metric_name, metric_fn in self.metrics.items():
                if not overwrite and metric_name in logs and logs[metric_name] and not logs[metric_name].get("generation_only"):
                    logger.info(f"Skipping {metric_name}, already evaluated.")
                    if "agg_value" in logs[metric_name]:
                        logger.info(
                            f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                        )
                    self.save_logs(self.summarize(logs), summary_file_path)
                    continue

                _ = logs.pop(metric_name, None)

                metric_kwargs = {
                    "tokenizer": kwargs.get("tokenizer", None),
                    "template_args": kwargs.get("template_args", None),
                }
                # Inject shared judge instance so metrics don't load their own
                if judge_instance is not None:
                    metric_kwargs["judge_instance"] = judge_instance

                metrics_args = self.eval_cfg.metrics[metric_name]

                result = metric_fn(
                    model,
                    metric_name=metric_name,
                    cache=logs,
                    **metric_kwargs,
                    **metrics_args,
                )
                if result.get("generation_only"):
                    logger.info(f"Generation complete for {metric_name} (judging deferred)")
                    continue  # Don't persist — allows re-run for judging

                if "agg_value" in result:
                    logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
                self.save_logs(logs, logs_file_path)
                self.save_logs(self.summarize(logs), summary_file_path)

            return self.summarize(logs)

        finally:
            if judge_instance is not None:
                logger.info("Cleaning up shared judge instance")
                del judge_instance
                torch.cuda.empty_cache()