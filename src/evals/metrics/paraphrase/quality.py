"""Forget and retain quality metrics using LLM judge.

Shared ``_evaluate_quality()`` core parameterized on task type, ICR variants,
and aggregation strategy.
"""
import logging
from typing import Any, Dict, List, Optional, Type, Union

import torch
from datasets import load_dataset
from tqdm import tqdm

from data.utils import preprocess_chat_instance
from evals.metrics.base import unlearning_metric
from evals.metrics.paraphrase.aggregation import AvgEval, WorstEval
from evals.metrics.paraphrase.generation import Evaluator
from evals.metrics.paraphrase.judges.quality import QualityJudge
from evals.metrics.paraphrase.utils import get_api_key

logger = logging.getLogger(__name__)

ICR_NUM_EXAMPLES = 3


def _generate_responses(
    model,
    tokenizer,
    data,
    max_samples: int,
    generation_cfg: Dict[str, Any],
    icr: bool,
    icr_dataset=None,
    system_prompt: Optional[str] = None,
    start_idx: int = 0,
    evaluator: Optional["Evaluator"] = None,
) -> List[Dict[str, str]]:
    """Generate model responses for all question variants in the dataset.

    Uses pre-tokenized input_ids from ParaphraseQADataset (eval mode) for the
    standard case, and re-tokenizes with ICR in-context examples when icr=True.
    All question variants for a sample are batched into a single generate call.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer.
        data: ParaphraseQADataset in eval mode.
        max_samples: Maximum number of samples to evaluate.
        generation_cfg: Generation configuration dict.
        icr: Whether to use in-context retention examples.
        icr_dataset: Loaded ICR dataset (required when icr=True).
        system_prompt: Optional system prompt to prepend to model input.
            For chat models (apply_chat_template=True), injected as system role.
            For base models, prepended as a plain text prefix.

    Returns:
        List of dicts, one per sample, mapping ``ans_<qid>`` to generated response.
    """
    max_new_tokens = generation_cfg.get("max_new_tokens", 128)
    do_sample = generation_cfg.get("do_sample", False)
    temperature = generation_cfg.get("temperature", 0.0)

    # Build template_args with optional system prompt override
    template_args = data.template_args
    if system_prompt:
        from omegaconf import OmegaConf
        template_args = OmegaConf.to_container(template_args, resolve=True)
        if template_args.get("apply_chat_template"):
            template_args["system_prompt"] = system_prompt
        else:
            template_args["system_prompt_with_special_tokens"] = (
                system_prompt + "\n"
            )

    num_samples = min(len(data), max_samples)
    logs: List[Dict[str, str]] = []

    if start_idx > 0:
        logger.info(f"Resuming generation from sample {start_idx}/{num_samples}")

    for idx in tqdm(range(start_idx, num_samples), desc=f"Generating (ICR={icr})"):
        sample = data[idx]

        # Include ground truth so the generation file is self-contained
        # and the judge doesn't need to reload the dataset.
        result_dict: Dict[str, str] = {"GT": sample["answer"]}

        # Collect all question variants for batched generation
        qids = []
        all_input_ids = []
        all_attention_masks = []

        for qid, item in sample.items():
            if qid in ("answer", "index"):
                continue

            result_dict[qid] = item["question_text"]
            qids.append(qid)

            if icr and icr_dataset is not None:
                icr_examples = icr_dataset.shuffle(seed=idx).select(range(ICR_NUM_EXAMPLES))
                prompt_msgs = [ex["question"] for ex in icr_examples] + [item["question_text"]]
                response_msgs = [ex["answer"] for ex in icr_examples] + [""]
                tokenized = preprocess_chat_instance(
                    tokenizer,
                    template_args,
                    prompt_msgs,
                    response_msgs,
                    data.max_length,
                    predict_with_generate=True,
                )
                all_input_ids.append(tokenized["input_ids"])
                all_attention_masks.append(tokenized["attention_mask"])
            elif system_prompt:
                # Re-tokenize with system prompt since pre-tokenized
                # input_ids don't include it
                tokenized = preprocess_chat_instance(
                    tokenizer,
                    template_args,
                    [item["question_text"]],
                    [""],
                    data.max_length,
                    predict_with_generate=True,
                )
                all_input_ids.append(tokenized["input_ids"])
                all_attention_masks.append(tokenized["attention_mask"])
            else:
                all_input_ids.append(item["input_ids"])
                all_attention_masks.append(item["attention_mask"])

        # Left-pad to the longest sequence in this batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = tokenizer.pad_token_id
        padded_input_ids = []
        padded_attention_masks = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            )
            padded_attention_masks.append(
                torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            )

        batch_input_ids = torch.stack(padded_input_ids).to(model.device)
        batch_attention_mask = torch.stack(padded_attention_masks).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens for each variant
        for i, qid in enumerate(qids):
            generated_ids = outputs[i, max_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            result_dict[f"ans_{qid}"] = response

        # Save incrementally if evaluator is provided
        if evaluator is not None:
            evaluator.append_log(result_dict)
        logs.append(result_dict)

    return logs


def _evaluate_quality(
    model,
    task: str,
    icr_variants: List[bool],
    aggregator_cls: Type[Union[WorstEval, AvgEval]],
    **kwargs,
) -> Dict[str, Any]:
    """Shared logic for forget/retain quality evaluation.

    Args:
        model: The model to evaluate.
        task: Either 'forget' or 'retain'.
        icr_variants: List of ICR flags to iterate over.
            Forget uses [False, True]; retain uses [False].
        aggregator_cls: WorstEval for forget, AvgEval for retain.
        **kwargs: Metric kwargs from the framework.

    Returns:
        Dict with agg_value and task-specific metrics.
    """
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    gemini_api_key: Optional[str] = kwargs.get("gemini_api_key")
    max_samples: int = kwargs.get("max_samples", 100 if task == 'forget' else 400)

    # Required configuration
    eval_task: str = kwargs["eval_task"]
    generation_only: bool = kwargs.get("generation_only", False)
    judge_cfg: Dict[str, Any] = kwargs.get("judge", {})
    generation_cfg: Dict[str, Any] = kwargs.get("generation", {})
    output_base_cfg: Dict[str, str] = kwargs.get("output_base", {})
    system_prompt: Optional[str] = kwargs.get("system_prompt")

    # Judge configuration
    judge_type: str = judge_cfg.get("type", "local")
    judge_model: Optional[str] = judge_cfg.get("model")
    judge_quantize: bool = judge_cfg.get("quantize", False)
    chunk_size: Optional[int] = judge_cfg.get("chunk_size")
    # TODO: Remove
    fix_qwen_keys: bool = judge_cfg.get("fix_qwen_keys", True)
    substring_heuristic: bool = judge_cfg.get("substring_heuristic", True)
    seed: Optional[int] = judge_cfg.get("seed", 42)
    prompt_file: Optional[str] = judge_cfg.get("prompt_file")

    # Shared judge instance (created once by ParaphraseEvaluator)
    judge_instance: Optional[Any] = kwargs.get("judge_instance")

    # Validate API key only if using Gemini
    api_key: Optional[str] = None
    if judge_type == "gemini":
        api_key = get_api_key(gemini_api_key)

    task_name: str = kwargs.get("task_name", f"{task}_eval")

    # Extract question keys from the ParaphraseQADataset
    sample = data[0]
    questions = [k for k in sample if k not in ("answer", "index")]

    # Load ICR dataset from config (only needed for forget task with ICR=True)
    icr_cfg: Dict[str, Any] = kwargs.get("icr", {})
    icr_dataset = None
    if True in icr_variants:
        icr_hf_args = icr_cfg.get("hf_args", {})
        if not icr_hf_args:
            raise ValueError(
                "ICR is enabled but no icr.hf_args config provided. "
                "Add an 'icr' section with 'hf_args' (path, name, split) "
                "to the forget_quality metric config."
            )
        logger.info(f"Loading ICR dataset: {icr_hf_args}")
        icr_dataset = load_dataset(**icr_hf_args)

    # Run evaluations for each ICR variant
    judge_files = []
    for icr in icr_variants:
        evaluator = Evaluator(
            eval_task=eval_task,
            output_base=output_base_cfg,
            generation=generation_cfg,
            icr=icr,
            task=task,
        )
        gen_file = evaluator.out_path
        judge_file = evaluator.jg_file_path

        # Check if already evaluated
        if judge_file.exists():
            logger.info(f"Using cached judge results: {judge_file}")
            judge_files.append(judge_file)
            continue

        # Run generation if needed (with resume support)
        if model is None:
            # Judge-only mode: generation files must already exist
            if not gen_file.exists():
                raise FileNotFoundError(
                    f"judge_only=True but generation file not found: {gen_file}. "
                    "Run the generation phase first."
                )
            logger.info(f"judge_only mode — using existing generation: {gen_file}")
        else:
            num_samples = min(len(data), max_samples)
            start_idx = evaluator.count_saved()
            if start_idx < num_samples:
                logger.info(f"Generating responses for {task} set with ICR={icr}")
                _generate_responses(
                    model=model,
                    tokenizer=tokenizer,
                    data=data,
                    max_samples=max_samples,
                    generation_cfg=generation_cfg,
                    icr=icr,
                    icr_dataset=icr_dataset,
                    system_prompt=system_prompt,
                    start_idx=start_idx,
                    evaluator=evaluator,
                )
            else:
                logger.info(f"Generation already complete ({start_idx} samples): {gen_file}")

        # Skip judging if generation_only mode
        if generation_only:
            logger.info(f"generation_only=True — skipping judge for {task} ICR={icr}")
            continue

        # Run judge evaluation
        if gen_file.exists():
            logger.info(f"Running {judge_type} judge for {task} set with ICR={icr}")
            judge = QualityJudge(
                eval_task=eval_task,
                output_base=output_base_cfg,
                questions=questions,
                api_key=api_key,
                task=task,
                gen_file=gen_file,
                icr=icr,
                chunk_size=chunk_size,
                judge_type=judge_type,
                judge_model=judge_model,
                judge_quantize=judge_quantize,
                fix_qwen_keys=fix_qwen_keys,
                substring_heuristic=substring_heuristic,
                judge_instance=judge_instance,
                seed=seed,
                prompt_file=prompt_file,
            )
            judge.generate()
            judge_files.append(judge.jg_file_path)
        else:
            raise FileNotFoundError(f"Generation file not found: {gen_file}")

    # In generation_only mode, return sentinel (no judging was done)
    if generation_only:
        return {"agg_value": None, "generation_only": True}

    # Compute aggregated metrics
    logger.info(f"Computing {task} metrics with {aggregator_cls.__name__}")

    aggregator_kwargs: Dict[str, Any] = {
        "run_name": task_name,
        "files": judge_files,
        "task": task,
    }
    if aggregator_cls is AvgEval:
        aggregator_kwargs["max_samples"] = max_samples

    aggregator = aggregator_cls(**aggregator_kwargs)
    results = aggregator.evaluate()

    # Build return dict based on task type
    if task == 'forget':
        return {
            "agg_value": results["J_W"],
            "J_P": results["J_P"],
            "J_ICR": results["J_ICR"],
            "J_W": results["J_W"],
        }
    else:  # retain
        return {
            "agg_value": results["J_avg"],
            "J_avg": results["J_avg"],
        }


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

@unlearning_metric(name="forget_quality")
def forget_quality(model, **kwargs) -> Dict[str, Any]:
    """Evaluate forget quality with an LLM judge on held-out paraphrases.

    Generates responses with and without in-context retention (ICR),
    then computes worst-case metrics: J_P, J_ICR, J_W.

    Returns:
        Dict with agg_value (J_W, lower is better), J_P, J_ICR, J_W.
    """
    return _evaluate_quality(
        model,
        task='forget',
        icr_variants=[False, True],
        aggregator_cls=WorstEval,
        **kwargs,
    )


@unlearning_metric(name="retain_quality")
def retain_quality(model, **kwargs) -> Dict[str, Any]:
    """Evaluate retain quality with an LLM judge on held-out paraphrases.

    Generates responses without ICR and computes average-case metric J_avg.

    Returns:
        Dict with agg_value (J_avg, higher is better), J_avg.
    """
    return _evaluate_quality(
        model,
        task='retain',
        icr_variants=[False],
        aggregator_cls=AvgEval,
        **kwargs,
    )
