"""Repetitiveness metric using n-gram entropy."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Optional dependency: keep the core repo importable without it.
# Install via `pip install nltk` (or `pip install .[paraphrase-judge]`).
try:
    import nltk
except ImportError:
    nltk = None
import numpy as np
import numpy.typing as npt
import scipy.stats.mstats
import torch
from tqdm import tqdm

from data.utils import preprocess_chat_instance
from evals.metrics.base import unlearning_metric

logger = logging.getLogger(__name__)


def n_gram_entropy(
    gen_texts: List[str],
    agg: Literal["arith", "geom"] = "arith"
) -> float:
    """Compute average n-gram entropy across generated texts.

    Args:
        gen_texts: List of generated text strings.
        agg: Aggregation method - "arith" for arithmetic mean, "geom" for geometric mean.

    Returns:
        The aggregated n-gram entropy value.
    """
    assert agg in ["arith", "geom"], f"agg must be 'arith' or 'geom', got {agg}"

    entropies = [compute_n_gram_entropy(txt) for txt in gen_texts]
    result = (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropies)
    return float(result)


def compute_n_gram_entropy(
    sentence: str,
    ns: Optional[List[int]] = None,
    weights: Optional[List[float]] = None,
    agg: Literal["arith", "geom"] = "arith"
) -> float:
    """Compute weighted n-gram entropy for a single sentence.

    Args:
        sentence: The input sentence.
        ns: List of n-gram orders to compute. Defaults to [2, 3].
        weights: Weights for each n-gram order. Defaults to [2/3, 4/3].
        agg: Aggregation method for combining n-gram entropies.

    Returns:
        The weighted n-gram entropy value.
    """
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"], f"agg must be 'arith' or 'geom', got {agg}"

    entropy_list: List[float] = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs: npt.NDArray[np.float64] = np.array([freq for _, freq in fdist.items()])
        if len(freqs) == 0:
            entropy_list.append(0.0)
            continue
        freqs = freqs / freqs.sum()
        entropy = np.sum(-freqs * np.log(freqs) / np.log(2))
        entropy_list.append(float(entropy))

    weighted_entropies: npt.NDArray[np.float64] = np.array(entropy_list) * np.array(weights)
    result = (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(weighted_entropies)
    return float(result)


def compute_freq(sentence: str, n: int = 2) -> nltk.FreqDist:
    """Compute frequency distribution of n-grams in a sentence.

    Args:
        sentence: The input sentence.
        n: The n-gram order.

    Returns:
        Frequency distribution of n-grams.
    """
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


@unlearning_metric(name="repetitiveness")
def repetitiveness(model, **kwargs) -> Dict[str, Any]:
    """Evaluate text repetitiveness using n-gram entropy.

    This metric measures how repetitive the model's generations are by computing
    n-gram entropy. Higher entropy indicates more diverse (less repetitive) text.

    Args:
        model: The language model to evaluate.
        **kwargs: Additional arguments including:
            - tokenizer: Model tokenizer
            - data: Dataset to evaluate on
            - batch_size: Batch size for generation (default: 1)
            - num_samples: Maximum samples to evaluate (default: 1000)
            - output_dir: Optional directory to save results

    Returns:
        Dict containing:
            - agg_value: Scaled entropy score (entropy * 100)
            - entropy: Raw n-gram entropy value
            - num_samples: Number of samples evaluated
    """
    if model is None:
        raise RuntimeError(
            "repetitiveness requires a model for generation. In judge_only mode, "
            "results must already be cached from the generation phase."
        )
    if nltk is None:
        raise ImportError(
            "The `nltk` package is required for the repetitiveness metric. "
            "Install it with `pip install nltk` or `pip install .[paraphrase-judge]`."
        )
    tokenizer = kwargs["tokenizer"]
    template_args = kwargs["template_args"]
    # Allow metric-level system prompt override (e.g. to use original prompt
    # when the global one was overridden for training/other evals)
    system_prompt_override = kwargs.get("system_prompt")
    if system_prompt_override is not None:
        from copy import deepcopy
        template_args = deepcopy(template_args)
        template_args["system_prompt"] = system_prompt_override
    data = kwargs["data"]
    batch_size: int = kwargs.get("batch_size", 1)
    num_samples: int = kwargs.get("num_samples", 1000)
    max_length: int = kwargs.get("max_length", 512)
    output_dir: Optional[str] = kwargs.get("output_dir")
    task_name: str = kwargs.get("task_name", "model")
    generation_cfg: Dict[str, Any] = kwargs.get("generation", {})
    max_new_tokens: int = generation_cfg.get("max_new_tokens", 128)

    # Download NLTK punkt tokenizer if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    # Tokenize prompts using the model's template_args
    all_input_ids = []
    all_attention_masks = []
    questions: List[Dict[str, Any]] = []

    for ix, sample in enumerate(data):
        if ix >= num_samples:
            break
        question_text = sample.get('instruction', sample.get('question', ''))
        tokenized = preprocess_chat_instance(
            tokenizer, template_args, [question_text], [""],
            max_length, predict_with_generate=True,
        )
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        questions.append(sample)

    # Set up partial checkpoint for preemption resilience
    partial_path: Optional[Path] = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        partial_path = output_path / "repetitiveness_partial.jsonl"

    # Resume from partial checkpoint if it exists
    outputs: List[str] = []
    resumed_questions: List[Dict[str, Any]] = []
    start_idx = 0
    if partial_path and partial_path.exists():
        with open(partial_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    outputs.append(entry["prediction"])
                    resumed_questions.append({"instruction": entry["instruction"]})
        start_idx = len(outputs)
        logger.info(f"Resuming repetitiveness from {start_idx} saved samples")

    # Generate completions in batches with left-padding
    logger.info(f"Generating {len(all_input_ids) - start_idx} remaining completions for repetitiveness evaluation")
    for i in tqdm(range(start_idx, len(all_input_ids), batch_size), desc=f"Repetitiveness (Batch Size: {batch_size})"):
        batch_ids = all_input_ids[i:i + batch_size]
        batch_masks = all_attention_masks[i:i + batch_size]

        max_len = max(ids.shape[0] for ids in batch_ids)
        pad_id = tokenizer.pad_token_id
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(batch_ids, batch_masks):
            pad_len = max_len - ids.shape[0]
            padded_ids.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype), ids])
            )
            padded_masks.append(
                torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            )

        batch_input = torch.stack(padded_ids).to(model.device)
        batch_mask = torch.stack(padded_masks).to(model.device)

        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=batch_input,
                attention_mask=batch_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Save each sample incrementally
        batch_entries = []
        for j in range(len(batch_ids)):
            generated = gen_outputs[j, max_len:]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            outputs.append(response)
            q = questions[i + j]
            batch_entries.append({
                "instruction": q.get("instruction", q.get("question", "")),
                "prediction": response,
            })

        if partial_path:
            with open(partial_path, "a") as f:
                for entry in batch_entries:
                    f.write(json.dumps(entry) + "\n")

    # Compute entropy
    entropy = n_gram_entropy(outputs)
    scaled_entropy = entropy * 100

    logger.info(f"Repetitiveness entropy: {scaled_entropy:.4f}")

    # Save final results and clean up partial checkpoint
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare results (only keep instruction + prediction)
        # For resumed samples, use the resumed questions; for new ones, use original questions
        all_questions = resumed_questions + questions[start_idx:]
        results = [
            {
                "instruction": q.get("instruction", q.get("question", "")),
                "prediction": answer,
            }
            for answer, q in zip(outputs, all_questions)
        ]

        output_result = {
            'entropy': scaled_entropy,
            'num_samples': len(outputs),
            'results': results,
        }

        # Save as repetitiveness_results.json (standard format)
        result_file = output_path / "repetitiveness_results.json"
        with open(result_file, 'w') as f:
            json.dump(output_result, f, indent=4)
        logger.info(f"Saved repetitiveness results to: {result_file}")

        # Also save in winrate-compatible format: {task_name}.jsonl
        rep_file = output_path / f"{task_name}.jsonl"
        with open(rep_file, 'w') as f:
            json.dump(output_result, f, indent=4)
        logger.info(f"Saved winrate-compatible results to: {rep_file}")

        # Clean up partial checkpoint
        if partial_path and partial_path.exists():
            partial_path.unlink()
            logger.info("Cleaned up partial checkpoint")

    return {
        "agg_value": scaled_entropy,
        "entropy": entropy,
        "num_samples": len(outputs),
    }
