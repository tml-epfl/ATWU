"""Aggregation logic for paraphrase judge evaluations.

Combines worst-case (WorstEval) and average-case (AvgEval) evaluators
with the underlying math helpers (compute_any_one_average, check_yes_no,
average_case_acc).
"""
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import numpy.typing as npt

from evals.metrics.paraphrase.utils import load_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def compute_any_one_average(
    data: Dict[str, List[int]],
    prefix: str = "outs_"
) -> Tuple[List[float], List[float]]:
    """Compute coverage and row-wise means for binary data.

    Args:
        data: Dictionary mapping keys to lists of binary values (0 or 1).
        prefix: String prefix to filter relevant keys.

    Returns:
        Tuple of (accuracies, row_means) where:
            - accuracies: List of cumulative coverage after each row
            - row_means: List of mean values for each row
    """
    relevant_keys = list(
        filter(
            lambda key: key.startswith(prefix),
            data.keys()
        )
    )
    filtered_data = [data[key] for key in relevant_keys]
    N = len(filtered_data[0])
    coverage = [0] * N
    accuracies: List[float] = []
    row_means = [sum(row) / len(row) for row in filtered_data]

    for row in filtered_data:
        for i in range(N):
            if row[i] == 1:
                coverage[i] = 1
        accuracy = sum(coverage) / N
        accuracies.append(accuracy)

    return accuracies, row_means


def check_yes_no(input_str: str, inside: int = 0) -> int:
    """Convert yes/no string to binary value.

    Args:
        input_str: String to check for yes/no variants.
        inside: If non-zero, always returns 1.

    Returns:
        1 for 'yes' variants, 0 for 'no' variants.

    Raises:
        ValueError: If input is empty or not recognized as yes/no.
    """
    yes_variants = {"yes", "yeah"}
    no_variants = {"no", "nope", "not sure"}

    normalized_input = input_str.strip().lower()

    if not normalized_input:
        raise ValueError("Input cannot be empty or just whitespace.")

    if inside:
        return 1

    if normalized_input in yes_variants:
        return 1
    if normalized_input in no_variants:
        return 0
    else:
        raise ValueError("Input not recognized as 'yes' or 'no'.")


def average_case_acc(
    data: Mapping[str, List[int] | List[List[int]]],
    prefix: str = "outs_"
) -> float:
    """Compute average accuracy across filtered data.

    Args:
        data: Dictionary mapping keys to lists or lists of lists of values.
        prefix: String prefix to filter relevant keys.

    Returns:
        The average accuracy as a float.
    """
    relevant_keys = [k for k in data if k.startswith(prefix)]
    filtered_data = [data[key] for key in relevant_keys if key in data]

    if isinstance(filtered_data[0], list):
        arr: npt.NDArray[np.float64] = np.array(filtered_data)
        mean_across = np.mean(arr, axis=0)
        final_mean = np.mean(mean_across)
    else:
        arr = np.array(filtered_data)
        final_mean = np.mean(arr, axis=0)

    return float(final_mean)


# ---------------------------------------------------------------------------
# Evaluator classes
# ---------------------------------------------------------------------------

class WorstEval:
    """Worst-case evaluator for forgetting tasks.

    Reads evaluation results from two JSON files (paraphrase and ICR variants),
    processes YES/NO answers to binary, and computes J_P, J_ICR, and J_W metrics.
    """

    responsefiles: List[str]
    run_name: str
    task: str
    logs: Dict[str, Any]

    def __init__(
        self,
        run_name: str,
        files: Optional[List[str]] = None,
        task: str = 'forget'
    ) -> None:
        self.responsefiles = files or []
        self.run_name = run_name
        self.task = task
        self.logs = {}

    def evaluate(self) -> Dict[str, Any]:
        """Compute worst-case metrics from two judge result files.

        Returns:
            Dictionary with J_P, J_ICR, J_W, and agg_value.
        """
        evals_para = load_jsonl(self.responsefiles[0])
        evals_icr = load_jsonl(self.responsefiles[1])

        # Align samples by GT to handle mismatched lengths
        if len(evals_para) != len(evals_icr):
            logger.warning(
                f"Mismatched sample counts: para={len(evals_para)}, icr={len(evals_icr)}. "
                "Aligning by GT field."
            )
            icr_by_gt = {d["GT"]: d for d in evals_icr}
            aligned_para, aligned_icr = [], []
            for d in evals_para:
                if d["GT"] in icr_by_gt:
                    aligned_para.append(d)
                    aligned_icr.append(icr_by_gt[d["GT"]])
            logger.info(f"Aligned to {len(aligned_para)} common samples")
            evals_para = aligned_para
            evals_icr = aligned_icr

        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())

        ans_newpara1: Dict[str, List[int]] = {}
        try:
            for qi in q_names:
                ans_list = []
                for d in evals_para:
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara1[f'{qi}'] = ans_list
            accs1, _ = compute_any_one_average(ans_newpara1, prefix="ans_")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute paraphrase accuracy: {e}")
            accs1 = [None]

        logger.info(f"J_P: {accs1[-1]}")

        ans_newpara2: Dict[str, List[int]] = {}
        try:
            for qi in q_names:
                ans_list = []
                for d in evals_icr:
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara2[f'{qi}'] = ans_list
            accs2, _ = compute_any_one_average(ans_newpara2, prefix="ans_")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute ICR accuracy: {e}")
            accs2 = [None]

        logger.info(f"J_ICR: {accs2[-1]}")

        acc_tens: Dict[str, List[int]] = {}
        for qi in q_names:
            acc_tens[f'outs_1_{qi}'] = ans_newpara1[qi]
            acc_tens[f'outs_2_{qi}'] = ans_newpara2[qi]
        accov2, _ = compute_any_one_average(acc_tens)
        logger.info(f"J_W: {accov2[-1]}")

        self.logs = {
            "Task_name": self.run_name,
            "Set": self.task,
            "J_P": accs1[-1],
            "J_ICR": accs2[-1],
            "J_W": accov2[-1],
            "agg_value": accov2[-1],
        }

        return self.logs


class AvgEval:
    """Average-case evaluator for retain tasks.

    Reads evaluation results from a single JSON file, processes YES/NO answers
    to binary, and computes J_avg metric.
    """

    responsefiles: List[str]
    run_name: str
    task: str
    logs: Dict[str, Any]
    max_samples: int

    def __init__(
        self,
        run_name: str,
        files: Optional[List[str]] = None,
        task: str = 'retain',
        max_samples: int = 400
    ) -> None:
        self.responsefiles = files or []
        self.run_name = run_name
        self.task = task
        self.max_samples = max_samples
        self.logs = {}

    def evaluate(self) -> Dict[str, Any]:
        """Compute average-case metrics from a single judge result file.

        Returns:
            Dictionary with J_avg and agg_value.
        """
        evals_para = load_jsonl(self.responsefiles[0])[:self.max_samples]
        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())

        try:
            ans_newpara1: Dict[str, List[int]] = {}
            for qi in q_names:
                ans_list = []
                for d in evals_para:
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara1[f'{qi}'] = ans_list
            accs = average_case_acc(ans_newpara1, prefix="ans_")
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to compute average accuracy: {e}")
            accs = None

        logger.info(f"J_avg: {accs}")

        self.logs = {
            "Task_name": self.run_name,
            "Set": self.task,
            "J_avg": accs,
            "agg_value": accs,
        }

        return self.logs
