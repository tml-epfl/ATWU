"""Output directory and file management for paraphrase evaluation."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from evals.metrics.paraphrase.utils import setup_logger
from trainer.utils import seed_everything

logger = logging.getLogger(__name__)


class Evaluator:
    """Manages output directories, file paths, and result persistence for
    paraphrase quality evaluation.

    Generation logic lives in ``quality._generate_responses``; this class
    handles where results are stored and provides ``save_logs``.

    Attributes:
        icr_data: Whether this evaluation uses in-context retention (ICR).
            Affects output file naming (``icr_True`` vs ``icr_False``).
        eval_task: Task identifier (e.g., 'forget', 'retain').
        task: The specific evaluation task.
        output_base: Output paths configuration dict.
        logs: List of generation result dicts to be saved.
        out_path: Path to the generation output JSONL file.
        jg_file_path: Path to the judge evaluation JSONL file.
    """

    def __init__(
        self,
        eval_task: str,
        output_base: Dict[str, str],
        generation: Dict[str, Any],
        icr: bool,
        task: str,
    ) -> None:
        self.icr_data = icr
        self.eval_task = eval_task
        self.task = task
        self.output_base = output_base
        self.generation = generation
        self.logs: List[Dict[str, Any]] = []

        self.set_out_dirs()

        logs_dir: str = self.output_base.get("logs_dir", "./logs")
        self.logger = setup_logger(self.log_path, logs_dir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")

        seed_everything()

    def set_out_dirs(self) -> None:
        """Sets up output directories and file paths for evaluation results."""
        generations_dir = Path(self.output_base.get("generations_dir", "./paraphrase_generations"))
        judgments_dir = Path(self.output_base.get("judgments_dir", "./paraphrase_judgments"))

        generations_dir.mkdir(parents=True, exist_ok=True)
        judgments_dir.mkdir(parents=True, exist_ok=True)

        icr_suffix = f'icr_{self.icr_data}'
        filename = f"{self.eval_task}_{icr_suffix}.jsonl"

        self.out_path = generations_dir / filename
        self.jg_file_path = judgments_dir / filename
        self.log_path = f"{icr_suffix}.log"

    def save_logs(self) -> None:
        """Save all logs to a JSONL file (one JSON object per line)."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            for entry in self.logs:
                f.write(json.dumps(entry) + "\n")
        self.logger.info(f"Saved {len(self.logs)} generations to: {self.out_path}")

    def append_log(self, entry: Dict[str, Any]) -> None:
        """Append a single entry to the JSONL file (incremental saving)."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def count_saved(self) -> int:
        """Count the number of entries already saved to the output file."""
        if not self.out_path.exists():
            return 0
        with open(self.out_path) as f:
            return sum(1 for line in f if line.strip())