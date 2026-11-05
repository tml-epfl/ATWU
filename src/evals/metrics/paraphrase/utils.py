"""Pure utility helpers for paraphrase evaluation."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_api_key(api_key: Optional[str] = None) -> str:
    """Get Gemini API key from parameter or environment.

    Args:
        api_key: Optional API key. If None, will check GOOGLE_API_KEY environment variable.

    Returns:
        The API key string.

    Raises:
        ValueError: If no API key is provided and GOOGLE_API_KEY is not set.
    """
    if api_key is not None:
        return api_key

    key = os.environ.get("GOOGLE_API_KEY")
    if key is None:
        raise ValueError(
            "Gemini API key required. Set GOOGLE_API_KEY environment "
            "variable or pass via api_key parameter."
        )
    return key


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a list of dicts, supporting both JSON array and JSONL formats.

    Args:
        path: Path to the file.

    Returns:
        List of dictionaries loaded from the file.
    """
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)  # legacy monolithic JSON array
        return [json.loads(line) for line in f if line.strip()]


def setup_logger(
    log_file_name: str = "application.log",
    log_dir: str = "./",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """Sets up a logger that prints to the console and saves to a file.

    Args:
        log_file_name: The name of the log file.
        log_dir: Directory to store the log file.
        console_level: The minimum logging level for console output.
        file_level: The minimum logging level for file output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("ParaphraseEval")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir_path / log_file_name

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logger '{logger.name}' set up")
    return logger