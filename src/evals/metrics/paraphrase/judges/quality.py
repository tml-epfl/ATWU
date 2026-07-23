"""Quality judge — YES/NO evaluation of model responses against ground truth."""
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import create_model
from tqdm import tqdm

from evals.metrics.paraphrase.judges.local import LocalJudge
from evals.metrics.paraphrase.utils import get_api_key, load_jsonl, setup_logger

warnings.filterwarnings("ignore")

# Optional judge-API dependencies: keep the core repo importable without them.
# Install via `pip install .[paraphrase-judge]` (or the individual packages).
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _require_judge_dep(obj, package: str, judge_type: str):
    if obj is None:
        raise ImportError(
            f"The `{package}` package is required for judge type '{judge_type}'. "
            f"Install it with `pip install {package}` or `pip install .[paraphrase-judge]`."
        )

# Suppress httpx request logging (used internally by OpenAI SDK)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output) for OpenAI models
OPENAI_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "o3-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-pro": (30.00, 180.00),
}


class QualityJudge:
    """Uses an LLM as a judge to evaluate model responses against ground truth.

    Orchestrates: loading generated responses, preparing judge prompts, sending
    chunks to the judge (Gemini / OpenAI / local), parsing YES/NO verdicts,
    and saving results.
    """

    eval_task: str
    task: str
    icr_data: bool
    gen_file: Path
    output_base: Dict[str, str]
    logs: List[Any]
    jg_file_path: Path
    log_path: str
    logger: logging.Logger
    chunk_size: int
    questions: List[str]
    formatted_response: List[Dict[str, Any]]
    api_key: str

    def __init__(
        self,
        eval_task: str,
        output_base: Dict[str, str],
        questions: List[str],
        api_key: Optional[str] = None,
        task: str = 'forget',
        gen_file: Optional[Path] = None,
        icr: bool = False,
        chunk_size: Optional[int] = None,
        judge_type: str = "local",
        judge_model: Optional[str] = None,
        judge_quantize: bool = False,
        fix_qwen_keys: bool = True,
        substring_heuristic: bool = True,
        judge_instance: Optional[Any] = None,
        seed: Optional[int] = 42,
        prompt_file: Optional[str] = None,
    ) -> None:
        self.eval_task = eval_task
        self.task = task
        self.icr_data = icr
        self.gen_file = gen_file or Path("")
        self.output_base = output_base
        self.questions = questions
        self.logs = []
        self.judge_type = judge_type
        self.prompt_file = prompt_file
        self.judge = None  # LocalJudge or OpenAI instance
        self.fix_qwen_keys = fix_qwen_keys
        self.substring_heuristic = substring_heuristic

        self.openai_model = None
        self.openai_seed = seed
        self.openai_costs = []

        # Use pre-built judge if provided, otherwise initialize a new one
        if judge_instance is not None:
            self.api_key = ""
            self.judge = judge_instance
            if self.judge_type == "openai":
                self.openai_model = judge_model or "gpt-4o-mini"
        elif self.judge_type == "openai":
            _require_judge_dep(OpenAI, "openai", "openai")
            self.api_key = ""
            self.openai_model = judge_model or "gpt-4o-mini"
            self.judge = OpenAI()
        elif self.judge_type == "gemini":
            self.api_key = get_api_key(api_key)
        else:  # local judge
            self.api_key = ""
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            logs_dir = self.output_base.get("logs_dir", "./logs")
            self.logger = setup_logger("temp.log", logs_dir)
            self.logger.info(f"Initializing local judge with model: {judge_model}")
            self.judge = LocalJudge(model_name=judge_model, max_new_tokens=1024, quantize=judge_quantize)

        self.set_out_dirs()
        logs_dir = self.output_base.get("logs_dir", "./logs")
        self.logger = setup_logger(self.log_path, logs_dir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")
        self.logger.info(f"Using judge type: {self.judge_type}")
        self.chunk_size = chunk_size if chunk_size is not None else (3 if self.task == 'forget' else 5)
        self.formatted_response = []

    @staticmethod
    def create_judge_instance(judge_cfg: Dict[str, Any]) -> Optional[Any]:
        """Create a reusable judge instance from config.

        Call this once, then pass the result as ``judge_instance`` to every
        constructor to avoid reloading the model each time.
        """
        judge_type = judge_cfg.get("type", "local")
        judge_model = judge_cfg.get("model")
        judge_quantize = judge_cfg.get("quantize", False)

        if judge_type == "local":
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            logger.info(f"Creating shared LocalJudge: {judge_model}")
            return LocalJudge(model_name=judge_model, max_new_tokens=1024, quantize=judge_quantize)
        elif judge_type == "openai":
            _require_judge_dep(OpenAI, "openai", "openai")
            logger.info(f"Creating shared OpenAI client: {judge_model or 'gpt-4o-mini'}")
            return OpenAI()
        else:
            return None

    def save_logs(self) -> None:
        """Save all judged responses to a JSONL file (one JSON object per line)."""
        self.jg_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jg_file_path, "w") as f:
            for entry in self.formatted_response:
                f.write(json.dumps(entry) + "\n")
        self.logger.info(f"Saved {len(self.formatted_response)} judgments to: {self.jg_file_path}")

    def _append_chunk(self, chunk_entries: List[Dict[str, Any]]) -> None:
        """Append a chunk of judged entries to the JSONL file (incremental saving)."""
        self.jg_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jg_file_path, "a") as f:
            for entry in chunk_entries:
                f.write(json.dumps(entry) + "\n")

    def _save_debug_log(self, alternate_json: List[Dict[str, Any]]) -> None:
        """Save a detailed debug log combining inputs and judge verdicts."""
        debug_path = self.jg_file_path.with_name(
            self.jg_file_path.name.replace(".jsonl", "_debug.json")
        )

        if not self.formatted_response:
            self.logger.warning("No judged responses to save in debug log")
            return

        judged_field_names = {k for k in self.formatted_response[0].keys() if k.startswith("ans_")}

        debug_entries = []
        judged_idx = 0
        for orig_idx, orig_entry in enumerate(alternate_json):
            gt = orig_entry.get("GT", "")
            entry: Dict[str, Any] = {
                "sample_idx": orig_idx,
                "GT": gt,
                "task": self.task,
                "icr": self.icr_data,
            }

            matched = (
                judged_idx < len(self.formatted_response)
                and self.formatted_response[judged_idx].get("GT") == gt
            )
            if matched:
                judged = self.formatted_response[judged_idx]
                judged_idx += 1
                entry["judged"] = True
            else:
                judged = None
                entry["judged"] = False

            comparisons = []
            for ans_key in sorted(k for k in orig_entry.keys() if k.startswith("ans_")):
                q_key = ans_key.replace("ans_", "", 1)
                comparison = {
                    "field": ans_key,
                    "question": orig_entry.get(q_key, ""),
                    "generated_answer": orig_entry.get(ans_key, ""),
                }
                if not entry["judged"]:
                    comparison["judge_verdict"] = "CHUNK_FAILED"
                elif ans_key not in judged_field_names:
                    comparison["judge_verdict"] = "NOT_IN_SCHEMA"
                elif judged and ans_key in judged:
                    comparison["judge_verdict"] = judged[ans_key]
                else:
                    comparison["judge_verdict"] = "PARSE_ERROR"
                comparisons.append(comparison)
            entry["comparisons"] = comparisons
            debug_entries.append(entry)

        with open(debug_path, "w") as f:
            json.dump(debug_entries, f, indent=2)
        self.logger.info(
            f"Saved debug log ({len(debug_entries)} entries, {judged_idx} judged, "
            f"{len(debug_entries) - judged_idx} failed) to: {debug_path}"
        )

    @staticmethod
    def _fix_qwen_keys_in_responses(
        parsed_responses: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Fix common LLM judge typo: 'ans_q_wen' -> 'ans_q_qwen'."""
        return [
            {k.replace("_wen", "_qwen"): v for k, v in resp.items()}
            for resp in parsed_responses
        ]

    def _apply_substring_heuristic(
        self,
        formatted_chunk: List[Dict[str, str]],
        original_chunk: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Override judge verdict to YES when GT and predicted answer share a substring match."""
        overrides = 0
        for judged, orig in zip(formatted_chunk, original_chunk):
            gt = orig.get("GT", "").lower().strip()
            if not gt:
                continue
            for key in filter(
                lambda key: key.startswith("ans_"),
                judged.keys()
            ):
                pred = orig.get(key, "").lower().strip()
                if not pred:
                    continue
                if pred in gt or gt in pred:
                    if judged[key] != "YES":
                        overrides += 1
                        judged[key] = "YES"
        if overrides > 0:
            self.logger.info(f"Substring heuristic overrode {overrides} verdicts to YES")
        return formatted_chunk

    def set_out_dirs(self) -> None:
        """Sets up output directories and file paths."""
        judgments_dir = Path(self.output_base.get("judgments_dir", "./paraphrase_judgments"))
        judgments_dir.mkdir(parents=True, exist_ok=True)

        icr_suffix = f'icr_{self.icr_data}'
        jg_filename = f"{self.eval_task}_{icr_suffix}.jsonl"

        self.jg_file_path = judgments_dir / jg_filename
        self.log_path = f"{self.eval_task}_{icr_suffix}.log"

    def _parse_local_judge_response(self, response_text: str, num_samples: int) -> Optional[List[Dict[str, str]]]:
        """Parse JSON response from local judge model."""
        try:
            response_text = response_text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            if isinstance(parsed, list) and len(parsed) >= num_samples:
                if len(parsed) > num_samples:
                    self.logger.warning(
                        f"Judge returned {len(parsed)} items, expected {num_samples}. "
                        f"Truncating to first {num_samples}."
                    )
                    parsed = parsed[:num_samples]
                if self.fix_qwen_keys:
                    parsed = self._fix_qwen_keys_in_responses(parsed)
                return parsed
            else:
                self.logger.warning(f"Parsed response is not a list or has wrong length. Expected list of {num_samples}, got: {type(parsed)} with length {len(parsed) if isinstance(parsed, list) else 'N/A'}")
                return None
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse local judge response as JSON: {e}")
            self.logger.warning(f"Raw response: {response_text[:500]}")
            return None

    def generate(self) -> None:
        """Performs the LLM-based judging process with incremental saving."""
        if self.judge_type == "gemini":
            _require_judge_dep(genai, "google-genai", "gemini")
            client = genai.Client(api_key=self.api_key)

        responses = load_jsonl(self.gen_file)

        # Load prompt from judges/prompts/ subdirectory
        prompt_dir = Path(__file__).parent / "prompts"
        prompt_filename = self.prompt_file or f"{self.judge_type}.txt"
        with open(prompt_dir / prompt_filename, "r") as file:
            prompt_base = file.read()

        # Build alternate_json from the enriched generation file
        # (which already contains GT, question texts, and ans_* fields)
        alternate_json: List[Dict[str, Any]] = []
        for entry in responses:
            processed: Dict[str, Any] = {"GT": entry["GT"]}
            for qid in self.questions:
                processed[qid] = entry[qid]
                processed[f"ans_{qid}"] = entry[f"ans_{qid}"].lower().replace("assistant", "--")
            alternate_json.append(processed)

        # Resume from existing partial judgments
        if self.jg_file_path.exists():
            self.formatted_response = load_jsonl(self.jg_file_path)
            self.logger.info(
                f"Resuming judging from {len(self.formatted_response)} "
                f"already-judged samples"
            )
        else:
            self.formatted_response = []
        start_sample = len(self.formatted_response)

        # Build dynamic Pydantic response schema from actual question fields
        answer_fields = {f"ans_{q}": (str, ...) for q in self.questions}
        response_schema_type = create_model("DynamicJudgeResponse", **answer_fields) # type: ignore

        # Determine score label for tqdm postfix
        if self.task == 'forget':
            score_label = "J_ICR" if self.icr_data else "J_P"
        else:
            score_label = "J_avg"

        # Determine where to start based on already-judged samples
        # Each chunk covers chunk_size samples, so start from the chunk boundary
        start_offset = (start_sample // self.chunk_size) * self.chunk_size

        pbar = tqdm(
            range(start_offset, len(alternate_json), self.chunk_size),
            desc=f"Judging {self.task} (ICR={self.icr_data})",
        )
        for i in pbar:
            chunk = alternate_json[i:i + self.chunk_size]
            chunk_num = i // self.chunk_size + 1
            total_chunks = (len(alternate_json) + self.chunk_size - 1) // self.chunk_size

            # Skip chunks already covered by resumed data
            if i + len(chunk) <= start_sample:
                continue

            if chunk_num == 1 and chunk:
                answer_fields_list = [k for k in chunk[0].keys() if k.startswith('ans_')]
                self.logger.info(f"Answer fields in input data: {sorted(answer_fields_list)}")
                self.logger.info(f"Expected answer fields for {self.task}: {sorted([f for f in response_schema_type.model_fields.keys()])}")

            self.logger.info(f"Processing chunk {chunk_num} of {total_chunks} of length {len(chunk)}")

            answer_fields_in_chunk = [k for k in chunk[0].keys() if k.startswith('ans_')]

            query = str(chunk)
            query += (
                f"\n\nREMEMBER: You are a Judge! Evaluate EACH of these answer fields: {', '.join(answer_fields_in_chunk)}. "
                "For each field, compare the answer against the ground truth (GT) and corresponding question. "
                "Return a valid JSON array with one object per sample. "
                f"Each object MUST have ALL {len(answer_fields_in_chunk)} answer fields: {', '.join(answer_fields_in_chunk)}. "
                "Each field value must be \"YES\" or \"NO\". "
                "Do not repeat answers or try to answer questions yourself!"
            )

            strr = (
                "\n\nHere are the tests to be evaluated. This is a list of JSON objects, "
                "where for each sample, we have:\n"
                "- 'GT': the ground truth answer\n"
                "- Question fields (e.g., 'question', 'q_para_0', 'q_para_1', etc.)\n"
                "- Answer fields (e.g., 'ans_question', 'ans_q_para_0', etc.) - "
                "these are the test responses to evaluate\n\n"
                "You must return a JSON array where each element corresponds to one input sample. "
                "Each element must be a JSON object with fields matching the answer field names "
                "from the input, and each field value must be \"YES\" or \"NO\" indicating whether "
                "that test response contains all the information from the ground truth.\n"
            )

            full_system_instruction = prompt_base + strr

            if self.judge_type == "gemini":
                my_response = None
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        response = client.models.generate_content( # type: ignore
                            model="gemini-flash-latest",
                            config=types.GenerateContentConfig(
                                system_instruction=full_system_instruction,
                                temperature=0.,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                                response_mime_type="application/json",
                                response_schema=list[response_schema_type], # type: ignore
                            ),
                            contents=query
                        )
                        my_response = response.parsed
                    except Exception as e:
                        self.logger.warning(f"Gemini API call failed (attempt {attempt}/{max_retries}): {e}")
                        continue

                    if my_response is not None:
                        break
                    self.logger.warning(f"Gemini returned empty parsed response (attempt {attempt}/{max_retries})")
            elif isinstance(self.judge, OpenAI):
                api_kwargs: Dict[str, Any] = {
                    "model": self.openai_model,
                    "messages": [
                        {"role": "system", "content": full_system_instruction},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.0,
                }
                if self.openai_seed is not None:
                    api_kwargs["seed"] = self.openai_seed

                my_response = None
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    # 1. API call
                    try:
                        response = self.judge.chat.completions.create(**api_kwargs)
                        response_text = response.choices[0].message.content
                    except Exception as e:
                        self.logger.warning(f"OpenAI API call failed (attempt {attempt}/{max_retries}): {e}")
                        continue

                    # Track cost for every attempt
                    usage = response.usage
                    if usage is not None:
                        input_price, output_price = OPENAI_PRICING.get(
                            self.openai_model, (0.0, 0.0) # type: ignore
                        )
                        chunk_cost = (
                            usage.prompt_tokens * input_price
                            + usage.completion_tokens * output_price
                        ) / 1_000_000
                        self.openai_costs.append({
                            "chunk": chunk_num,
                            "attempt": attempt,
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "cost_usd": chunk_cost,
                        })
                        self.logger.info(
                            f"Chunk {chunk_num} (attempt {attempt}) cost: ${chunk_cost:.6f} "
                            f"({usage.prompt_tokens} in, {usage.completion_tokens} out)"
                        )

                    if response_text is None:
                        self.logger.warning(f"Empty OpenAI response (attempt {attempt}/{max_retries})")
                        continue

                    # 2. Parse JSON
                    parsed_responses = self._parse_local_judge_response(response_text, len(chunk))
                    if parsed_responses is None:
                        self.logger.warning(f"Parse failed for chunk {chunk_num} (attempt {attempt}/{max_retries})")
                        continue

                    # 3. Validate with Pydantic
                    try:
                        my_response = [response_schema_type(**resp) for resp in parsed_responses]
                    except Exception as e:
                        self.logger.warning(f"Pydantic validation failed (attempt {attempt}/{max_retries}): {e}")
                        continue

                    break
            elif isinstance(self.judge, LocalJudge):
                response_text = self.judge.judge(
                    system_prompt=full_system_instruction,
                    user_prompt=query
                )
                if response_text is None:
                    my_response = None
                else:
                    parsed_responses = self._parse_local_judge_response(response_text, len(chunk))
                    if parsed_responses is not None:
                        try:
                            my_response = [response_schema_type(**resp) for resp in parsed_responses]
                        except Exception as e:
                            self.logger.warning(f"Pydantic validation failed: {e}")
                            self.logger.warning(f"Expected schema fields: {list(response_schema_type.model_fields.keys())}")
                            if parsed_responses:
                                self.logger.warning(f"Received fields in first response: {list(parsed_responses[0].keys())}")

                            answer_fields_in_chunk = [k for k in chunk[0].keys() if k.startswith('ans_')]
                            all_valid = True
                            for resp in parsed_responses:
                                resp_fields = set(resp.keys())
                                expected_fields = set(answer_fields_in_chunk)
                                if resp_fields != expected_fields:
                                    missing = expected_fields - resp_fields
                                    extra = resp_fields - expected_fields
                                    if missing:
                                        self.logger.warning(f"Missing fields: {missing}")
                                    if extra:
                                        self.logger.warning(f"Extra fields: {extra}")
                                    all_valid = False
                                    break

                            if all_valid:
                                self.logger.info("Using flexible validation - responses match input data structure")
                                class MockJudgeResponse:
                                    def __init__(self, data):
                                        self._data = data
                                    def model_dump(self):
                                        return self._data
                                my_response = [MockJudgeResponse(resp) for resp in parsed_responses]
                            else:
                                my_response = None
                    else:
                        my_response = None
            else:
                self.logger.error(f"No judge available for type: {self.judge_type}")
                my_response = None

            if my_response is None:
                logs_dir = Path(self.output_base.get("logs_dir", "./logs"))
                problematic_file = logs_dir / "problematic_chunks.txt"
                with open(problematic_file, 'a') as f:
                    f.write(f"Chunk {chunk_num} of {total_chunks} of length {len(chunk)}\n")
                self.logger.warning(f"Failed to parse response for chunk {chunk_num}")
            else:
                formatted_response_chunk = [m.model_dump() for m in my_response] # type: ignore
                for j, single_response in enumerate(formatted_response_chunk):
                    if j == len(chunk):
                        break
                    single_response.update({"GT": chunk[j]['GT']})

                if self.substring_heuristic:
                    formatted_response_chunk = self._apply_substring_heuristic(
                        formatted_response_chunk, chunk[:len(formatted_response_chunk)]
                    )

                self.formatted_response.extend(formatted_response_chunk)
                self._append_chunk(formatted_response_chunk)

                # Update running score in tqdm postfix
                if self.task == 'forget':
                    # Coverage: fraction of samples with at least one YES
                    n_covered = sum(
                        1 for resp in self.formatted_response
                        if any(
                            resp[k].upper() == "YES"
                            for k in resp if k.startswith("ans_")
                        )
                    )
                    score = n_covered / len(self.formatted_response)
                else:
                    # Average: mean YES rate across all answer fields
                    total_yes = sum(
                        1 for resp in self.formatted_response
                        for k in resp if k.startswith("ans_") and resp[k].upper() == "YES"
                    )
                    total_fields = sum(
                        1 for resp in self.formatted_response
                        for k in resp if k.startswith("ans_")
                    )
                    score = total_yes / total_fields if total_fields else 0.0
                pbar.set_postfix(**{score_label: f"{score:.3f}"})

        self.logger.info(f"Paraphrase {self.judge_type} judge evals done for {self.task} set - iCR {self.icr_data}")
        self.logger.info(f"Total judged: {len(self.formatted_response)} samples saved to: {self.jg_file_path}")
        self._save_debug_log(alternate_json)

        if self.openai_costs:
            total_cost = sum(c["cost_usd"] for c in self.openai_costs)
            total_prompt = sum(c["prompt_tokens"] for c in self.openai_costs)
            total_completion = sum(c["completion_tokens"] for c in self.openai_costs)
            self.logger.info(
                f"Total OpenAI cost: ${total_cost:.6f} "
                f"({total_prompt} prompt + {total_completion} completion tokens)"
            )
            cost_report = {
                "model": self.openai_model,
                "task": self.task,
                "icr": self.icr_data,
                "total_cost_usd": total_cost,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "chunks": self.openai_costs,
            }
            icr_suffix = f"icr_{self.icr_data}"
            cost_file = Path(self.output_base.get("logs_dir", "./logs")) / f"openai_costs_{self.task}_{icr_suffix}.json"
            cost_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cost_file, "w") as f:
                json.dump(cost_report, f, indent=2)
            self.logger.info(f"Cost report saved to: {cost_file}")