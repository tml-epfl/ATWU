"""Win-rate judge — compares pretrained vs. unlearned model responses."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from evals.metrics.paraphrase.judges.local import LocalJudge
from evals.metrics.paraphrase.judges.quality import (
    OpenAI,
    _require_judge_dep,
    genai,
    types,
)
from evals.metrics.paraphrase.utils import get_api_key


@dataclass
class JudgeResponse:
    """Structured response from a win-rate judge."""
    score_assistant_1: int
    score_assistant_2: int
    explanation: str


class WinrateJudge:
    """Manages the win-rate evaluation comparing two models.

    Loads repetitiveness results for both a pretrained baseline and an
    unlearned model, uses a judge to compare responses, and computes
    win rate with tie-breaking (0.5 per tie).
    """

    name: str
    task_name: str
    output_base: Dict[str, str]
    nsamples: int
    repfile_base: Path
    repfile_unlearnt: Path
    out_file_path: Path
    sys_prompt: str
    prompt_template: str
    criteria: str
    api_key: str
    logger: logging.Logger

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_base: Dict[str, str],
        api_key: Optional[str] = None,
        judge_type: str = "local",
        judge_model: Optional[str] = None,
        judge_quantize: bool = False,
        judge_instance: Optional[Any] = None,
        seed: Optional[int] = 42,
        baseline_path: Optional[str] = None,
    ) -> None:
        self.name = model_name
        self.task_name = task_name
        self.output_base = output_base
        self.nsamples = 100
        self.judge_type = judge_type
        self.baseline_path = baseline_path
        self.logger = logging.getLogger(__name__)
        self.judge = None  # LocalJudge or OpenAI instance
        self.openai_model = None
        self.openai_seed = seed

        if judge_instance is not None:
            self.api_key = ""
            self.judge = judge_instance
            if self.judge_type == "openai":
                self.openai_model = judge_model or "gpt-4o-mini"
        elif self.judge_type == "gemini":
            self.api_key = get_api_key(api_key)
        elif self.judge_type == "openai":
            _require_judge_dep(OpenAI, "openai", "openai")
            self.api_key = ""
            self.openai_model = judge_model or "gpt-4o-mini"
            self.judge = OpenAI()
        else:  # local judge
            self.api_key = ""
            judge_model = judge_model or "meta-llama/Llama-3.1-8B-Instruct"
            self.logger.info(f"Initializing local judge with model: {judge_model}")
            self.judge = LocalJudge(model_name=judge_model, quantize=judge_quantize)

        self.set_out_dirs()
        self.prompts()

    def set_out_dirs(self, prefix: str = 'WR') -> None:
        """Sets up output directories and file paths."""
        winrate_dir = Path(self.output_base.get("winrate_dir", "./winrate"))
        repetitiveness_dir = Path(self.output_base.get("repetitiveness_dir", "./repetitiveness"))

        winrate_dir.mkdir(parents=True, exist_ok=True)

        if self.baseline_path is not None:
            self.repfile_base = Path(self.baseline_path)
        else:
            self.repfile_base = repetitiveness_dir / "pretrained.jsonl"
        assert self.repfile_base.exists(), (
            f"Baseline repetitiveness file not found: {self.repfile_base}. "
            "Either run `eval_task=repet` for the pretrained model first, "
            "or set `baseline_path` to an existing file."
        )

        self.repfile_unlearnt = repetitiveness_dir / f"{self.task_name}.jsonl"
        assert self.repfile_unlearnt.exists(), (
            "The AlpacaEval generations do not yet exist for the unlearnt model; "
            "do `eval_task=repet`"
        )

        self.out_file_path = winrate_dir / f"{prefix}_{self.task_name}.jsonl"

    def prompts(self) -> None:
        """Set up prompts for the judge evaluation."""
        self.sys_prompt = (
            "You are a helpful and precise assistant for checking the quality of the answer. "
            "Your response MUST be a valid JSON object on a single line. "
            "Do NOT use newlines inside string values - keep the entire JSON on one line. "
            "The JSON must conform to this structure: "
            "{\"score_assistant_1\": int, \"score_assistant_2\": int, \"explanation\": str}. "
            "Example: {\"score_assistant_1\": 7, \"score_assistant_2\": 8, \"explanation\": \"Assistant 2 provides more detail and accuracy.\"}"
        )
        self.prompt_template = (
            "[Question]\n{question}\n\n"
            "[The Start of Assistant 1's Answer]\n{answer_1}\n\n"
            "[The End of Assistant 1's Answer]\n\n"
            "[The Start of Assistant 2's Answer]\n{answer_2}\n\n"
            "[The End of Assistant 2's Answer]\n\n"
            "[System]\n{criteria}\n\n"
        )
        self.criteria = (
            "We would like to request your feedback on the performance of two AI assistants "
            "in response to the user question displayed above.\n"
            "Please rate the helpfulness, relevance, accuracy, level of details of their responses. "
            "Each assistant receives an overall score on a scale of 1 to 10, where a higher score "
            "indicates better overall performance.\n"
            "Provide the scores for Assistant 1 and 2, and a comprehensive explanation of your evaluation, "
            "avoiding any potential bias and ensuring that the order in which the responses were presented "
            "does not affect your judgment, all within the specified JSON format."
        )

    def _parse_judge_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON response from the judge, handling markdown fences."""
        response_text = response_text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        if not response_text.startswith('{') and '{' in response_text:
            start_idx = response_text.index('{')
            response_text = response_text[start_idx:]

        parsed_response = json.loads(response_text.strip())
        if isinstance(parsed_response, list) and len(parsed_response) > 0:
            judge_response = parsed_response[0]
        elif isinstance(parsed_response, dict):
            judge_response = parsed_response
        else:
            self.logger.warning(f"Unexpected JSON structure: {parsed_response}")
            return None

        if all(k in judge_response for k in ["score_assistant_1", "score_assistant_2", "explanation"]):
            return judge_response
        else:
            self.logger.warning(f"Missing required keys in response: {judge_response}")
            return None

    def call_judge_api(self, full_prompt: str) -> Optional[Dict[str, Any]]:
        """Calls the judge (Gemini / OpenAI / local).

        Returns:
            The parsed judge response dict, or None if the call fails.
        """
        if self.judge_type == "gemini":
            _require_judge_dep(genai, "google-genai", "gemini")
            client = genai.Client(api_key=self.api_key)
            query = [{"role": "user", "parts": [{"text": full_prompt}]}]

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model="gemini-flash-latest",
                        config=types.GenerateContentConfig(
                            system_instruction=self.sys_prompt,
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=list[JudgeResponse],
                        ),
                        contents=query # type: ignore
                    )
                except Exception as e:
                    self.logger.warning(f"Gemini API call failed (attempt {attempt}/{max_retries}): {e}")
                    continue

                if response and response.text:
                    try:
                        parsed_response = json.loads(response.text)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Gemini JSON parse failed (attempt {attempt}/{max_retries}): {e}")
                        continue
                    if parsed_response and isinstance(parsed_response, list) and len(parsed_response) > 0:
                        return parsed_response[0]
                    self.logger.warning(f"Unexpected JSON structure (attempt {attempt}/{max_retries}): {parsed_response}")
                    continue

                self.logger.warning(f"Empty Gemini response (attempt {attempt}/{max_retries})")

            self.logger.error("Gemini judge failed after all retries")
            return None

        elif isinstance(self.judge, OpenAI):
            api_kwargs: Dict[str, Any] = {
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                "temperature": 0.0,
            }
            if self.openai_seed is not None:
                api_kwargs["seed"] = self.openai_seed

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.judge.chat.completions.create(**api_kwargs)
                    response_text = response.choices[0].message.content
                except Exception as e:
                    self.logger.warning(f"OpenAI API call failed (attempt {attempt}/{max_retries}): {e}")
                    continue

                if response_text is None:
                    self.logger.warning(f"Empty OpenAI response (attempt {attempt}/{max_retries})")
                    continue

                parsed = self._parse_judge_response(response_text)
                if parsed is not None:
                    return parsed
                self.logger.warning(f"Parse failed (attempt {attempt}/{max_retries})")

            self.logger.error("OpenAI judge failed after all retries")
            return None

        elif isinstance(self.judge, LocalJudge):
            try:
                response_text = self.judge.judge(
                    system_prompt=self.sys_prompt,
                    user_prompt=full_prompt
                )
                if response_text is None:
                    return None

                self.logger.debug(f"Attempting to parse: {response_text[:200]}...")
                return self._parse_judge_response(response_text)

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse local judge response as JSON: {e}")
                self.logger.error(f"Raw response text: {response_text[:500]}") # type: ignore
                return None
            except Exception as e:
                self.logger.error(f"Local judge request failed: {e}")
                raise

        else:
            self.logger.error(f"No judge available for type: {self.judge_type}")
            return None

    def calculate_win_rate(self, evaluations_data: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Calculates the win rate of assistant_2 against assistant_1."""
        wins = 0
        losses = 0
        ties = 0
        total_comparisons = len(evaluations_data)

        if total_comparisons == 0:
            return 0.0, {"win-rate": 0.0, "wins": 0, "losses": 0, "ties": 0}

        for evaluation in evaluations_data:
            score_a1 = evaluation.get('score_assistant_1')
            score_a2 = evaluation.get('score_assistant_2')
            if isinstance(score_a1, int) and isinstance(score_a2, int):
                if score_a2 > score_a1:
                    wins += 1
                elif score_a2 < score_a1:
                    losses += 1
                else:
                    ties += 1

        win_rate = (wins + 0.5 * ties) / total_comparisons

        return win_rate, {"win-rate": win_rate, "wins": wins, "losses": losses, "ties": ties}

    def win_rate_evaluation(self) -> None:
        """Perform win-rate evaluation comparing pretrained and unlearned models."""
        if self.out_file_path.is_file():
            self.logger.info(f"Responses from judge already exist at {self.out_file_path}")

            with open(self.out_file_path, "r") as f:
                evaluation_results = json.load(f)["results"]
            win_rate, counts = self.calculate_win_rate(evaluation_results)

            self.logger.info(f"Total comparisons: {len(evaluation_results)}")
            self.logger.info(f"Assistant 2 Wins: {counts['wins']}")
            self.logger.info(f"Assistant 1 Wins: {counts['losses']}")
            self.logger.info(f"Ties: {counts['ties']}")
            self.logger.info(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")
            return

        data_pairs: List[Dict[str, str]] = []

        with open(self.repfile_base, "r") as f:
            data1 = json.load(f)['results']
        with open(self.repfile_unlearnt, "r") as f:
            data2 = json.load(f)['results']

        ct = 0
        for (item1, item2) in zip(data1, data2):
            data_pairs.append({
                "question": f"{item1['instruction']}. (Pair {ct+1})",
                "answer_1": f"{item1['prediction']}. (Assistant 1, Pair {ct+1})",
                "answer_2": f"{item2['prediction']}. (Assistant 2, Pair {ct+1})"
            })
            ct += 1
            if ct >= self.nsamples:
                break

        # Resume from partial checkpoint if it exists
        partial_path = self.out_file_path.with_suffix(".partial")
        evaluation_results: List[Dict[str, Any]] = []
        start_idx = 0
        if partial_path.exists():
            with open(partial_path) as f:
                for line in f:
                    if line.strip():
                        evaluation_results.append(json.loads(line))
            start_idx = len(evaluation_results)
            self.logger.info(f"Resuming winrate from {start_idx} saved pairs")

        self.logger.info(
            f"Evaluating {len(data_pairs) - start_idx} remaining pairs "
            f"(of {len(data_pairs)} total) for unlearn-model against {self.name}"
        )

        pbar = tqdm(
            enumerate(data_pairs[start_idx:], start=start_idx),
            total=len(data_pairs) - start_idx,
            desc="Winrate",
        )
        for i, pair in pbar:
            question = pair["question"]
            answer_1 = pair["answer_1"]
            answer_2 = pair["answer_2"]

            full_prompt = self.prompt_template.format(
                question=question,
                answer_1=answer_1,
                answer_2=answer_2,
                criteria=self.criteria
            )

            try:
                judge_response_obj = self.call_judge_api(full_prompt)

                if judge_response_obj:
                    score_assistant_1 = judge_response_obj.get('score_assistant_1')
                    score_assistant_2 = judge_response_obj.get('score_assistant_2')
                    explanation = judge_response_obj.get('explanation')
                else:
                    score_assistant_1 = None
                    score_assistant_2 = None
                    explanation = "Failed to get structured response from API."
                    self.logger.error(f"Error: No structured response for question: {question}")

            except Exception as e:
                self.logger.error(f"An error occurred while processing question: {question}. Error: {e}")
                score_assistant_1 = None
                score_assistant_2 = None
                explanation = f"Processing error: {e}"

            result_entry = {
                "question": question,
                "answer_asst_1": answer_1,
                "answer_asst_2": answer_2,
                "score_assistant_1": score_assistant_1,
                "score_assistant_2": score_assistant_2,
                "explanation": explanation
            }
            evaluation_results.append(result_entry)

            # Save incrementally to partial checkpoint
            with open(partial_path, "a") as f:
                f.write(json.dumps(result_entry) + "\n")

            wr, cts = self.calculate_win_rate(evaluation_results)
            pbar.set_postfix(wr=f"{wr:.3f}", W=cts["wins"], L=cts["losses"], T=cts["ties"])

        win_rate, counts = self.calculate_win_rate(evaluation_results)

        self.logger.info(f"Total comparisons: {len(evaluation_results)}")
        self.logger.info(f"Assistant 2 Wins: {counts['wins']}")
        self.logger.info(f"Assistant 1 Wins: {counts['losses']}")
        self.logger.info(f"Ties: {counts['ties']}")
        self.logger.info(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")

        savefile: Dict[str, Any] = {
            "winrate": win_rate,
            "counts": counts,
            "results": evaluation_results
        }

        try:
            with open(self.out_file_path, 'w', encoding='utf-8') as f:
                json.dump(savefile, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Evaluation complete. Results saved to {self.out_file_path}")

            # Clean up partial checkpoint
            if partial_path.exists():
                partial_path.unlink()
                self.logger.info("Cleaned up partial checkpoint")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while saving results: {e}")
            raise
