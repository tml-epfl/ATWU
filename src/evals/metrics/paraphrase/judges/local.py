"""Local judge using open-source models for on-device inference."""
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)


class LocalJudge:
    """Local judge using open-source models for paraphrase evaluation.

    Uses 4-bit quantization (BitsAndBytes fp4) to fit large models on consumer GPUs,
    and right padding for single-sequence generation.

    Args:
        model_name: HuggingFace model name/path for the judge model
        device: Device to run inference on ('cuda' or 'cpu')
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        batch_size: Batch size for inference
        token: HuggingFace API token
        quantize: Whether to use 4-bit quantization (default False)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        batch_size: int = 4,
        token: Optional[str] = None,
        quantize: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        logger.info(f"Loading local judge model: {model_name} (quantize={quantize})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "token": token,
            "trust_remote_code": True,
            "device_map": "auto" if device == "cuda" else None,
        }
        if quantize and device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()

        logger.info(f"Local judge model loaded on {device}")

    def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Run judge inference on a single prompt.

        Args:
            system_prompt: System instruction for the judge
            user_prompt: User query containing the evaluation task
            response_format: Optional JSON schema (not used for local models)

        Returns:
            Generated response from the judge model, or None if generation fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during local judge generation: {e}")
            return None

    def judge_batch(
        self,
        system_prompt: str,
        user_prompts: List[str],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> List[Optional[str]]:
        """Run judge inference on a batch of prompts.

        Args:
            system_prompt: System instruction for the judge
            user_prompts: List of user queries
            response_format: Optional JSON schema (not used for local models)

        Returns:
            List of generated responses
        """
        results = []

        for i in range(0, len(user_prompts), self.batch_size):
            batch_prompts = user_prompts[i:i+self.batch_size]

            messages_batch = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                for prompt in batch_prompts
            ]

            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in messages_batch
            ]

            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                for output in outputs:
                    generated_text = self.tokenizer.decode(
                        output[inputs.input_ids.shape[1]:],
                        skip_special_tokens=True,
                    )
                    results.append(generated_text.strip())

            except Exception as e:
                logger.error(f"Error during batch generation: {e}")
                results.extend([None] * len(batch_prompts))

        return results

    def __del__(self):
        """Cleanup model from memory."""
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
