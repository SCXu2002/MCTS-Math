"""LLM backend implementations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .prompt import MathPrompt, as_plain_text


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_new_tokens: int = 1024
    top_p: float = 0.95


class ApiMathSolver:
    """Math solver using an OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.generation_config = generation_config or GenerationConfig()
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def solve(self, prompt: MathPrompt) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=self.generation_config.temperature,
            max_tokens=self.generation_config.max_new_tokens,
            top_p=self.generation_config.top_p,
        )
        return response.choices[0].message.content or ""


class TransformersMathSolver:
    """Math solver using a local Hugging Face causal language model."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        generation_config: Optional[GenerationConfig] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)

        self.generation_config = generation_config or GenerationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def solve(self, prompt: MathPrompt) -> str:
        import torch

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = as_plain_text(prompt)

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=self.generation_config.temperature > 0,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                max_new_tokens=self.generation_config.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
