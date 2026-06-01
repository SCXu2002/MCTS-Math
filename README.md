# Simple Math Solver

A lightweight math-problem solver with two interchangeable LLM backends:

- API backend: OpenAI-compatible chat completions API
- Local backend: Hugging Face `transformers` causal language model

The first goal is intentionally small: send a math problem to an LLM with a clear solving prompt and print the model's answer.

## Install

```bash
pip install -r requirements.txt
```

For API usage, copy the example config:

```bash
copy api_config.example.json api_config.json
```

Then edit `api_config.json`:

```json
{
  "api_key": "your_api_key_here",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4.1-mini"
}
```

`api_config.json` is ignored by Git so your real key is not uploaded.

## API Example

```bash
python -m math_solver.cli api ^
  --problem "Find all positive integers n such that n^2 + n + 1 is divisible by 7."
```

## Local Transformers Example

```bash
python -m math_solver.cli local ^
  --model-path C:\path\to\local\model ^
  --problem "Find all positive integers n such that n^2 + n + 1 is divisible by 7."
```

## Prompt

The solver asks the model to reason step by step and place the final answer in:

```text
\boxed{...}
```

This makes answer extraction easier in later iterations.
