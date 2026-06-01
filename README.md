# Simple Math Solver

A lightweight math-problem solver with two interchangeable LLM backends:

- API backend: OpenAI-compatible chat completions API
- Local backend: Hugging Face `transformers` causal language model

The first goal is intentionally small: send a math problem to an LLM with a clear solving prompt and print the model's answer.

## Install

```bash
pip install -r requirements.txt
```

For API usage, set:

```bash
set OPENAI_API_KEY=your_api_key
```

Optional API settings:

```bash
set OPENAI_BASE_URL=https://api.openai.com/v1
```

## API Example

```bash
python -m math_solver.cli api ^
  --model gpt-4.1-mini ^
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
