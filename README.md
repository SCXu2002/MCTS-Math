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
  "model": "gpt-5.4"
}
```

`api_config.json` is ignored by Git so your real key is not uploaded.

`base_url` can be either an OpenAI-compatible base URL such as
`https://api.openai.com/v1` or the full chat completions endpoint.

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

## Benchmark Evaluation

You can evaluate directly from Hugging Face datasets. The first run downloads and caches the dataset; later runs reuse
the local cache.

```bash
python -m math_solver.evaluate --dataset gsm8k
```

Built-in dataset shortcuts:

```text
gsm8k    -> gsm8k, main, test
math     -> HuggingFaceH4/MATH-500, test
aime     -> HuggingFaceH4/aime_2024, train
```

You can still use a local dataset file in `.jsonl`, `.json`, or `.csv` format. Each example should include a problem
field named `problem`, `question`, `prompt`, or `input`. Optional answer fields are `answer`, `final_answer`, `target`,
`label`, or `solution`.

Example JSONL:

```jsonl
{"id": "gsm8k-1", "question": "Natalia sold clips...", "answer": "#### 72"}
{"id": "aime-1", "problem": "Find the value of ...", "answer": "123"}
```

Run a small test:

```bash
python -m math_solver.evaluate ^
  --dataset gsm8k ^
  --limit 10
```

Results are saved to `results/<dataset>_results.jsonl`.

## Prompt

The solver asks the model to reason step by step and place the final answer in:

```text
\boxed{...}
```

This makes answer extraction easier in later iterations.
