"""Command line interface for the lightweight math solver."""

from __future__ import annotations

import argparse

from .backends import ApiMathSolver, GenerationConfig, TransformersMathSolver
from .config import DEFAULT_API_CONFIG_PATH, load_api_config
from .prompt import build_math_prompt


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=0.95)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve math problems with an LLM.")
    subparsers = parser.add_subparsers(dest="backend", required=True)

    api_parser = subparsers.add_parser("api", help="Use an OpenAI-compatible API.")
    api_parser.add_argument("--model", default=None, help="API model name. Defaults to api_config.json.")
    api_parser.add_argument("--problem", required=True, help="Math problem to solve.")
    api_parser.add_argument("--api-key", default=None, help="API key. Defaults to api_config.json.")
    api_parser.add_argument("--base-url", default=None, help="API base URL. Defaults to api_config.json.")
    api_parser.add_argument(
        "--config",
        default=None,
        help=f"Path to API config JSON. Defaults to {DEFAULT_API_CONFIG_PATH.name}.",
    )
    add_generation_args(api_parser)

    local_parser = subparsers.add_parser("local", help="Use a local transformers model.")
    local_parser.add_argument("--model-path", required=True, help="Local model path or Hugging Face model id.")
    local_parser.add_argument("--problem", required=True, help="Math problem to solve.")
    local_parser.add_argument("--device-map", default="auto")
    local_parser.add_argument("--torch-dtype", default="auto", help="auto, float16, bfloat16, or float32.")
    add_generation_args(local_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    generation_config = GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )
    prompt = build_math_prompt(args.problem)

    if args.backend == "api":
        api_config = load_api_config(args.config)
        model = args.model or api_config.get("model")
        api_key = args.api_key or api_config.get("api_key")
        base_url = args.base_url or api_config.get("base_url")

        if not model:
            parser.error("API model is required. Set --model or add model to api_config.json.")
        if not api_key:
            parser.error("API key is required. Set --api-key or add api_key to api_config.json.")

        solver = ApiMathSolver(
            model=model,
            api_key=api_key,
            base_url=base_url,
            generation_config=generation_config,
        )
    elif args.backend == "local":
        solver = TransformersMathSolver(
            model_path=args.model_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            generation_config=generation_config,
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    print(solver.solve(prompt))


if __name__ == "__main__":
    main()
