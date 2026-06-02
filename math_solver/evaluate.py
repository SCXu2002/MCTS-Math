"""Run the math solver on local benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .backends import ApiMathSolver, GenerationConfig, TransformersMathSolver
from .config import DEFAULT_API_CONFIG_PATH, load_api_config
from .prompt import build_math_prompt
from .tree_search import TreeSearchConfig, TreeSearchMathSolver


PROBLEM_KEYS = ("problem", "question", "prompt", "input")
ANSWER_KEYS = ("answer", "final_answer", "target", "label", "solution")
ID_KEYS = ("id", "idx", "problem_id", "question_id")
DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_name: str | None
    split: str


DATASET_SPECS = {
    "gsm8k": DatasetSpec(hf_path="gsm8k", hf_name="main", split="test"),
    "math": DatasetSpec(hf_path="HuggingFaceH4/MATH-500", hf_name=None, split="test"),
    "math500": DatasetSpec(hf_path="HuggingFaceH4/MATH-500", hf_name=None, split="test"),
    "aime": DatasetSpec(hf_path="HuggingFaceH4/aime_2024", hf_name=None, split="train"),
    "aime2024": DatasetSpec(hf_path="HuggingFaceH4/aime_2024", hf_name=None, split="train"),
}


@dataclass
class BenchmarkExample:
    id: str
    dataset: str
    problem: str
    answer: str | None = None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Line {line_number} in {path} is not a JSON object.")
                yield item


def read_json(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in ("data", "examples", "problems", "questions"):
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list, or an object with a data/examples/problems/questions list.")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} in {path} is not a JSON object.")
        yield item


def read_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(read_jsonl(path))
    if suffix == ".json":
        return list(read_json(path))
    if suffix == ".csv":
        return list(read_csv(path))
    raise ValueError(f"Unsupported dataset format: {path.suffix}. Use .jsonl, .json, or .csv.")


def first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        if key in lowered and lowered[key] not in (None, ""):
            return lowered[key]
    return None


def as_problem_text(value: Any) -> str:
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict) and item.get("content"):
                messages.append(str(item["content"]))
            else:
                messages.append(str(item))
        return "\n".join(messages)
    return str(value)


def records_to_examples(records: Iterable[dict[str, Any]], dataset: str) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []

    for index, record in enumerate(records, start=1):
        problem = first_value(record, PROBLEM_KEYS)
        if not problem:
            raise ValueError(
                f"Example {index} has no problem field. Expected one of: {', '.join(PROBLEM_KEYS)}."
            )

        example_id = first_value(record, ID_KEYS) or index
        answer = first_value(record, ANSWER_KEYS)
        examples.append(
            BenchmarkExample(
                id=str(example_id),
                dataset=dataset,
                problem=as_problem_text(problem),
                answer=None if answer is None else str(answer),
            )
        )

    return examples


def load_examples_from_file(path: Path, dataset: str) -> list[BenchmarkExample]:
    return records_to_examples(read_records(path), dataset)


def local_dataset_candidates(dataset: str, data_root: Path) -> list[Path]:
    names = (dataset, f"{dataset}_test", f"{dataset}-test", f"{dataset}_train", f"{dataset}-train")
    suffixes = (".jsonl", ".json", ".csv")
    return [data_root / f"{name}{suffix}" for name in names for suffix in suffixes]


def load_hf_dataset_from_disk(path: Path, dataset: str) -> list[BenchmarkExample]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install -r requirements.txt") from exc

    hf_dataset = load_from_disk(str(path))
    return records_to_examples(hf_dataset, dataset)


def load_hf_dataset(args: argparse.Namespace) -> list[BenchmarkExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install -r requirements.txt") from exc

    dataset_key = args.dataset.lower()
    spec = DATASET_SPECS.get(dataset_key)
    hf_path = args.hf_path or (spec.hf_path if spec else None)
    hf_name = args.hf_name if args.hf_name is not None else (spec.hf_name if spec else None)
    split = args.split or (spec.split if spec else "test")

    if not hf_path:
        raise ValueError(
            f"No default Hugging Face dataset is configured for {args.dataset!r}. "
            "Set --hf-path, and optionally --hf-name and --split."
        )

    cache_dir = Path(args.data_root) / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if hf_name:
        hf_dataset = load_dataset(hf_path, hf_name, split=split, cache_dir=str(cache_dir))
    else:
        hf_dataset = load_dataset(hf_path, split=split, cache_dir=str(cache_dir))

    disk_path = Path(args.data_root) / "hf_datasets" / args.dataset.lower()
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    if not disk_path.exists():
        hf_dataset.save_to_disk(str(disk_path))

    return records_to_examples(hf_dataset, args.dataset)


def load_benchmark_examples(args: argparse.Namespace) -> list[BenchmarkExample]:
    if args.data:
        return load_examples_from_file(Path(args.data), args.dataset)

    data_root = Path(args.data_root)
    disk_path = data_root / "hf_datasets" / args.dataset.lower()
    if disk_path.exists():
        print(f"Loading local Hugging Face dataset from {disk_path}")
        return load_hf_dataset_from_disk(disk_path, args.dataset)

    for path in local_dataset_candidates(args.dataset.lower(), data_root):
        if path.exists():
            print(f"Loading local dataset file from {path}")
            return load_examples_from_file(path, args.dataset)

    print(f"No local dataset found for {args.dataset}; downloading with Hugging Face datasets.")
    return load_hf_dataset(args)


def extract_boxed(text: str) -> str | None:
    match = re.search(r"\\boxed\{([^{}]+)\}", text)
    if match:
        return match.group(1)
    return None


def normalize_answer(text: str | None) -> str | None:
    if text is None:
        return None

    text = text.strip()

    # 对应 regexes_to_ignore: (?s).*#### 
    if "####" in text:
        text = re.sub(r"(?s).*#### ", "", text).strip()
        # 兼容 "####123" 没有空格的情况
        if "####" in text:
            text = text.split("####")[-1].strip()

    # 兼容 LaTeX boxed answer
    boxed = extract_boxed(text)
    if boxed:
        text = boxed.strip()

    # 对应 regexes_to_ignore
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\.$", "", text)

    # 额外清理常见 LaTeX 包裹
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\left|\\right", "", text)

    # GSM8K exact_match 通常比较最终数字，去掉空白更稳
    text = re.sub(r"\s+", "", text)

    return text.lower()


def build_solver(args: argparse.Namespace) -> ApiMathSolver | TransformersMathSolver:
    if args.backend == "local":
        if not args.model_path:
            raise ValueError("Local model path is required. Set --model-path.")

        solver = TransformersMathSolver(
            model_path=args.model_path,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            generation_config=GenerationConfig(
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
            ),
        )
        return wrap_search_solver(solver, args)

    api_config = load_api_config(args.config)
    model = args.model or api_config.get("model")
    api_key = args.api_key or api_config.get("api_key")
    base_url = args.base_url or api_config.get("base_url")

    if not model:
        raise ValueError("API model is required. Set --model or add model to api_config.json.")
    if not api_key:
        raise ValueError("API key is required. Set --api-key or add api_key to api_config.json.")

    solver = ApiMathSolver(
        model=model,
        api_key=api_key,
        base_url=base_url,
        generation_config=GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
        ),
        timeout=args.timeout,
    )
    return wrap_search_solver(solver, args)


def wrap_search_solver(
    solver: ApiMathSolver | TransformersMathSolver,
    args: argparse.Namespace,
) -> ApiMathSolver | TransformersMathSolver | TreeSearchMathSolver:
    if not args.search:
        return solver
    return TreeSearchMathSolver(
        solver,
        config=TreeSearchConfig(
            branches=args.search_branches,
            max_depth=args.search_depth,
        ),
        include_trace=args.show_search_trace,
    )


def output_path_for(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    return Path("results") / f"{args.dataset}_results.jsonl"


def run_evaluation(args: argparse.Namespace) -> None:
    examples = load_benchmark_examples(args)
    examples = examples[args.start :]
    if args.limit is not None:
        examples = examples[: args.limit]

    solver = build_solver(args)
    output_path = output_path_for(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    scored = 0

    with output_path.open("w", encoding="utf-8") as file:
        for offset, example in enumerate(examples, start=1):
            print(f"[{offset}/{len(examples)}] Solving {example.dataset}:{example.id}")
            result = {
                **asdict(example),
                "prediction": None,
                "normalized_answer": normalize_answer(example.answer),
                "normalized_prediction": None,
                "is_correct": None,
                "error": None,
            }

            try:
                prediction = solver.solve(build_math_prompt(example.problem))
                normalized_prediction = normalize_answer(prediction)
                normalized_answer = result["normalized_answer"]
                is_correct = (
                    normalized_answer is not None
                    and normalized_prediction is not None
                    and normalized_answer == normalized_prediction
                )

                result["prediction"] = prediction
                result["normalized_prediction"] = normalized_prediction
                result["is_correct"] = is_correct if normalized_answer is not None else None

                if normalized_answer is not None:
                    scored += 1
                    correct += int(is_correct)
            except Exception as exc:
                result["error"] = str(exc)

            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()

    print(f"Saved results to {output_path}")
    if scored:
        print(f"Accuracy: {correct}/{scored} = {correct / scored:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the API solver on a math benchmark.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. gsm8k, math, aime.")
    parser.add_argument("--backend", choices=("api", "local"), default="api", help="Solver backend. Defaults to api.")
    parser.add_argument("--data", default=None, help="Optional path to a .jsonl, .json, or .csv dataset file.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Local data/cache root. Defaults to data.")
    parser.add_argument("--hf-path", default=None, help="Optional Hugging Face dataset path override.")
    parser.add_argument("--hf-name", default=None, help="Optional Hugging Face dataset config/name override.")
    parser.add_argument("--split", default=None, help="Dataset split. Defaults depend on --dataset.")
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults to results/<dataset>_results.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to run.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start offset.")
    parser.add_argument("--config", default=None, help=f"Path to API config JSON. Defaults to {DEFAULT_API_CONFIG_PATH.name}.")
    parser.add_argument("--model", default=None, help="API model name. Defaults to api_config.json.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to api_config.json.")
    parser.add_argument("--base-url", default=None, help="API base URL. Defaults to api_config.json.")
    parser.add_argument("--model-path", default=None, help="Local model path or Hugging Face model id.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", help="auto, float16, bfloat16, or float32.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--search", action="store_true", help="Use greedy tree search instead of direct solving.")
    parser.add_argument("--search-branches", type=int, default=3, help="Number of branches to generate at each step.")
    parser.add_argument("--search-depth", type=int, default=4, help="Maximum tree-search depth.")
    parser.add_argument("--show-search-trace", action="store_true", help="Save branch scores and selected path in prediction.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
