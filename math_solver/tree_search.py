"""Greedy tree-search solver for math problems."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .prompt import DEFAULT_SYSTEM_PROMPT, MathPrompt


class TextGenerator(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Generate text from a system/user prompt pair."""


@dataclass(frozen=True)
class TreeSearchConfig:
    branches: int = 3
    max_depth: int = 4


@dataclass
class SearchStep:
    depth: int
    candidates: list[str]
    scores: list[float]
    selected_index: int


@dataclass
class TreeSearchResult:
    answer: str
    reasoning_path: list[str]
    steps: list[SearchStep] = field(default_factory=list)


class TreeSearchMathSolver:
    """Solve a problem by repeatedly expanding and scoring reasoning branches."""

    def __init__(
        self,
        generator: TextGenerator,
        config: TreeSearchConfig | None = None,
        include_trace: bool = False,
    ) -> None:
        self.generator = generator
        self.config = config or TreeSearchConfig()
        self.include_trace = include_trace

        if self.config.branches < 1:
            raise ValueError("Tree search branches must be at least 1.")
        if self.config.max_depth < 1:
            raise ValueError("Tree search max_depth must be at least 1.")

    def solve(self, prompt: MathPrompt) -> str:
        result = self.solve_with_trace(prompt)
        if not self.include_trace:
            return result.answer

        trace = ["\n\nSearch trace:"]
        for step in result.steps:
            trace.append(f"Depth {step.depth}: selected branch {step.selected_index + 1}")
            for index, (candidate, score) in enumerate(zip(step.candidates, step.scores), start=1):
                marker = "*" if index == step.selected_index + 1 else "-"
                trace.append(f"{marker} [{score:.1f}] {candidate}")
        return result.answer + "\n" + "\n".join(trace)

    def solve_with_trace(self, prompt: MathPrompt) -> TreeSearchResult:
        problem = _extract_problem(prompt.user)
        path: list[str] = []
        steps: list[SearchStep] = []

        for depth in range(1, self.config.max_depth + 1):
            candidates = self._generate_branches(problem, path)
            if not candidates:
                break

            scores = [self._score_branch(problem, path, candidate) for candidate in candidates]
            selected_index = max(range(len(candidates)), key=lambda index: scores[index])
            path.append(candidates[selected_index])
            steps.append(
                SearchStep(
                    depth=depth,
                    candidates=candidates,
                    scores=scores,
                    selected_index=selected_index,
                )
            )

            if "\\boxed{" in candidates[selected_index]:
                break

        return TreeSearchResult(
            answer=self._finalize_solution(problem, path),
            reasoning_path=path,
            steps=steps,
        )

    def _generate_branches(self, problem: str, path: list[str]) -> list[str]:
        user = f"""We are solving this math problem with tree search.

Problem:
{problem}

Current reasoning path:
{_format_path(path)}

Generate exactly {self.config.branches} different possible next reasoning steps.
Each branch should be a concise, mathematically meaningful continuation from the current path.
Do not repeat the same idea in different wording.
If the problem can now be finished, a branch may include the final answer in \\boxed{{...}}.

Return only a JSON array of strings."""

        raw = self.generator.complete(DEFAULT_SYSTEM_PROMPT, user)
        branches = _parse_string_list(raw)
        return branches[: self.config.branches]

    def _score_branch(self, problem: str, path: list[str], candidate: str) -> float:
        user = f"""Score this proposed next reasoning step for solving the math problem.

Problem:
{problem}

Current reasoning path:
{_format_path(path)}

Proposed next step:
{candidate}

Give a score from 0 to 100. Prefer steps that are correct, useful, non-circular, and move toward a verified final answer.
Return only JSON in this form: {{"score": 87, "reason": "short reason"}}"""

        raw = self.generator.complete(DEFAULT_SYSTEM_PROMPT, user)
        return _parse_score(raw)

    def _finalize_solution(self, problem: str, path: list[str]) -> str:
        user = f"""Solve the following math problem using the selected reasoning path.

Problem:
{problem}

Selected reasoning path:
{_format_path(path)}

Write a concise complete solution. Verify the arithmetic and put the final answer in \\boxed{{...}}."""

        return self.generator.complete(DEFAULT_SYSTEM_PROMPT, user)


def _extract_problem(user_prompt: str) -> str:
    marker = "Problem:"
    solution_marker = "\n\nSolution:"
    if marker not in user_prompt:
        return user_prompt.strip()
    problem = user_prompt.split(marker, 1)[1]
    if solution_marker in problem:
        problem = problem.split(solution_marker, 1)[0]
    return problem.strip()


def _format_path(path: list[str]) -> str:
    if not path:
        return "(root only; no reasoning steps selected yet)"
    return "\n".join(f"{index}. {step}" for index, step in enumerate(path, start=1))


def _parse_string_list(text: str) -> list[str]:
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass

    quoted = re.findall(r'"([^"\n]+)"', cleaned)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]

    items = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if line:
            items.append(line)
    return items


def _parse_score(text: str) -> float:
    cleaned = _strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "score" in data:
            return _clamp_score(float(data["score"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r"\b(?:score|评分)\b\D*(\d+(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", cleaned)
    if not match:
        return 0.0
    return _clamp_score(float(match.group(1)))


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()
