"""Prompt construction for math problem solving."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MathPrompt:
    system: str
    user: str


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Solve problems rigorously, show the key reasoning steps, "
    "and verify arithmetic before giving the final answer."
)


def build_math_prompt(problem: str) -> MathPrompt:
    """Build a reusable math-solving prompt for chat or text-completion models."""
    user_prompt = f"""Solve the following math problem.

Requirements:
1. Reason step by step.
2. Keep the solution concise but complete.
3. Put the final answer in \\boxed{{...}}.

Problem:
{problem}

Solution:"""

    return MathPrompt(system=DEFAULT_SYSTEM_PROMPT, user=user_prompt)


def as_plain_text(prompt: MathPrompt) -> str:
    """Convert a chat-style prompt to plain text for local causal LMs."""
    return f"System: {prompt.system}\n\nUser: {prompt.user}\n\nAssistant:"
