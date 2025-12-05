"""Trace-generation reinforcement learning environment for Unsloth models.

This module builds a lightweight RL environment around the three verification
benchmarks included in this repository:

* ``xor.c`` verified by :mod:`xor_z3`
* ``dummy.c`` verified by :mod:`dummy_z3`
* ``control_temp.c`` verified by :mod:`control_temp_z3`

Each task produces a system/user prompt pair that instructs the model to emit a
branch trace inside a `````trace`` code fence. The emitted trace is parsed and
scored by the corresponding ``verify_trace`` helper which combines symbolic
execution and coverage information to deliver a dense reward.  The resulting
environment can be plugged into an Unsloth GRPO trainer by feeding the
``messages`` returned by :class:`TraceRLEnvironment` into the policy model and
passing the generated text back to :meth:`TraceRLEnvironment.evaluate`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .control_temp_z3 import verify_trace as verify_control_temp_trace
from .dummy_z3 import verify_trace as verify_dummy_trace
from .xor_z3 import verify_trace as verify_xor_trace

TraceDecision = Tuple[int, str]
VerifyTraceFn = Callable[[Sequence[TraceDecision], int], Mapping[str, object]]

TRACE_SYSTEM_PROMPT = dedent(
    """
    # HOW YOU SHOULD THINK AND ANSWER

    First draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.

    Your thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response.

    Reasoning effort: low.

    You are a program analysis assistant. Your task is to reason about control flow in a small C program and output a branch trace that either reaches a specified target line or states that the line is unreachable. Think before you answer, but NEVER include chain-of-thought in the final output.

    Always respond using a single ```trace fenced block that starts with ``answer: <reachable|unreachable>`` followed by zero or more lines that encode the branch decisions along one feasible path. Each branch line must be ``<line-number> <T|F>`` where the line number refers to the numbered listing provided in the user prompt and ``T``/``F`` tell whether that branch is taken toward the target line.

    If you believe the target line is unreachable, output only the answer line within the `````trace`` block. Otherwise, list the branch decisions in the order they are executed.

    Example program
    ```
    1: void foo() {
    2:     int a = 0;
    3:     if (a) {
    4:         a++;
    5:     }
    6:     if (a == 0) {
    7:         a--;
    8:     }
    9: }
    ```

    Example of a reachable answer:
    ```trace
    answer: reachable
    3 F
    6 T
    ```

    Example of an unreachable answer:
    ```trace
    answer: unreachable
    ```
    """
).strip()


@dataclass(frozen=True)
class TraceTask:
    """Single program-under-test along with its verification oracle."""

    name: str
    file_name: str
    target_line: int
    source: str
    verifier: VerifyTraceFn

    def build_user_prompt(self) -> str:
        numbered = add_line_numbers(self.source)
        return dedent(
            f"""File name: {self.file_name}
Target line: {self.target_line}

Program listing:
```
{numbered}
```
            """
        ).strip()

    def as_conversation(self, system_prompt: str = TRACE_SYSTEM_PROMPT) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.build_user_prompt()},
        ]


@dataclass(frozen=True)
class ParsedTrace:
    """Structured representation of a model-produced trace block."""

    answer: str
    decisions: List[TraceDecision]


@dataclass(frozen=True)
class TraceRewardResult:
    sat: bool
    reward: float
    reason: str
    parsed: Optional[ParsedTrace]


class TraceParseError(ValueError):
    """Raised when the model output cannot be parsed into a trace."""


TRACE_BLOCK_RE = re.compile(r"```trace\s*(.*?)```", re.DOTALL | re.IGNORECASE)
MIN_REWARD = -1.0


def add_line_numbers(source: str, width: int = 3) -> str:
    """Return the program with 1-based line numbers."""

    numbered_lines: List[str] = []
    for idx, line in enumerate(source.splitlines(), start=1):
        numbered_lines.append(f"{idx: {width}d}: {line}")
    return "\n".join(numbered_lines)


def parse_trace_block(raw_output: str) -> ParsedTrace:
    """Extract the ```trace block from the model output and parse it."""

    if '[/THINK]' in raw_output:
        raw_output = raw_output[raw_output.rfind('[/THINK]'):]

    match = TRACE_BLOCK_RE.search(raw_output)
    if not match:
        raise TraceParseError("missing ```trace block")

    contents = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    if not contents:
        raise TraceParseError("empty trace block")

    header = contents[0]
    if not header.lower().startswith("answer:"):
        raise TraceParseError("first line must be an answer declaration")

    answer = header.split(":", 1)[1].strip().lower()
    if answer not in {"reachable", "unreachable"}:
        raise TraceParseError(f"invalid answer token: {answer}")

    decisions: List[TraceDecision] = []
    for line in contents[1:]:
        parts = line.split()
        if len(parts) != 2:
            raise TraceParseError(f"invalid trace line: {line}")
        try:
            cond_line = int(parts[0])
        except ValueError as exc:  # pragma: no cover - defensive
            raise TraceParseError(f"invalid line number: {parts[0]}") from exc
        direction = parts[1].upper()
        if direction not in {"T", "F"}:
            raise TraceParseError(f"branch token must be T or F: {parts[1]}")
        decisions.append((cond_line, direction))

    return ParsedTrace(answer=answer, decisions=decisions)


def _default_tasks() -> List[TraceTask]:
    """Load the built-in benchmark tasks from disk."""

    root = Path(__file__).resolve().parent
    return [
        TraceTask(
            name="xor",
            file_name="xor.c",
            target_line=11,
            source=(root / "xor.c").read_text(encoding="utf-8"),
            verifier=verify_xor_trace,
        ),
        TraceTask(
            name="dummy",
            file_name="dummy.c",
            target_line=8,
            source=(root / "dummy.c").read_text(encoding="utf-8"),
            verifier=verify_dummy_trace,
        ),
        TraceTask(
            name="control_temp",
            file_name="control_temp.c",
            target_line=47,
            source=(root / "control_temp.c").read_text(encoding="utf-8"),
            verifier=verify_control_temp_trace,
        ),
    ]


def evaluate_trace(task: TraceTask, completion: str) -> TraceRewardResult:
    """Parse ``completion`` and score it against ``task``'s verifier."""

    try:
        parsed = parse_trace_block(completion)
    except TraceParseError as exc:
        return TraceRewardResult(
            sat=False,
            reward=MIN_REWARD * 2,
            reason=f"parse error: {exc}",
            parsed=None,
        )

    # TODO: Support unreachable
    if parsed.answer != "reachable":
        return TraceRewardResult(
            sat=False,
            reward=MIN_REWARD,
            reason="model predicted unreachable",
            parsed=parsed,
        )

    verify_result = task.verifier(parsed.decisions, task.target_line)
    sat = bool(verify_result.get("sat"))
    reward = float(verify_result.get("reward", MIN_REWARD))
    reason = str(verify_result.get("reason", ""))

    return TraceRewardResult(sat=sat, reward=reward, reason=reason, parsed=parsed)


class TraceRLEnvironment:
    """Utility wrapper for sampling tasks and scoring model outputs."""

    def __init__(
        self,
        tasks: Optional[Sequence[TraceTask]] = None,
        *,
        system_prompt: str = TRACE_SYSTEM_PROMPT,
    ) -> None:
        self._tasks: Tuple[TraceTask, ...] = tuple(tasks or _default_tasks())
        if not self._tasks:
            raise ValueError("at least one task is required")
        self.system_prompt = system_prompt

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._tasks)

    @property
    def tasks(self) -> Tuple[TraceTask, ...]:  # pragma: no cover - trivial
        return self._tasks

    def build_messages(self, task_index: int) -> List[Dict[str, str]]:
        task = self._tasks[task_index]
        return task.as_conversation(self.system_prompt)

    def sample(self, rng: Optional[random.Random] = None) -> Tuple[int, List[Dict[str, str]]]:
        """Return (task_index, chat_messages) for a random task."""

        rng = rng or random
        idx = rng.randrange(len(self._tasks))
        return idx, self.build_messages(idx)

    def evaluate(self, task_index: int, completion: str) -> TraceRewardResult:
        return evaluate_trace(self._tasks[task_index], completion)


def preview_environment(sample_seed: Optional[int] = None) -> None:
    """Quick manual smoke test for the environment."""

    env = TraceRLEnvironment()
    rng = random.Random(sample_seed)
    idx, messages = env.sample(rng)
    task = env.tasks[idx]
    print(f"Sampled task: {task.name}\n")
    print("System prompt:\n" + messages[0]["content"])
    print("\nUser prompt snippet:\n" + messages[1]["content"][:400] + "\n...")

    # Demonstrate scoring with a handcrafted trace for the xor task.
    if task.name == "xor":
        completion = dedent(
            """
            ```trace
            answer: reachable
            5 F
            7 T
            10 T
            ```
            """
        )
    elif task.name == "dummy":
        completion = dedent(
            """
            ```trace
            answer: reachable
            3 F
            5 F
            ```
            """
        )
    else:
        completion = dedent(
            """
            ```trace
            answer: reachable
            13 T
            15 T
            20 F
            23 T
            31 F
            35 T
            42 F
            44 F
            51 F
            57 T
            ```
            """
        )

    reward = env.evaluate(idx, completion)
    print(f"\nVerification result: sat={reward.sat}, reward={reward.reward:.4f}, reason={reward.reason}")


if __name__ == "__main__":
    preview_environment(sample_seed=0)
