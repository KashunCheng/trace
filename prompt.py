#!/usr/bin/env python3
"""
gen_trace_prompt.py

Given a source file and a target line number, generate a prompt for an LLM
to predict a reachability answer and a branch trace.

Usage:
    python gen_trace_prompt.py path/to/program.c --target-line 42 --file-name program.c
"""

import argparse
from pathlib import Path
from textwrap import indent


def add_line_numbers(source: str, width: int = 3) -> str:
    """
    Return the source code with 1-based line numbers padded to `width` digits.

    Example output line:
        001: int main() {
    """
    numbered_lines = []
    for i, line in enumerate(source.splitlines(), start=1):
        numbered_lines.append(f"{i:0{width}d}: {line}")
    return "\n".join(numbered_lines)


def build_trace_prompt(source: str, file_name: str, target_line: int) -> str:
    """
    Build the full LLM prompt for branch-trace prediction.

    Uses a simple line-based format:

    ```trace
    answer: reachable
    2 T
    3 T
    ```

    or

    ```trace
    answer: unreachable
    ```
    """
    numbered = add_line_numbers(source)

    prompt = f"""
You are a program analysis assistant. Your task is to reason about control flow in a program and output a structured branch trace that either reaches a specified target line, or an answer that the target is unreachable.

You will be given:
  1. A file name.
  2. The target line number in that file.
  3. The full program source with line numbers.

Your job:
  - Decide whether the target line is reachable or unreachable.
  - If you believe it is reachable, produce ONE candidate branch trace that could lead execution from the entry point to the target line.
  - If you believe it is unreachable, you may output an empty trace.

You MUST output your answer in a single block wrapped in ```trace fences, using the SIMPLE LINE-BASED FORMAT defined below.

---
TRACE FORMAT SCHEMA
---

The entire response must be:

<trace-output> ::= "```trace" <newline>
                   <answer-line>
                   {{ <newline> <step-line> }}
                   <newline> "```"

<answer-line> ::= "answer:" <space> <answer>

<answer> ::= "reachable" | "unreachable"

<step-line> ::= <line-number> <space> <branch-token>

<line-number> ::= <digit> {{ <digit> }}   ; decimal integer (e.g., 2, 15, 123)

<branch-token> ::= "T" | "F"

Semantics:
  - The first non-empty line inside the ```trace block MUST be the answer line. It specifies whether the target is reachable.
  - If <answer> is "reachable", then each subsequent non-empty line inside the block describes one branch decision along ONE possible execution path.
  - If <answer> is "unreachable", you may omit any additional lines.
  - <line-number> is the source line number of the branch.
  - <branch-token>:
      "T" means the branch condition is taken toward the target.
      "F" means the branch condition is not taken toward the target.
  - The order of <step-line> entries is the order in which branches are encountered along the execution path.

---
EXAMPLE PROGRAM AND VALID OUTPUT
---

Example program (for illustration):
```
001: int example(int x) {{
002:   if (x > 0) {{
003:     if (x % 2 == 0) {{
004:       return 1; // TARGET
005:     }} else {{
006:       return 2;
007:     }}
008:   }} else {{
009:     return 3;
010:   }}
011: }}
```

Suppose:
  - file = "example.c"
  - target_line = 4

A valid LLM output for a reachable path might be:

```trace
answer: reachable
2 T
3 T
````

Another valid output (if you believe line 4 is unreachable) would be:

```trace
answer: unreachable
```

---
REAL TASK INPUT
---

Below is the REAL program you must analyze.

File name: {file_name}
Target line: {target_line}

```
{numbered}
```

---
INSTRUCTIONS FOR YOUR OUTPUT
---

1. Think about the control flow of the numbered program.
2. Decide whether the target line {target_line} in file "{file_name}" is reachable or unreachable.
3. If reachable, choose one plausible path of branch decisions that could lead to executing that line.
4. Output ONLY a single ```trace block, conforming to the TRACE FORMAT SCHEMA above.
5. Do NOT include any explanations, comments, or additional text outside the ```trace block.
""".strip("\n")

    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an LLM prompt for branch-trace prediction."
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to the source file to analyze.",
    )
    parser.add_argument(
        "--target-line",
        type=int,
        required=True,
        help="Target line number to reach.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        help="Logical file name to report in the trace (default: basename of source_path).",
    )

    args = parser.parse_args()
    source_path: Path = args.source_path
    target_line: int = args.target_line
    file_name: str = args.file_name or source_path.name

    source_text = source_path.read_text(encoding="utf-8")
    prompt = build_trace_prompt(
        source_text, file_name=file_name, target_line=target_line
    )
    print(prompt)


if __name__ == "__main__":
    main()
