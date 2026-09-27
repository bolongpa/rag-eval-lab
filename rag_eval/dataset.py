"""JSONL dataset loading and validation.

Record schema (one JSON object per line)::

    {
      "question": "How many vacation days do employees get?",
      "contexts": ["Handbook excerpt 1...", "Handbook excerpt 2..."],
      "ground_truth": "15 vacation days per year.",
      "answer": "Employees get 15 vacation days per year."
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QARecord:
    question: str
    contexts: list[str]
    ground_truth: str
    answer: str


def _validate_record(obj: object, lineno: int) -> QARecord:
    if not isinstance(obj, dict):
        raise ValueError(f"line {lineno}: record must be a JSON object, got {type(obj).__name__}")
    for key in ("question", "contexts", "ground_truth", "answer"):
        if key not in obj:
            raise ValueError(f"line {lineno}: missing required field {key!r}")
    question, contexts, ground_truth, answer = (
        obj["question"],
        obj["contexts"],
        obj["ground_truth"],
        obj["answer"],
    )
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"line {lineno}: 'question' must be a non-empty string")
    if (
        not isinstance(contexts, list)
        or not contexts
        or not all(isinstance(c, str) and c.strip() for c in contexts)
    ):
        raise ValueError(
            f"line {lineno}: 'contexts' must be a non-empty list of non-empty strings"
        )
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError(f"line {lineno}: 'ground_truth' must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"line {lineno}: 'answer' must be a non-empty string")
    return QARecord(
        question=question.strip(),
        contexts=[c.strip() for c in contexts],
        ground_truth=ground_truth.strip(),
        answer=answer.strip(),
    )


def load_jsonl(path: str | Path) -> list[QARecord]:
    """Load and validate a JSONL dataset.

    Blank lines are skipped. Raises ``ValueError`` with the offending
    line number on malformed JSON or schema violations, and if the file
    contains no records at all.
    """
    path = Path(path)
    records: list[QARecord] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno}: invalid JSON ({e})") from e
            records.append(_validate_record(obj, lineno))
    if not records:
        raise ValueError(f"{path}: no records found")
    return records
