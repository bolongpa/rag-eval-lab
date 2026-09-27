"""Judge implementations for LLM-as-judge evaluation.

A ``Judge`` takes an evaluation prompt and returns the judge's verdict as
a dict with (at least) a ``"raw"`` key holding the model's free-text
response. Metrics in :mod:`rag_eval.metrics` then parse a numeric score
out of that text.

Prompts instruct models to emit a machine-readable verdict::

    SCORE: <float in [0, 1]>
    REASONING: <short explanation>

``FakeJudge`` emits that exact format with canned scores, so demos and
unit tests exercise the same parsing path as real model calls.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Protocol

import requests


class Judge(Protocol):
    """Anything that can turn an evaluation prompt into a verdict dict."""

    def score(self, prompt: str) -> dict:
        """Return a verdict dict with at least a ``"raw"`` text field."""
        ...


_METRIC_RE = re.compile(r"(?im)^\s*metric\s*:\s*([a-z_]+)")


def _metric_from_prompt(prompt: str) -> str:
    m = _METRIC_RE.search(prompt)
    return m.group(1) if m else "unknown"


@dataclass
class FakeJudge:
    """Deterministic stand-in for a real judge model.

    Returns canned per-metric scores in the exact ``SCORE:/REASONING:``
    format that real judges are asked to produce, so demos and tests
    exercise the full prompt -> parse pipeline without network access.
    """

    scores: dict = field(
        default_factory=lambda: {
            "faithfulness": 0.92,
            "answer_relevancy": 0.85,
            "context_precision": 0.78,
            "context_recall": 0.64,
        }
    )

    def score(self, prompt: str) -> dict:
        metric = _metric_from_prompt(prompt)
        value = float(self.scores.get(metric, 0.5))
        raw = (
            f"SCORE: {value}\n"
            f"REASONING: Canned deterministic score for '{metric}'. "
            "This is a fake judge; no model was called."
        )
        return {"raw": raw, "metric": metric}


@dataclass
class OpenAICompatibleJudge:
    """Judge backed by any OpenAI-compatible ``/chat/completions`` endpoint.

    Configuration comes from the environment so no secrets live in code:

    - ``LLM_BASE_URL`` (default ``https://api.openai.com/v1``)
    - ``LLM_API_KEY`` (required)
    - ``LLM_MODEL`` (default ``gpt-4o-mini``)

    Works with OpenAI, Azure OpenAI (via a compatible proxy), vLLM,
    Ollama, LiteLLM, OpenRouter, and similar gateways.
    """

    base_url: str = field(
        default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    timeout: int = 60

    def score(self, prompt: str) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Export it (and optionally LLM_BASE_URL / "
                "LLM_MODEL) before using OpenAICompatibleJudge."
            )
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise, impartial evaluator of "
                        "retrieval-augmented generation (RAG) systems. Follow the "
                        "user's evaluation instructions exactly and always end "
                        "your response with the required SCORE/REASONING format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return {"raw": content}
