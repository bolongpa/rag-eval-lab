"""LLM-as-judge metrics for RAG evaluation.

Each metric is a function ``(question, contexts, answer, ground_truth, judge)
-> MetricResult``. It builds a focused evaluation prompt, sends it to a
:class:`~rag_eval.judges.Judge`, and parses a numeric score in ``[0, 1]``
out of the judge's free-text verdict.

The four metrics cover the two halves of RAG quality:

- *Generation quality* — is the answer correct and on-topic?
  (:func:`faithfulness`, :func:`answer_relevancy`)
- *Retrieval quality* — did we fetch the right chunks?
  (:func:`context_precision`, :func:`context_recall`)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .judges import Judge


@dataclass
class MetricResult:
    metric: str
    score: float  # always in [0, 1]
    reasoning: str


_SCORE_PATTERNS = [
    # The requested format: "SCORE: 0.85" (also tolerates "=").
    # The (?!\s*/) guard keeps "SCORE: 8/10" from matching here as a bare 8.
    (re.compile(r"(?im)^\s*score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)(?!\s*/)"), 1.0),
    # JSON-ish: {"score": 0.9}
    (re.compile(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)'), 1.0),
    # Fractions of 1: "0.75/1"
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*1(?:\.0+)?\b"), 1.0),
    # Fractions of 10: "8/10" -> 0.8
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b"), 10.0),
    # Percentages: "85%" -> 0.85
    (re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%"), 100.0),
    # Last resort: first bare number in the text.
    (re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\b"), 1.0),
]


def parse_score(raw: str) -> tuple[float, str]:
    """Extract ``(score, reasoning)`` from a judge's free-text verdict.

    Understands several common formats so small model-format drift does
    not break the pipeline. Values on 0-10 or 0-100 scales are
    normalized to 0-1; anything outside [0, 1] after normalization is
    clamped.

    Raises:
        ValueError: if the response is empty or contains no number.
    """
    if not raw or not raw.strip():
        raise ValueError("empty judge response: cannot parse a score")
    text = raw.strip()

    value: float | None = None
    for pattern, divisor in _SCORE_PATTERNS:
        m = pattern.search(text)
        if m:
            value = float(m.group(1)) / divisor
            break
    if value is None:  # unreachable: the bare-number pattern always matches digits
        raise ValueError(
            f"could not parse a numeric score from judge response: {text[:200]!r}"
        )
    value = min(1.0, max(0.0, value))
    return value, _extract_reasoning(text)


def _extract_reasoning(raw: str) -> str:
    m = re.search(r"(?im)^\s*reasoning\s*[:\-]\s*", raw)
    if m:
        return raw[m.end():].strip()[:1000]
    # Fallback: everything except the score line.
    lines = [
        ln
        for ln in raw.splitlines()
        if not re.match(r"(?i)^\s*score\s*[:=]", ln.strip())
    ]
    return "\n".join(lines).strip()[:1000]


def _contexts_block(contexts: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts))


def _run_judge(metric: str, prompt: str, judge: Judge) -> MetricResult:
    verdict = judge.score(prompt)
    raw = verdict.get("raw", "") if isinstance(verdict, dict) else str(verdict)
    score, reasoning = parse_score(raw)
    return MetricResult(metric=metric, score=score, reasoning=reasoning)


def faithfulness(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    judge: Judge,
) -> MetricResult:
    """Is every claim in the answer supported by the retrieved contexts?

    The judge splits the answer into atomic claims and checks each one
    against the contexts. Score = supported claims / total claims.
    This is the primary hallucination detector: a fluent answer that
    invents facts scores low even when it *sounds* right.
    """
    prompt = f"""METRIC: faithfulness

You are evaluating the FAITHFULNESS of an answer produced by a retrieval-augmented generation (RAG) system.

QUESTION:
{question.strip()}

RETRIEVED CONTEXTS:
{_contexts_block(contexts)}

ANSWER TO EVALUATE:
{answer.strip()}

INSTRUCTIONS:
1. Break the ANSWER into individual factual claims. Ignore purely stylistic filler (e.g. "Here is what I found:").
2. For each claim, decide whether it is directly supported by the RETRIEVED CONTEXTS. A claim that goes beyond, contradicts, or cannot be verified from the contexts counts as UNSUPPORTED.
3. Compute score = (number of supported claims) / (total number of claims). If the answer contains no factual claims (e.g. it abstains from answering), score 1.0.

Respond in exactly this format:
SCORE: <a float between 0 and 1>
REASONING: <e.g. "3 of 4 claims supported; unsupported: ...">
"""
    return _run_judge("faithfulness", prompt, judge)


def answer_relevancy(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    judge: Judge,
) -> MetricResult:
    """Does the answer directly and completely address the question?

    Penalizes irrelevant or tangential content, evasive non-answers, and
    partial answers. The ground truth is provided as reference only —
    an answer may phrase things differently and still score 1.0.
    """
    prompt = f"""METRIC: answer_relevancy

You are evaluating the ANSWER RELEVANCY of a response produced by a retrieval-augmented generation (RAG) system.

QUESTION:
{question.strip()}

ANSWER TO EVALUATE:
{answer.strip()}

REFERENCE (a human-written ground-truth answer; the evaluated answer may phrase things differently):
{ground_truth.strip()}

INSTRUCTIONS:
1. Judge whether the ANSWER directly addresses the QUESTION.
2. Penalize: irrelevant or tangential content, evasive non-answers, and answers that address only part of the question.
3. Score 1.0 = fully and directly answers the question; 0.0 = completely irrelevant or refuses to engage with the question.

Respond in exactly this format:
SCORE: <a float between 0 and 1>
REASONING: <one or two sentences>
"""
    return _run_judge("answer_relevancy", prompt, judge)


def context_precision(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    judge: Judge,
) -> MetricResult:
    """Of the retrieved chunks, how many are actually relevant?

    Score = relevant chunks / total chunks. Low precision means the
    retriever is fetching noise — wasted tokens, higher cost, and more
    surface for the generator to hallucinate from.
    """
    prompt = f"""METRIC: context_precision

You are evaluating the retrieval quality (CONTEXT PRECISION) of a retrieval-augmented generation (RAG) system.

QUESTION:
{question.strip()}

RETRIEVED CONTEXTS (in rank order):
{_contexts_block(contexts)}

INSTRUCTIONS:
1. For each context chunk, judge whether it contains information relevant to answering the QUESTION. A chunk that is only tangentially related counts as NOT relevant.
2. Compute score = (number of relevant chunks) / (total number of chunks).

Respond in exactly this format:
SCORE: <a float between 0 and 1>
REASONING: <e.g. "2 of 3 chunks relevant; chunk 3 is about parking and irrelevant">
"""
    return _run_judge("context_precision", prompt, judge)


def context_recall(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    judge: Judge,
) -> MetricResult:
    """Do the retrieved chunks cover everything needed to answer?

    The judge extracts the key facts from the ground-truth answer and
    checks each against the retrieved contexts. Score = covered facts /
    total facts. Low recall means the retriever is missing information
    the generator needs — the classic "right answer, wrong chunks"
    failure mode.
    """
    prompt = f"""METRIC: context_recall

You are evaluating the retrieval quality (CONTEXT RECALL) of a retrieval-augmented generation (RAG) system.

QUESTION:
{question.strip()}

GROUND-TRUTH ANSWER:
{ground_truth.strip()}

RETRIEVED CONTEXTS:
{_contexts_block(contexts)}

INSTRUCTIONS:
1. Identify the key facts in the GROUND-TRUTH ANSWER that are needed to answer the QUESTION.
2. For each key fact, check whether it can be found in (or reasonably inferred from) the RETRIEVED CONTEXTS.
3. Compute score = (key facts covered by the contexts) / (total key facts).

Respond in exactly this format:
SCORE: <a float between 0 and 1>
REASONING: <e.g. "2 of 3 key facts covered; missing: ...">
"""
    return _run_judge("context_recall", prompt, judge)


#: Registry of all available metrics, keyed by CLI name.
METRICS: dict[str, Callable[..., MetricResult]] = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}
