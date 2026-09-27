"""rag-eval-lab: LLM-as-judge evaluation harness for RAG pipelines."""

from .dataset import QARecord, load_jsonl
from .judges import FakeJudge, Judge, OpenAICompatibleJudge
from .metrics import (
    METRICS,
    MetricResult,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    parse_score,
)
from .runner import EvalReport, SampleResult, evaluate

__all__ = [
    "QARecord",
    "load_jsonl",
    "Judge",
    "FakeJudge",
    "OpenAICompatibleJudge",
    "METRICS",
    "MetricResult",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "parse_score",
    "EvalReport",
    "SampleResult",
    "evaluate",
]

__version__ = "0.1.0"
