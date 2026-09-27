"""Evaluation orchestration and reporting.

:func:`evaluate` runs a set of metrics over every record of a dataset
with a given judge and returns an :class:`EvalReport` with per-sample
scores plus mean aggregates, renderable as JSON or Markdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .dataset import QARecord
from .judges import Judge
from .metrics import METRICS, MetricResult


@dataclass
class SampleResult:
    index: int
    question: str
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalReport:
    metric_names: list[str]
    samples: list[SampleResult] = field(default_factory=list)
    judge_name: str = ""

    @property
    def aggregates(self) -> dict[str, float]:
        """Mean score per metric across all samples."""
        agg: dict[str, float] = {}
        for name in self.metric_names:
            vals = [s.scores[name] for s in self.samples if name in s.scores]
            agg[name] = sum(vals) / len(vals) if vals else 0.0
        return agg

    def to_json(self) -> str:
        return json.dumps(
            {
                "judge": self.judge_name,
                "n_samples": len(self.samples),
                "metrics": self.metric_names,
                "aggregates": self.aggregates,
                "samples": [
                    {
                        "index": s.index,
                        "question": s.question,
                        "scores": s.scores,
                        "reasoning": s.reasoning,
                    }
                    for s in self.samples
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

    def to_markdown(self) -> str:
        lines = [
            "# RAG Evaluation Report",
            "",
            f"- Samples: {len(self.samples)}",
            f"- Judge: `{self.judge_name}`",
            f"- Metrics: {', '.join(self.metric_names)}",
            "",
            "## Aggregate scores (mean)",
            "",
            "| metric | mean |",
            "|---|---|",
        ]
        for name, mean in self.aggregates.items():
            lines.append(f"| {name} | {mean:.3f} |")
        lines += ["", "## Per-sample scores", ""]
        header = "| # | question | " + " | ".join(self.metric_names) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (2 + len(self.metric_names)))
        for s in self.samples:
            q = s.question if len(s.question) <= 60 else s.question[:57] + "..."
            row = " | ".join(f"{s.scores.get(m, float('nan')):.3f}" for m in self.metric_names)
            lines.append(f"| {s.index + 1} | {q} | {row} |")
        lines.append("")
        return "\n".join(lines)


def evaluate(
    dataset: list[QARecord],
    metric_names: list[str],
    judge: Judge,
) -> EvalReport:
    """Score every record in ``dataset`` on each of ``metric_names``.

    Raises:
        ValueError: if any metric name is not in :data:`rag_eval.metrics.METRICS`.
    """
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        raise ValueError(
            f"unknown metrics: {unknown}; available: {sorted(METRICS)}"
        )
    report = EvalReport(
        metric_names=list(metric_names), judge_name=type(judge).__name__
    )
    for i, record in enumerate(dataset):
        sample = SampleResult(index=i, question=record.question)
        for name in metric_names:
            result: MetricResult = METRICS[name](
                record.question,
                record.contexts,
                record.answer,
                record.ground_truth,
                judge,
            )
            sample.scores[name] = result.score
            sample.reasoning[name] = result.reasoning
        report.samples.append(sample)
    return report
