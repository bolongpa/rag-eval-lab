"""Command-line interface.

Usage::

    rag-eval run --dataset data/sample_qa.jsonl --judge fake --report report.md
    python -m rag_eval.cli run --dataset data/sample_qa.jsonl --metrics faithfulness,answer_relevancy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import load_jsonl
from .judges import FakeJudge, OpenAICompatibleJudge
from .metrics import METRICS
from .runner import evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag-eval",
        description="LLM-as-judge evaluation harness for RAG pipelines.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Evaluate a JSONL dataset and write a report.")
    run.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL dataset (see data/sample_qa.jsonl for the schema).",
    )
    run.add_argument(
        "--metrics",
        default=",".join(METRICS),
        help=(
            "Comma-separated metric names (default: all). "
            f"Available: {', '.join(METRICS)}."
        ),
    )
    run.add_argument(
        "--judge",
        choices=["fake", "openai"],
        default="fake",
        help="'fake' uses a deterministic canned judge (no network); "
        "'openai' calls an OpenAI-compatible endpoint (needs LLM_API_KEY).",
    )
    run.add_argument(
        "--report",
        default=None,
        help="Write the Markdown report to this path.",
    )
    run.add_argument(
        "--json-report",
        default=None,
        help="Write the JSON report to this path.",
    )
    return p


def _make_judge(name: str):
    if name == "fake":
        return FakeJudge()
    return OpenAICompatibleJudge()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_names:
        print("error: --metrics must name at least one metric", file=sys.stderr)
        return 2

    dataset = load_jsonl(args.dataset)
    judge = _make_judge(args.judge)
    report = evaluate(dataset, metric_names, judge)

    print("\nRAG evaluation complete")
    print(f"  samples : {len(report.samples)}")
    print(f"  judge   : {report.judge_name}")
    print("  means   :")
    for name, mean in report.aggregates.items():
        print(f"    {name:<18} {mean:.3f}")

    if args.report:
        Path(args.report).write_text(report.to_markdown(), encoding="utf-8")
        print(f"  markdown: {args.report}")
    if args.json_report:
        Path(args.json_report).write_text(report.to_json(), encoding="utf-8")
        print(f"  json    : {args.json_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
