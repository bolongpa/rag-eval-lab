"""Tests for rag_eval.runner (aggregation + reports). No network calls."""

import json

import pytest

from rag_eval.dataset import QARecord
from rag_eval.judges import FakeJudge
from rag_eval.runner import evaluate


def _dataset():
    return [
        QARecord(
            question="How many vacation days?",
            contexts=["Employees accrue 15 vacation days per year."],
            ground_truth="15 vacation days per year.",
            answer="15 days per year.",
        ),
        QARecord(
            question="What is the 401(k) match?",
            contexts=["Acme matches 100% up to 4% of salary."],
            ground_truth="100% match up to 4% of salary.",
            answer="Acme matches 100% up to 6% of salary.",
        ),
    ]


def test_evaluate_runs_all_metrics_per_sample():
    report = evaluate(
        _dataset(),
        ["faithfulness", "answer_relevancy"],
        FakeJudge(),
    )
    assert len(report.samples) == 2
    for sample in report.samples:
        assert set(sample.scores) == {"faithfulness", "answer_relevancy"}
        assert set(sample.reasoning) == {"faithfulness", "answer_relevancy"}


def test_aggregates_are_means_of_sample_scores():
    report = evaluate(
        _dataset(),
        ["faithfulness", "context_recall"],
        FakeJudge(),
    )
    assert report.aggregates["faithfulness"] == pytest.approx(0.92)
    assert report.aggregates["context_recall"] == pytest.approx(0.64)


def test_unknown_metric_raises_helpful_error():
    with pytest.raises(ValueError, match="unknown metrics.*not_a_metric"):
        evaluate(_dataset(), ["not_a_metric"], FakeJudge())


def test_judge_name_recorded():
    report = evaluate(_dataset(), ["faithfulness"], FakeJudge())
    assert report.judge_name == "FakeJudge"


def test_json_report_roundtrips():
    report = evaluate(_dataset(), ["faithfulness", "context_precision"], FakeJudge())
    data = json.loads(report.to_json())
    assert data["n_samples"] == 2
    assert data["judge"] == "FakeJudge"
    assert data["aggregates"]["faithfulness"] == pytest.approx(0.92)
    assert len(data["samples"]) == 2
    assert data["samples"][0]["scores"]["context_precision"] == pytest.approx(0.78)


def test_markdown_report_contains_tables():
    report = evaluate(_dataset(), ["faithfulness", "answer_relevancy"], FakeJudge())
    md = report.to_markdown()
    assert "# RAG Evaluation Report" in md
    assert "| faithfulness | 0.920 |" in md
    assert "| answer_relevancy | 0.850 |" in md
    assert "How many vacation days?" in md
    assert "What is the 401(k) match?" in md


def test_empty_metric_list_gives_empty_aggregates():
    report = evaluate(_dataset(), [], FakeJudge())
    assert report.aggregates == {}
    assert len(report.samples) == 2
