"""Tests for rag_eval.metrics (parsing + metric plumbing). No network calls."""

import pytest

from rag_eval.judges import FakeJudge
from rag_eval.metrics import (
    METRICS,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    parse_score,
)


# --- parse_score edge cases -------------------------------------------------


def test_parse_score_standard_format():
    score, reasoning = parse_score("SCORE: 0.85\nREASONING: mostly supported")
    assert score == pytest.approx(0.85)
    assert "mostly supported" in reasoning


def test_parse_score_tolerates_equals_and_case():
    score, _ = parse_score("score = 0.7")
    assert score == pytest.approx(0.7)


def test_parse_score_json_format():
    score, _ = parse_score('{"score": 0.9, "reasoning": "fine"}')
    assert score == pytest.approx(0.9)


def test_parse_score_fraction_of_one():
    score, _ = parse_score("The answer gets 0.75/1 for relevance.")
    assert score == pytest.approx(0.75)


def test_parse_score_fraction_of_ten_normalizes():
    score, _ = parse_score("SCORE: 8/10")
    assert score == pytest.approx(0.8)


def test_parse_score_percent_normalizes():
    score, _ = parse_score("REASONING: solid. 85%")
    assert score == pytest.approx(0.85)


def test_parse_score_clamps_out_of_range():
    assert parse_score("SCORE: 1.5\nREASONING: x")[0] == pytest.approx(1.0)
    assert parse_score("SCORE: 0\nREASONING: x")[0] == pytest.approx(0.0)


def test_parse_score_empty_raises():
    with pytest.raises(ValueError):
        parse_score("")


def test_parse_score_whitespace_only_raises():
    with pytest.raises(ValueError):
        parse_score("   \n  ")


def test_parse_score_no_number_raises():
    with pytest.raises(ValueError):
        parse_score("REASONING: no numeric verdict here")


def test_parse_score_reasoning_falls_back_to_body():
    # No explicit REASONING: line -> the rest of the text is kept.
    score, reasoning = parse_score("0.42 -- decent but incomplete")
    assert score == pytest.approx(0.42)
    assert "decent but incomplete" in reasoning


# --- metric plumbing with FakeJudge ----------------------------------------


def _sample_args():
    return (
        "What is the policy?",
        ["The policy is documented in section 3."],
        "The policy is in section 3.",
        "Section 3 documents the policy.",
    )


def test_all_four_metrics_registered():
    assert set(METRICS) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (faithfulness, 0.92),
        (answer_relevancy, 0.85),
        (context_precision, 0.78),
        (context_recall, 0.64),
    ],
)
def test_metrics_return_canned_fake_judge_scores(fn, expected):
    result = fn(*_sample_args(), FakeJudge())
    assert result.metric == fn.__name__
    assert result.score == pytest.approx(expected)
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning  # non-empty


def test_metric_scores_always_in_unit_interval():
    judge = FakeJudge(scores={"faithfulness": 5.0})  # misbehaving "model"
    result = faithfulness(*_sample_args(), judge)
    assert 0.0 <= result.score <= 1.0


def test_custom_fake_judge_scores_are_used():
    judge = FakeJudge(scores={"context_recall": 0.1})
    result = context_recall(*_sample_args(), judge)
    assert result.score == pytest.approx(0.1)
