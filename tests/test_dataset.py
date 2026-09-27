"""Tests for rag_eval.dataset (JSONL loading + validation)."""

import pytest

from rag_eval.dataset import load_jsonl


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _record(**overrides):
    rec = {
        "question": "What is the vacation policy?",
        "contexts": ["Employees accrue 15 vacation days per year."],
        "ground_truth": "15 vacation days per year.",
        "answer": "15 days per year.",
    }
    rec.update(overrides)
    return rec


def test_load_valid_jsonl(tmp_path):
    import json

    p = _write(
        tmp_path / "data.jsonl",
        [json.dumps(_record()), json.dumps(_record(question="Second?"))],
    )
    records = load_jsonl(p)
    assert len(records) == 2
    assert records[0].question == "What is the vacation policy?"
    assert records[1].contexts == ["Employees accrue 15 vacation days per year."]


def test_blank_lines_are_skipped(tmp_path):
    import json

    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(_record()) + "\n\n   \n", encoding="utf-8")
    assert len(load_jsonl(p)) == 1


def test_missing_field_raises_with_line_number(tmp_path):
    import json

    rec = _record()
    del rec["answer"]
    p = _write(tmp_path / "data.jsonl", [json.dumps(rec)])
    with pytest.raises(ValueError, match="line 1.*'answer'"):
        load_jsonl(p)


def test_contexts_must_be_nonempty_list_of_strings(tmp_path):
    import json

    for bad in ("just a string", [], [""], ["ok", 42]):
        p = _write(tmp_path / "data.jsonl", [json.dumps(_record(contexts=bad))])
        with pytest.raises(ValueError, match="line 1.*'contexts'"):
            load_jsonl(p)


def test_empty_question_raises(tmp_path):
    import json

    p = _write(tmp_path / "data.jsonl", [json.dumps(_record(question="  "))])
    with pytest.raises(ValueError, match="line 1.*'question'"):
        load_jsonl(p)


def test_invalid_json_raises(tmp_path):
    p = _write(tmp_path / "data.jsonl", ['{"question": broken'])
    with pytest.raises(ValueError, match="line 1.*invalid JSON"):
        load_jsonl(p)


def test_second_bad_record_reports_line_2(tmp_path):
    import json

    rec = _record()
    del rec["ground_truth"]
    p = _write(
        tmp_path / "data.jsonl",
        [json.dumps(_record()), json.dumps(rec)],
    )
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(p)


def test_empty_file_raises(tmp_path):
    p = _write(tmp_path / "data.jsonl", [])
    with pytest.raises(ValueError, match="no records"):
        load_jsonl(p)
