"""Golden Set loading and validation: exact membership, >=2 relevant docs, >=10 entries."""

import json
from pathlib import Path

import pytest

from rag.errors import GoldenSetError
from rag.evaluation import GoldenQuestion, load_golden_set

COMMITTED_SET = Path(__file__).resolve().parents[1] / "data" / "golden_set.json"
CORPUS_IDS = {"pytest-fixtures", "pytest-conftest", "pytest-markers", "pytest-mocking"}


def _entry(question_id: str = "q1", document_ids: tuple[str, ...] = ("pytest-fixtures", "pytest-conftest")):
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "relevant_document_ids": list(document_ids),
    }


def _write(tmp_path: Path, entries: list[dict], name: str = "golden_set.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_valid_entry_is_accepted(tmp_path):
    questions = load_golden_set(_write(tmp_path, [_entry()]), corpus_ids=CORPUS_IDS)

    assert questions == [
        GoldenQuestion(
            question_id="q1",
            question="Question q1?",
            relevant_document_ids=("pytest-fixtures", "pytest-conftest"),
        )
    ]


def test_unknown_corpus_id_is_rejected_and_named(tmp_path):
    path = _write(tmp_path, [_entry(document_ids=("pytest-fixtures", "pytest-unknown"))])

    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path, corpus_ids=CORPUS_IDS)

    assert "pytest-unknown" in str(excinfo.value)


def test_corpus_matching_is_exact_never_substring(tmp_path):
    # "pytest-fixtures-advanced" contains "pytest-fixtures" but is a different document.
    path = _write(tmp_path, [_entry(document_ids=("pytest-fixtures", "pytest-fixtures-advanced"))])

    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path, corpus_ids={"pytest-fixtures"})

    assert "pytest-fixtures-advanced" in str(excinfo.value)


def test_duplicate_question_id_is_rejected(tmp_path):
    path = _write(tmp_path, [_entry("q1"), _entry("q1", ("pytest-markers", "pytest-mocking"))])

    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path)

    assert "q1" in str(excinfo.value)


def test_single_relevant_document_is_rejected_citing_the_redundancy_identity(tmp_path):
    path = _write(tmp_path, [_entry(document_ids=("pytest-fixtures",))])

    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path)

    assert "R@k = k·P@k" in str(excinfo.value)


def test_duplicate_ids_inside_one_question_collapse_below_the_minimum(tmp_path):
    path = _write(tmp_path, [_entry(document_ids=("pytest-fixtures", "pytest-fixtures"))])

    with pytest.raises(GoldenSetError):
        load_golden_set(path)


def test_minimum_question_count_is_enforced(tmp_path):
    path = _write(tmp_path, [_entry("q1"), _entry("q2")])

    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path, min_questions=3)

    assert "3" in str(excinfo.value)


def test_missing_file_raises_a_golden_set_error(tmp_path):
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(tmp_path / "absent.json")

    assert "absent.json" in str(excinfo.value)


def test_malformed_entry_is_rejected(tmp_path):
    path = _write(tmp_path, [{"question_id": "q1", "question": "", "relevant_document_ids": ["a", "b"]}])

    with pytest.raises(GoldenSetError):
        load_golden_set(path)


def test_committed_golden_set_is_gradeable():
    questions = load_golden_set(COMMITTED_SET, min_questions=10)

    assert len(questions) >= 10
    assert all(len(question.relevant_document_ids) >= 2 for question in questions)
    assert len({question.question_id for question in questions}) == len(questions)
    assert all(question.question.strip() for question in questions)
