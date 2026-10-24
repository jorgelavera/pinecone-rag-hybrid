"""Scoring and report payload shape, with retrieval injected as plain id lists."""

import json
from pathlib import Path

import pytest

from rag.config import Settings
from rag.errors import GoldenSetError
from rag.evaluation import (
    GoldenQuestion,
    MetricsReport,
    QuestionScore,
    build_report_payload,
    load_calibration_set,
    score_retrieval,
)
from rag.metrics import normalized_precision, precision_ceiling

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "last_eval.json"
CALIBRATION_SET = ROOT / "data" / "calibration_set.json"
CORPUS_IDS = {
    "pytest-basics",
    "pytest-conftest",
    "pytest-coverage",
    "pytest-fixtures",
    "pytest-markers",
    "pytest-mocking",
    "pytest-parametrize",
    "pytest-plugins",
}

GOLDEN = [
    GoldenQuestion("q1", "How do fixture scopes work?", ("pytest-fixtures", "pytest-conftest")),
    GoldenQuestion("q2", "How do markers filter tests?", ("pytest-markers", "pytest-basics")),
]

SIX_RELEVANT = [GoldenQuestion("q1", "Six relevant documents?", tuple(f"doc{index}" for index in range(6)))]

SETTINGS = Settings(pinecone_api_key="test-key", openai_api_key="test-key", index_name="rag-pytest")


def test_score_retrieval_keeps_question_order_and_computes_values():
    report = score_retrieval(
        GOLDEN,
        [
            ["pytest-fixtures", "pytest-other"],
            ["pytest-markers", "pytest-mocking", "pytest-basics"],
        ],
        k=5,
    )

    assert [score.question_id for score in report.per_question] == ["q1", "q2"]

    first, second = report.per_question
    assert isinstance(first, QuestionScore)
    assert first.retrieved_document_ids == ("pytest-fixtures", "pytest-other")
    assert (first.hits, first.m) == (1, 2)
    assert first.precision == pytest.approx(0.2)
    assert first.recall == pytest.approx(0.5)

    assert (second.hits, second.m) == (2, 2)
    assert second.precision == pytest.approx(0.4)
    assert second.recall == pytest.approx(1.0)

    assert isinstance(report, MetricsReport)
    assert (report.k, report.n_questions) == (5, 2)
    assert report.precision == pytest.approx(0.3)
    assert report.recall == pytest.approx(0.75)
    assert report.ceiling == pytest.approx(0.4)
    assert report.normalized_precision == pytest.approx(0.75)


def test_score_retrieval_rejects_a_mismatched_retrieval_count():
    with pytest.raises(ValueError):
        score_retrieval(GOLDEN, [["pytest-fixtures"]], k=5)


def test_score_retrieval_rejects_an_empty_golden_set():
    with pytest.raises(ValueError):
        score_retrieval([], [], k=5)


def test_build_report_payload_matches_the_committed_artifact_shape():
    report = score_retrieval(SIX_RELEVANT, [["doc0", "doc1", "doc2", "doc3", "doc4"]], k=5)

    payload = build_report_payload(
        settings=SETTINGS,
        golden=SIX_RELEVANT,
        ablation={"vector": report, "bm25": report, "hybrid": report},
        generated_at="2026-10-22T23:00:00Z",
    )

    assert set(payload) == {
        "schema_version",
        "generated_at",
        "index_name",
        "namespace",
        "embedding_model",
        "pipeline_version",
        "k",
        "n_questions",
        "golden_set_path",
        "primary",
        "ablations",
        "per_question",
    }
    assert payload["schema_version"] == 1
    assert payload["generated_at"] == "2026-10-22T23:00:00Z"
    assert payload["index_name"] == "rag-pytest"
    assert payload["namespace"] == "rag-docs"
    assert payload["embedding_model"] == "text-embedding-3-small"
    assert payload["pipeline_version"] == "1"
    assert (payload["k"], payload["n_questions"]) == (5, 1)
    assert payload["golden_set_path"] == "data/golden_set.json"

    assert payload["primary"] == {
        "mode": "hybrid",
        "precision": 1.0,
        "recall": 0.833333,
        "ceiling": 1.0,
        "normalized_precision": 1.0,
    }
    assert list(payload["ablations"]) == ["vector", "bm25", "hybrid"]
    assert payload["ablations"]["hybrid"] == {
        "precision": 1.0,
        "recall": 0.833333,
        "ceiling": 1.0,
        "normalized_precision": 1.0,
    }

    (question,) = payload["per_question"]
    assert question["question_id"] == "q1"
    assert question["retrieved_document_ids"] == ["doc0", "doc1", "doc2", "doc3", "doc4"]
    assert (question["hits"], question["m"]) == (5, 6)
    assert question["precision"] == 1.0
    assert question["recall"] == 0.833333

    # Must survive a round-trip through disk for the committed artifact.
    assert json.loads(json.dumps(payload)) == payload


def test_build_report_payload_requires_the_hybrid_primary_mode():
    report = score_retrieval(SIX_RELEVANT, [["doc0"]], k=5)

    with pytest.raises(GoldenSetError):
        build_report_payload(
            settings=SETTINGS,
            golden=SIX_RELEVANT,
            ablation={"vector": report},
            generated_at="2026-10-22T23:00:00Z",
        )


def test_build_report_payload_rejects_a_report_for_a_different_golden_set():
    report = score_retrieval(SIX_RELEVANT, [["doc0"]], k=5)

    with pytest.raises(GoldenSetError):
        build_report_payload(
            settings=SETTINGS,
            golden=GOLDEN,
            ablation={"hybrid": report},
            generated_at="2026-10-22T23:00:00Z",
        )


# --------------------------------------------------------------------------- #
# Frozen calibration set (data/calibration_set.json)
# --------------------------------------------------------------------------- #


def test_load_calibration_set_returns_questions_and_fixed_retrieved_ids():
    questions, retrieved = load_calibration_set(CALIBRATION_SET, corpus_ids=CORPUS_IDS)

    assert [question.question_id for question in questions] == ["cal-1", "cal-2", "cal-3", "cal-4", "cal-5"]
    assert [len(question.relevant_document_ids) for question in questions] == [1, 1, 1, 3, 6]
    assert len(retrieved) == len(questions)
    assert all(isinstance(ids, tuple) and ids for ids in retrieved)

    # The frozen worked example: scoring the fixed lists reproduces the
    # documented aggregates exactly (design section 6).
    report = score_retrieval(questions, retrieved, k=5)
    assert report.precision == pytest.approx(0.36)
    assert report.recall == pytest.approx(0.833333, abs=1e-6)
    assert report.ceiling == pytest.approx(0.44)
    assert report.normalized_precision == pytest.approx(0.818182, abs=1e-6)
    assert [score.hits for score in report.per_question] == [1, 1, 1, 1, 5]


def test_load_calibration_set_rejects_unknown_retrieved_ids(tmp_path):
    path = tmp_path / "calibration_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "cal-x",
                    "question": "retrieves an id that is not in the corpus",
                    "relevant_document_ids": ["pytest-fixtures"],
                    "retrieved_document_ids": ["pytest-fixtures", "pytest-unknown"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldenSetError, match="pytest-unknown"):
        load_calibration_set(path, corpus_ids=CORPUS_IDS)


def test_load_calibration_set_requires_retrieved_ids_per_entry(tmp_path):
    path = tmp_path / "calibration_set.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "cal-x",
                    "question": "missing the fixed retrieved list",
                    "relevant_document_ids": ["pytest-fixtures"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(GoldenSetError, match="retrieved_document_ids"):
        load_calibration_set(path, corpus_ids=CORPUS_IDS)


# --------------------------------------------------------------------------- #
# The committed artifact must be internally consistent
# --------------------------------------------------------------------------- #


def _synthetic_golden_and_retrieval(payload: dict) -> tuple[list[GoldenQuestion], list[list[str]]]:
    """Rebuild (relevant, retrieved) pairs that reproduce each per_question row.

    A row stores hits/m and the retrieved ids but not the relevant set, so the
    relevant set is synthesized: the first ``hits`` distinct retrieved ids plus
    ``m - hits`` unseen stand-ins. Scoring that pair recomputes exactly the
    row's hits, precision and recall through the real metric code.
    """
    golden: list[GoldenQuestion] = []
    retrieved: list[list[str]] = []
    for row in payload["per_question"]:
        retrieved_ids = list(row["retrieved_document_ids"])
        distinct = list(dict.fromkeys(retrieved_ids))
        hits, m = row["hits"], row["m"]
        assert 0 <= hits <= min(m, len(distinct)), f"row {row['question_id']} has impossible hits/m"
        relevant = distinct[:hits] + [f"__unseen__{row['question_id']}_{index}" for index in range(m - hits)]
        golden.append(GoldenQuestion(row["question_id"], row["question_id"], tuple(relevant)))
        retrieved.append(retrieved_ids)
    return golden, retrieved


def test_committed_artifact_aggregates_match_its_per_question_rows():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    per_question = payload["per_question"]
    k = payload["k"]

    assert payload["n_questions"] == len(per_question)
    assert len(per_question) > 0

    golden, retrieved = _synthetic_golden_and_retrieval(payload)
    report = score_retrieval(golden, retrieved, k=k)

    primary = payload["primary"]
    assert round(report.precision, 6) == primary["precision"]
    assert round(report.recall, 6) == primary["recall"]
    assert round(report.ceiling, 6) == primary["ceiling"]
    assert round(report.normalized_precision, 6) == primary["normalized_precision"]

    for row, score in zip(per_question, report.per_question):
        assert row["hits"] == score.hits
        assert row["m"] == score.m
        assert row["precision"] == round(score.precision, 6)
        assert row["recall"] == round(score.recall, 6)
        # retrieved ids are document_ids, never chunk ids like "doc#000"
        assert all("#" not in doc_id for doc_id in row["retrieved_document_ids"])

    # per_question belongs to the primary (hybrid) mode.
    assert payload["ablations"]["hybrid"] == {
        key: primary[key] for key in ("precision", "recall", "ceiling", "normalized_precision")
    }

    # Ablations share the per-question m values: the ceiling is mode-independent
    # and normalized precision is derived, so both are fully recomputable.
    ceiling = precision_ceiling([row["m"] for row in per_question], k=k)
    assert list(payload["ablations"]) == ["vector", "bm25", "hybrid"]
    for mode in ("vector", "bm25", "hybrid"):
        ablation = payload["ablations"][mode]
        assert ablation["ceiling"] == round(ceiling, 6)
        assert ablation["normalized_precision"] == round(
            normalized_precision(ablation["precision"], ceiling), 6
        )
