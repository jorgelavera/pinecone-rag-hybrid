"""Scoring and report payload shape, with retrieval injected as plain id lists."""

import json

import pytest

from rag.config import Settings
from rag.errors import GoldenSetError
from rag.evaluation import (
    GoldenQuestion,
    MetricsReport,
    QuestionScore,
    build_report_payload,
    score_retrieval,
)

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
