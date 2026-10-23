"""Golden Set loading, retrieval scoring and report payloads.

Standard library only, and no retrieval client: the retriever is injected as
plain document-id sequences, so this module can be tested (and imported) with no
credentials and no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rag.errors import GoldenSetError
from rag.metrics import (
    hits_at_k,
    normalized_precision,
    precision_at_k,
    precision_ceiling,
    recall_at_k,
)

SCHEMA_VERSION = 1
DEFAULT_K = 5
PRIMARY_MODE = "hybrid"
REPORT_PRECISION = 6
REDUNDANCY_HINT = (
    "a question with one relevant document is redundant because R@k = k·P@k, "
    "so at least two relevant documents are required"
)


@dataclass(frozen=True)
class GoldenQuestion:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionScore:
    question_id: str
    retrieved_document_ids: tuple[str, ...]
    hits: int
    m: int
    precision: float
    recall: float


@dataclass(frozen=True)
class MetricsReport:
    k: int
    n_questions: int
    precision: float
    recall: float
    ceiling: float
    normalized_precision: float
    per_question: tuple[QuestionScore, ...]


def load_golden_set(
    path: Path,
    *,
    corpus_ids: set[str] | None = None,
    min_questions: int = 0,
    min_relevant: int = 2,
) -> list[GoldenQuestion]:
    """Parse and validate the Golden Set.

    ``corpus_ids`` enables exact (never substring) membership validation;
    ``min_questions`` enforces the committed-file floor during validation.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(
            f"Golden Set not found: {path}. Create data/golden_set.json with at least 10 questions."
        ) from exc
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"Golden Set is not valid JSON: {path} ({exc.msg})") from exc

    if not isinstance(raw, list):
        raise GoldenSetError(f"Golden Set must be a JSON array of questions, got {type(raw).__name__}")

    seen_ids: set[str] = set()
    questions = [
        _parse_question(entry, index, seen_ids, corpus_ids, min_relevant)
        for index, entry in enumerate(raw)
    ]
    if len(questions) < min_questions:
        raise GoldenSetError(
            f"Golden Set has {len(questions)} questions but at least {min_questions} are required"
        )
    return questions


def score_retrieval(
    golden: Sequence[GoldenQuestion],
    retrieved_ids: Sequence[Sequence[str]],
    k: int = DEFAULT_K,
) -> MetricsReport:
    """Score one retrieved id sequence per question, in Golden Set order."""
    if not golden:
        raise ValueError("score_retrieval needs at least one Golden Set question")
    if len(retrieved_ids) != len(golden):
        raise ValueError(
            f"expected one retrieved list per question ({len(golden)}), got {len(retrieved_ids)}"
        )

    per_question = []
    for question, retrieved in zip(golden, retrieved_ids):
        relevant = set(question.relevant_document_ids)
        per_question.append(
            QuestionScore(
                question_id=question.question_id,
                retrieved_document_ids=tuple(retrieved),
                hits=hits_at_k(retrieved, relevant, k=k),
                m=len(relevant),
                precision=precision_at_k(retrieved, relevant, k=k),
                recall=recall_at_k(retrieved, relevant, k=k),
            )
        )

    precision = sum(score.precision for score in per_question) / len(per_question)
    recall = sum(score.recall for score in per_question) / len(per_question)
    ceiling = precision_ceiling([score.m for score in per_question], k=k)

    return MetricsReport(
        k=k,
        n_questions=len(per_question),
        precision=precision,
        recall=recall,
        ceiling=ceiling,
        normalized_precision=normalized_precision(precision, ceiling),
        per_question=tuple(per_question),
    )


def build_report_payload(
    *,
    settings,
    golden: Sequence[GoldenQuestion],
    ablation: Mapping[str, MetricsReport],
    generated_at: str,
) -> dict[str, object]:
    """Build the ``artifacts/last_eval.json`` payload (JSON-serializable)."""
    primary = ablation.get(PRIMARY_MODE)
    if primary is None:
        raise GoldenSetError(f"ablation report must include the {PRIMARY_MODE!r} primary mode")
    if primary.n_questions != len(golden):
        raise GoldenSetError(
            f"primary report covers {primary.n_questions} questions but the Golden Set has {len(golden)}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "index_name": settings.index_name,
        "namespace": settings.namespace,
        "embedding_model": settings.embedding_model,
        "pipeline_version": settings.pipeline_version,
        "k": primary.k,
        "n_questions": primary.n_questions,
        "golden_set_path": _as_posix(settings.golden_set_path),
        "primary": {"mode": PRIMARY_MODE, **_report_numbers(primary)},
        "ablations": {mode: _report_numbers(report) for mode, report in ablation.items()},
        "per_question": [_question_payload(score) for score in primary.per_question],
    }


def _parse_question(
    entry: object,
    index: int,
    seen_ids: set[str],
    corpus_ids: set[str] | None,
    min_relevant: int,
) -> GoldenQuestion:
    if not isinstance(entry, dict):
        raise GoldenSetError(f"Golden Set entry #{index} must be a JSON object")

    question_id = entry.get("question_id")
    question = entry.get("question")
    relevant = entry.get("relevant_document_ids")

    if not isinstance(question_id, str) or not question_id.strip():
        raise GoldenSetError(f"Golden Set entry #{index} needs a non-empty 'question_id'")
    if question_id in seen_ids:
        raise GoldenSetError(f"Duplicate question_id {question_id!r} in the Golden Set")
    seen_ids.add(question_id)

    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"Golden Set question {question_id!r} needs a non-empty 'question'")
    if not isinstance(relevant, list) or not all(
        isinstance(document_id, str) and document_id.strip() for document_id in relevant
    ):
        raise GoldenSetError(
            f"Golden Set question {question_id!r} needs 'relevant_document_ids' "
            "as a list of non-empty strings"
        )

    document_ids = tuple(dict.fromkeys(relevant))
    if len(document_ids) < min_relevant:
        raise GoldenSetError(
            f"Golden Set question {question_id!r} lists {len(document_ids)} distinct relevant "
            f"document(s); at least {min_relevant} required ({REDUNDANCY_HINT})"
        )

    if corpus_ids is not None:
        unknown = [document_id for document_id in document_ids if document_id not in corpus_ids]
        if unknown:
            raise GoldenSetError(
                f"Golden Set question {question_id!r} references unknown document id(s): "
                f"{', '.join(unknown)}. Matching against the corpus is exact, never substring-based."
            )

    return GoldenQuestion(
        question_id=question_id,
        question=question,
        relevant_document_ids=document_ids,
    )


def _report_numbers(report: MetricsReport) -> dict[str, float]:
    return {
        "precision": round(report.precision, REPORT_PRECISION),
        "recall": round(report.recall, REPORT_PRECISION),
        "ceiling": round(report.ceiling, REPORT_PRECISION),
        "normalized_precision": round(report.normalized_precision, REPORT_PRECISION),
    }


def _question_payload(score: QuestionScore) -> dict[str, object]:
    return {
        "question_id": score.question_id,
        "retrieved_document_ids": list(score.retrieved_document_ids),
        "hits": score.hits,
        "m": score.m,
        "precision": round(score.precision, REPORT_PRECISION),
        "recall": round(score.recall, REPORT_PRECISION),
    }


def _as_posix(path: Path) -> str:
    """Report paths with forward slashes so the artifact is OS-independent."""
    return Path(path).as_posix()
