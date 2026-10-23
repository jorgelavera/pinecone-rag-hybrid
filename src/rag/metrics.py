"""Pure retrieval metrics. Standard library only.

Conventions enforced here (and asserted in ``tests/test_metrics.py``):

* duplicate ids in the retrieved list are removed preserving first occurrence,
  *before* truncation to ``k``;
* the precision denominator is always ``k``, so thin retrieval is penalised;
* the ceiling clamps each question to ``min(m_q, k)``;
* aggregation is macro (per-question mean), never micro.

No third-party import belongs in this module: it is the anchor of the
credential-free tier.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def _require_positive_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _unique_preserving_first_occurrence(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def hits_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> int:
    """Number of distinct relevant documents inside the first ``k`` retrieved ids."""
    _require_positive_k(k)
    return sum(1 for doc_id in _unique_preserving_first_occurrence(retrieved_ids)[:k] if doc_id in relevant_ids)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """``|retrieved[:k] ∩ relevant| / k`` — the denominator never shrinks."""
    _require_positive_k(k)
    return hits_at_k(retrieved_ids, relevant_ids, k) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """``|retrieved[:k] ∩ relevant| / |relevant|``; undefined for an empty set."""
    _require_positive_k(k)
    if not relevant_ids:
        raise ValueError("recall_at_k is undefined for an empty relevant set (|relevant| = 0)")
    return hits_at_k(retrieved_ids, relevant_ids, k) / len(relevant_ids)


def macro_precision(
    pairs: Sequence[tuple[Sequence[str], Collection[str]]],
    k: int = 5,
) -> float:
    """Mean of per-question precision, weighting every question equally."""
    if not pairs:
        raise ValueError("macro_precision needs at least one question to average")
    return sum(precision_at_k(retrieved, relevant, k) for retrieved, relevant in pairs) / len(pairs)


def macro_recall(
    pairs: Sequence[tuple[Sequence[str], Collection[str]]],
    k: int = 5,
) -> float:
    """Mean of per-question recall, weighting every question equally."""
    if not pairs:
        raise ValueError("macro_recall needs at least one question to average")
    return sum(recall_at_k(retrieved, relevant, k) for retrieved, relevant in pairs) / len(pairs)


def precision_ceiling(relevant_sizes: Sequence[int], k: int = 5) -> float:
    """``(1/N) · Σ min(m_q, k) / k`` — the best precision reachable per question."""
    _require_positive_k(k)
    sizes = list(relevant_sizes)
    if not sizes:
        raise ValueError("precision_ceiling needs at least one question to average")
    return sum(min(size, k) / k for size in sizes) / len(sizes)


def normalized_precision(precision: float, ceiling: float) -> float:
    """Precision rescaled by the structural ceiling; ``0.0`` when it is unreachable."""
    if ceiling == 0:
        return 0.0
    return precision / ceiling
