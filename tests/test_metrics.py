"""Pure metric tests: frozen calibration case plus deliberate regression traps.

Each test is written so a plausible-but-wrong implementation fails:

* precision using ``len(retrieved)`` as denominator  -> thin-retrieval test
* missing first-occurrence deduplication              -> duplicates test
* ceiling without ``min(m_q, k)`` clamping            -> m=6 calibration case
* micro instead of macro averaging                    -> macro-vs-micro test
"""

import pytest

from rag.metrics import (
    hits_at_k,
    macro_precision,
    macro_recall,
    normalized_precision,
    precision_at_k,
    precision_ceiling,
    recall_at_k,
)

K = 5


def _pair(relevant_count: int, hits: int) -> tuple[list[str], set[str]]:
    """Build a (retrieved, relevant) pair with exactly `hits` matches."""
    relevant = [f"d{index}" for index in range(relevant_count)]
    return relevant[:hits], set(relevant)


# ---------------------------------------------------------------------------
# FROZEN CALIBRATION CASE (design section 6). Derivation kept on purpose:
#   k = 5, N = 5, m_q = [1, 1, 1, 3, 6], hits_q = [1, 1, 1, 1, 5]
#   P@5     = mean(hits_q / k)       = (0.2 + 0.2 + 0.2 + 0.2 + 1.0) / 5 = 0.36
#   R@5     = mean(hits_q / m_q)     = (1 + 1 + 1 + 1/3 + 5/6) / 5       = 0.833333
#   ceiling = mean(min(m_q, k) / k)  = (0.2 + 0.2 + 0.2 + 0.6 + 1.0) / 5 = 0.44
#             a ceiling without min() would give 0.48, so clamping is proven
#   normalized_precision = 0.36 / 0.44                                   = 0.818182
# Note: the per-question precision of the first four questions is 0.2 (1 hit out
# of a fixed denominator of 5), which is what forces the 0.36 aggregate.
# ---------------------------------------------------------------------------
CALIBRATION = [_pair(1, 1), _pair(1, 1), _pair(1, 1), _pair(3, 1), _pair(6, 5)]
CALIBRATION_SIZES = [1, 1, 1, 3, 6]


def test_frozen_calibration_aggregates():
    assert macro_precision(CALIBRATION, k=K) == pytest.approx(0.36)
    assert macro_recall(CALIBRATION, k=K) == pytest.approx(0.833333, abs=1e-6)

    ceiling = precision_ceiling(CALIBRATION_SIZES, k=K)

    assert ceiling == pytest.approx(0.44)
    assert normalized_precision(0.36, ceiling) == pytest.approx(0.818182, abs=1e-6)


def test_frozen_calibration_per_question_values():
    precisions = [precision_at_k(retrieved, relevant, k=K) for retrieved, relevant in CALIBRATION]
    recalls = [recall_at_k(retrieved, relevant, k=K) for retrieved, relevant in CALIBRATION]

    assert precisions == pytest.approx([0.2, 0.2, 0.2, 0.2, 1.0])
    assert recalls == pytest.approx([1.0, 1.0, 1.0, 1 / 3, 5 / 6])


def test_ceiling_clamps_relevant_sizes_to_k():
    # m=6 at k=5 must count as 5/5, not 6/5.
    assert precision_ceiling([6], k=K) == pytest.approx(1.0)
    assert precision_ceiling([3], k=K) == pytest.approx(0.6)


def test_hand_computed_precision_and_recall():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "F"}

    assert precision_at_k(retrieved, relevant, k=K) == pytest.approx(0.4)
    assert recall_at_k(retrieved, relevant, k=K) == pytest.approx(2 / 3)


def test_thin_retrieval_uses_k_as_the_denominator():
    # A denominator of len(retrieved) would wrongly report 1.0.
    assert precision_at_k(["A"], {"A", "B"}, k=K) == pytest.approx(0.2)
    assert hits_at_k(["A"], {"A", "B"}, k=K) == 1


def test_duplicates_are_deduplicated_preserving_first_occurrence():
    # Counting duplicates would report 3 hits out of 5 -> 0.6.
    assert precision_at_k(["A", "A", "B"], {"A", "B"}, k=K) == pytest.approx(0.4)
    assert hits_at_k(["A", "A", "B"], {"A", "B"}, k=K) == 2


def test_deduplication_happens_before_truncation():
    # unique([A, A, B])[:2] = [A, B] -> 2 hits; truncating first would keep [A, A].
    assert hits_at_k(["A", "A", "B"], {"A", "B"}, k=2) == 2


def test_macro_averaging_is_not_micro_averaging():
    # Spec worked example: (hits, m) = (3,3), (1,2), (2,2), (1,1), (2,3) at k=5.
    pairs = [_pair(3, 3), _pair(2, 1), _pair(2, 2), _pair(1, 1), _pair(3, 2)]

    total_hits = sum(hits_at_k(retrieved, relevant, k=K) for retrieved, relevant in pairs)
    total_relevant = sum(len(relevant) for _, relevant in pairs)
    micro_recall = total_hits / total_relevant  # 9 / 11 = 0.818182

    assert (total_hits, total_relevant) == (9, 11)
    assert micro_recall == pytest.approx(0.818182, abs=1e-6)
    assert macro_recall(pairs, k=K) == pytest.approx(0.833333, abs=1e-6)
    assert macro_recall(pairs, k=K) != pytest.approx(micro_recall)
    assert macro_precision(pairs, k=K) == pytest.approx(0.36)


def test_macro_average_rejects_an_empty_question_list():
    with pytest.raises(ValueError):
        macro_precision([], k=K)
    with pytest.raises(ValueError):
        macro_recall([], k=K)


def test_recall_rejects_an_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["A"], set(), k=K)
    with pytest.raises(ValueError):
        macro_recall([(["A"], set())], k=K)


def test_precision_with_an_empty_relevant_set_is_zero():
    assert precision_at_k(["A", "B"], set(), k=K) == 0.0
    assert hits_at_k(["A", "B"], set(), k=K) == 0


def test_normalized_precision_with_zero_ceiling_is_zero():
    assert normalized_precision(0.0, 0.0) == 0.0
    assert normalized_precision(0.4, 0.0) == 0.0


def test_redundancy_identity_with_a_single_relevant_document():
    # Documents WHY the Golden Set requires >= 2 relevant documents per question.
    retrieved, relevant = ["A"], {"A"}

    precision = precision_at_k(retrieved, relevant, k=K)
    recall = recall_at_k(retrieved, relevant, k=K)

    assert precision == pytest.approx(0.2)
    assert recall == pytest.approx(1.0)
    assert recall == pytest.approx(K * precision)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, k=0)
    with pytest.raises(ValueError):
        precision_ceiling([1], k=0)
