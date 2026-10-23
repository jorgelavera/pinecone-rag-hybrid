# evaluation-metrics Specification

## Purpose

Define the Golden Set schema, the pure retrieval metrics (Precision@k, Recall@k), macro-averaging over questions, and the ceiling/normalized reporting that makes scores interpretable.

## Requirements

### Requirement: Golden Set Schema

`data/golden_set.json` SHALL be a JSON array of objects with `question_id` (unique string), `question` (string), and `relevant_document_ids` (non-empty list of unique `document_id` values present in the corpus). The committed set SHALL contain at least 10 questions, each with at least 2 relevant documents.

#### Scenario: Valid entry accepted

- GIVEN `{"question_id":"q1","question":"...","relevant_document_ids":["pytest-fixtures","pytest-conftest"]}`
- WHEN the Golden Set loads
- THEN the entry is accepted and matching against `document_id` is exact, never substring-based
- AND an entry naming an id absent from the corpus raises `GoldenSetError` naming that id

#### Scenario: Single relevant doc rejected in committed set

- GIVEN the committed Golden Set contains a question with one relevant document
- WHEN the set is validated
- THEN validation fails, citing the `R@k = k·P@k` redundancy

### Requirement: precision_at_k

`precision_at_k(retrieved_ids, relevant_ids, k)` SHALL return `|retrieved_ids[:k] ∩ relevant_ids| / k`. Duplicate ids in `retrieved_ids` SHALL be removed preserving first occurrence. The denominator SHALL always be `k`, even when fewer than `k` documents were retrieved.

#### Scenario: Hand-computed value

- GIVEN `retrieved=[A,B,C,D,E]`, `relevant={A,C,F}`, `k=5`
- WHEN `precision_at_k` is called
- THEN the result is `0.4` (2 hits / 5)

#### Scenario: Thin retrieval is penalized

- GIVEN `retrieved=[A]`, `relevant={A,B}`, `k=5`
- WHEN `precision_at_k` is called
- THEN the result is `0.2` (1 / 5), not `1.0`

#### Scenario: Duplicates deduplicated

- GIVEN `retrieved=[A,A,B]`, `relevant={A,B}`, `k=5`
- WHEN `precision_at_k` is called
- THEN the result is `0.4` (2 hits / 5)

### Requirement: recall_at_k

`recall_at_k(retrieved_ids, relevant_ids, k)` SHALL return `|retrieved_ids[:k] ∩ relevant_ids| / |relevant_ids|`.

#### Scenario: Hand-computed value

- GIVEN `retrieved=[A,B,C,D,E]`, `relevant={A,C,F}`, `k=5`
- WHEN `recall_at_k` is called
- THEN the result is `0.6666667` (2 / 3)

#### Scenario: Empty relevant set rejected

- GIVEN `relevant` is empty
- WHEN `recall_at_k` is called
- THEN `ValueError` is raised because `|relevant_ids| = 0`
- AND `precision_at_k` still returns `0.0`, since a fixed denominator of `k` remains defined

### Requirement: Macro-Averaging Over Questions

Aggregation SHALL weight every question equally: `P@k = (1/N)·Σ_q P@k(q)` and `R@k = (1/N)·Σ_q R@k(q)`. An empty question list SHALL raise `ValueError`.

#### Scenario: Worked example

- GIVEN five questions with `(hits, m)` = (3,3), (1,2), (2,2), (1,1), (2,3)
- WHEN evaluated at `k=5`
- THEN `P@5 = 0.36` and `R@5 = 0.833333`
- AND per-question values are `0.6/1.0`, `0.2/0.5`, `0.4/1.0`, `0.2/1.0`, `0.4/0.666667`

### Requirement: Ceiling and Normalized Precision

Because Precision@k is structurally capped at `min(m_q,k)/k`, the system SHALL report `ceiling = (1/N)·Σ_q min(m_q,k)/k` and `normalized_precision = P@k / ceiling` alongside raw precision.

#### Scenario: Worked example ceiling

- GIVEN the same five questions with `m = 3,2,2,1,3`
- WHEN the report is built
- THEN `ceiling = 0.44`, raw `P@5 = 0.36`, and `normalized_precision = 0.818182`

### Requirement: Redundancy Identity With One Relevant Document

When every question has exactly one relevant document, `R@k = k·P@k`. A test SHALL assert this identity to document why Golden Sets use multiple relevant documents.

#### Scenario: Identity holds

- GIVEN `retrieved=[A]`, `relevant={A}`, `k=5`
- WHEN both metrics are computed
- THEN `P@5 = 0.2` and `R@5 = 1.0`
- AND `recall_at_k(...) == 5 * precision_at_k(...)`

### Requirement: Console Report and Offline Mode

`evaluate.py` SHALL print raw P@5, R@5, ceiling, normalized precision, and a three-row ablation (vector-only, BM25-only, hybrid); write `artifacts/last_eval.json`; and support `--offline` to reprint the last results with no credentials.

#### Scenario: Offline reprint

- GIVEN a valid `artifacts/last_eval.json` and no API keys
- WHEN `evaluate.py --offline` runs
- THEN it prints the stored P@5, R@5, ceiling, and normalized precision
- AND exits 0 without constructing any client
