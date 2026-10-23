# hybrid-retrieval Specification

## Purpose

Define `RAGSystem`: hybrid retrieval combining BM25 (lexical) and Pinecone vector search (semantic) via `EnsembleRetriever`, returning the top-5 distinct documents for a query. Retrieval only — no answer generation.

## Requirements

### Requirement: Hybrid Composition and Weights

`RAGSystem` SHALL build a `BM25Retriever` over the hydrated corpus and a Pinecone vector retriever over the configured namespace, each with `k=20`, and combine them in an `EnsembleRetriever` with `weights=[bm25_weight, vector_weight]` (default `[0.4, 0.6]`) and `id_key="document_id"`.

#### Scenario: Default weights

- GIVEN no weight overrides in `.env`
- WHEN `RAGSystem` is constructed
- THEN ensemble weights are `[0.4, 0.6]` and both retrievers use `k=20`
- AND `id_key == "document_id"`

#### Scenario: Invalid weights rejected

- GIVEN `ENSEMBLE_BM25_WEIGHT=4` and `ENSEMBLE_VECTOR_WEIGHT=6`
- WHEN settings are loaded
- THEN `ConfigError` is raised naming both values and stating weights MUST be positive and sum to 1.0 (±1e-6)
- AND no retriever is constructed

### Requirement: Top-5 Document-Level Result Contract

`RAGSystem.retrieve(query)` SHALL return at most 5 results ordered by fused score with distinct `document_id` values. Because `EnsembleRetriever` returns the union of both candidate lists, the system SHALL slice the fused output to `[:5]` itself.

#### Scenario: Ensemble over-returns

- GIVEN the ensemble yields 9 documents, 3 of which share one `document_id`
- WHEN `retrieve("fixture scope")` runs
- THEN exactly 5 results are returned
- AND their `document_id` values are distinct and in fused order

#### Scenario: Fewer than five distinct documents

- GIVEN the ensemble yields 3 distinct documents
- WHEN `retrieve(...)` runs
- THEN 3 results are returned, not padded and not duplicated

### Requirement: Result Payload

Each result SHALL expose `document_id`, the vector id (`chunk_id`), the original chunk `text`, and remaining metadata, so evaluation can match on `document_id` with no extra lookup.

#### Scenario: Payload fields

- GIVEN a retrieved chunk `pytest-fixtures#002` whose metadata has `document_id="pytest-fixtures"`
- WHEN a result is materialized
- THEN `result.document_id == "pytest-fixtures"`, `result.id == "pytest-fixtures#002"`, and `result.text` is the stored chunk text

### Requirement: Single-Namespace Consistency

Every read SHALL use the one namespace from configuration (`PINECONE_NAMESPACE`, default `rag-docs`). The value `None` SHALL NOT be passed to any Pinecone call, because `None` means the default empty namespace.

#### Scenario: Namespace propagated

- GIVEN `PINECONE_NAMESPACE=rag-docs`
- WHEN hydration and the vector retriever execute
- THEN the fake index records `"rag-docs"` for every `list`, `fetch`, and `similarity_search` call

#### Scenario: Default is not the empty namespace

- GIVEN `PINECONE_NAMESPACE` is unset
- WHEN settings load
- THEN the namespace is `"rag-docs"`, never `""`

### Requirement: BM25 Corpus Hydration

`RAGSystem.__init__` SHALL build the BM25 corpus from Pinecone (`index.list()` → `index.fetch()`, at most 1000 ids per fetch) and cache it at `artifacts/bm25_corpus.json`, keyed by `(index_name, namespace, pipeline_version, ceil(vector_count/1000))`.

#### Scenario: Cache hit

- GIVEN a cache file whose key matches current index stats
- WHEN `RAGSystem` initializes
- THEN zero `list` and `fetch` calls are made

#### Scenario: Paginated fetch

- GIVEN 1,500 chunk ids in the namespace and no valid cache
- WHEN hydration runs
- THEN exactly 2 `fetch` calls occur with 1,000 and 500 ids
- AND the corpus holds 1,500 documents with `text` popped from metadata into `page_content`

#### Scenario: Fallback chain

- GIVEN Pinecone raises a connection error
- WHEN hydration runs
- THEN it falls back to re-chunking the local corpus
- AND if that also fails, `HydrationError` is raised naming the cache path and remediation

### Requirement: Empty Retrieval Is an Error

The system SHALL raise `RetrievalEmptyError` naming the index and namespace when a query returns zero documents, instead of silently returning an empty list.

#### Scenario: Silent-empty trap surfaced

- GIVEN a namespace mismatch makes the vector retriever return nothing
- WHEN `retrieve("conftest")` runs
- THEN `RetrievalEmptyError` is raised and its message contains the index name and namespace
