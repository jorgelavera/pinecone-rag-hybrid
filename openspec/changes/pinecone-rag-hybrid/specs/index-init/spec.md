# index-init Specification

## Purpose

Guarantee a correctly configured Pinecone Serverless index exists before ingest or retrieval, using the legacy SDK surface available in `pinecone==7.3.0`.

## Requirements

### Requirement: Preflight Before Client Construction

`scripts/init_index.py` SHALL validate `PINECONE_API_KEY` and `INDEX_NAME` before constructing `Pinecone`. On failure it SHALL print remediation and exit with code 2.

#### Scenario: Missing API key

- GIVEN `PINECONE_API_KEY` is unset
- WHEN `python scripts/init_index.py` runs
- THEN the output names `PINECONE_API_KEY` and points to `.env.example`
- AND exit code is 2 and no network call is attempted

### Requirement: Verify Then Create (Legacy API)

The script SHALL call `pc.has_index(INDEX_NAME)`. When absent it SHALL create the index with `pc.create_index(name=INDEX_NAME, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))`. It SHALL NOT use `pc.indexes` or any schema API.

#### Scenario: Index created

- GIVEN no index named `rag-pytest` exists
- WHEN the script runs
- THEN exactly one `create_index` call occurs with `dimension=1536`, `metric="cosine"`, cloud `aws`, region `us-east-1`
- AND exit code is 0

#### Scenario: Index already exists

- GIVEN `rag-pytest` exists and is ready
- WHEN the script runs
- THEN zero `create_index` calls occur, stats are printed, and exit code is 0

### Requirement: Idempotency

Running the script twice in a row SHALL succeed both times and SHALL create at most one index.

#### Scenario: Second run is a no-op

- GIVEN the script created the index in a previous run
- WHEN it runs again
- THEN it reports that the index exists and calls `describe_index_stats()` once

### Requirement: Existing Index Configuration Guard

If the index exists with `dimension != 1536` or `metric != "cosine"`, the script SHALL abort with `IndexConfigMismatchError` naming expected and actual values, and SHALL NOT delete or recreate anything.

#### Scenario: Wrong dimension

- GIVEN an existing index with `dimension=768`
- WHEN the script runs
- THEN `IndexConfigMismatchError` reports expected 1536 and actual 768
- AND no delete or create call occurs

### Requirement: Namespace Visibility, No Namespace Creation

The script SHALL print `describe_index_stats().namespaces` so the target namespace is visible, and SHALL NOT create per-category namespaces.

#### Scenario: Namespace printed

- GIVEN the index holds vectors in `rag-docs`
- WHEN stats are printed
- THEN the output includes `rag-docs` with its `vector_count`
- AND no other namespace is created

### Requirement: Environment File Handling

Settings SHALL load `.env` via `python-dotenv` with `override=False`, so real environment variables win. `.env.example` SHALL be committed with `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME`, `PINECONE_NAMESPACE=rag-docs`, `ENSEMBLE_BM25_WEIGHT=0.4`, `ENSEMBLE_VECTOR_WEIGHT=0.6`; `.env` SHALL be gitignored.

#### Scenario: Real env wins

- GIVEN `.env` sets `INDEX_NAME=from-file` and the process environment sets `INDEX_NAME=from-shell`
- WHEN settings load
- THEN `settings.index_name == "from-shell"`

#### Scenario: Example file complete

- GIVEN a fresh clone
- WHEN `.env.example` is inspected
- THEN all six variable names above are present and no secret values are committed
