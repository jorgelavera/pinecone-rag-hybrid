# ingest-pipeline Specification

## Purpose

Load the Markdown corpus, split it into token-sized chunks, attach flat metadata (including the original text), embed with `text-embedding-3-small` (1536 dims), and upsert into a single Pinecone namespace.

## Requirements

### Requirement: Document Loading and Corpus Floor

The ingester SHALL load every `data/corpus/*.md` file and derive `document_id` as the filename stem. It SHALL abort with `CorpusTooSmallError` when fewer than 6 documents load.

#### Scenario: Corpus loaded

- GIVEN 8 files under `data/corpus/`
- WHEN ingest starts
- THEN 8 documents load with `document_id` values `pytest-basics`, `pytest-fixtures`, ...
- AND each `source` is `data/corpus/<file>.md`

#### Scenario: Corpus floor enforced

- GIVEN only 5 corpus files exist
- WHEN ingest starts
- THEN `CorpusTooSmallError` is raised before any embedding call or upsert

### Requirement: Token-Based Chunking

Chunking SHALL use `RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=600, chunk_overlap=80, add_start_index=True)`, so `chunk_size` counts tokens rather than characters. Target chunk size is 500–800 tokens.

#### Scenario: Token sizing

- GIVEN a 1,500-token document
- WHEN it is split
- THEN every chunk has `0 < token_count <= 800`
- AND the splitter was configured with the `cl100k_base` encoder, not the default `len`

#### Scenario: chunk_id format

- GIVEN two chunks from `pytest-fixtures`
- WHEN metadata is built
- THEN `chunk_id` values are `pytest-fixtures#000` and `pytest-fixtures#001`, with `chunk_index` 0 and 1

### Requirement: Metadata Schema

Each chunk SHALL carry 9 flat metadata fields — `document_id`, `chunk_id`, `chunk_index`, `source`, `page`, `section`, `category`, `tags`, `token_count` — plus `text` holding the original chunk text. `chunk_id` SHALL also be the vector id. Values SHALL be flat scalars or `list[str]`; no nested objects and no `None`.

#### Scenario: Schema valid

- GIVEN a Markdown chunk under heading `Fixture scopes`
- WHEN metadata is built
- THEN `page == 0` (int, not string), `section == "Fixture scopes"`, and `tags` is a `list[str]`
- AND `text` contains the verbatim chunk text

#### Scenario: Size guard

- GIVEN a misconfigured splitter produces a 3,000-token chunk
- WHEN metadata is validated
- THEN `MetadataTooLargeError` is raised because the UTF-8 JSON exceeds 40,000 bytes
- AND no upsert occurs for that batch

### Requirement: Embedding Model and Dimension Guard

Embeddings SHALL use `model="text-embedding-3-small"` with `dimensions=1536` set explicitly. Before any upsert the system SHALL assert every vector has length 1536.

#### Scenario: Dimension mismatch blocked

- GIVEN an embedding provider returns a 1024-dim vector
- WHEN ingest validates vectors
- THEN `DimensionMismatchError` is raised naming expected 1536 and actual 1024
- AND zero vectors are upserted

### Requirement: Batched Namespaced Upsert

Upserts SHALL target the single configured namespace in batches of at most 32 records (never one unbounded request) and SHALL be idempotent because the vector id equals `chunk_id`.

#### Scenario: Batching

- GIVEN 24 chunks to upsert
- WHEN ingest runs
- THEN `add_texts` is called with `batch_size=32` and one batch of 24 records

#### Scenario: Re-run is idempotent

- GIVEN the same corpus is ingested twice into the same namespace
- WHEN index stats are read afterwards
- THEN `vector_count` is 24 both times, because upsert by `chunk_id` overwrites

### Requirement: Preflight Without Credentials

`scripts/ingest.py` SHALL validate `PINECONE_API_KEY`, `OPENAI_API_KEY`, and `INDEX_NAME` before constructing any client. On missing values it SHALL print an actionable message and exit with code 2, with no traceback and no partial upsert.

#### Scenario: Missing keys

- GIVEN `OPENAI_API_KEY` is unset
- WHEN `python scripts/ingest.py` runs
- THEN the output names `OPENAI_API_KEY`, points to `.env.example`, and mentions `uv run pytest -q` for credential-free checks
- AND exit code is 2
