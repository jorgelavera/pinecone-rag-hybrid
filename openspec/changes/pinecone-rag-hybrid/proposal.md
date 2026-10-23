# Proposal: pinecone-rag-hybrid

## Intent

Build a reproducible, gradeable cloud RAG system using Pinecone (Serverless) that demonstrates hybrid retrieval (vector + lexical BM25) with evaluation metrics, while enforcing strict reproducibility and credential-free grading paths. The deliverable is a course pre-entrega 4 repo that can be replicated end-to-end with explicit steps in README.

## Scope

### In Scope
- **Init script** (`scripts/init_index.py`): verify if index exists, create if not using Pinecone Serverless with `ServerlessSpec(cloud="aws", region="us-east-1")` (legacy API).
- **Ingest pipeline** (`scripts/ingest.py` + `src/rag/*`): load Markdown corpus from `data/corpus/`, chunk (~500-800 tokens via tiktoken), build metadata (9 flat fields, original text in `text`, `chunk_id` as vector id, `document_id`, size guard), upsert to Pinecone Serverless index in a single explicit namespace.
- **Hybrid retriever**: `RAGSystem` class wrapping `EnsembleRetriever` (BM25 + vector) with `[0.4 BM25, 0.6 vector]` (env-configurable, validated sum to 1.0), doc-level fusion (`id_key="document_id"`), over-fetch `k=20` per retriever, return **top-5 distinct docs**.
- **BM25 hydration**: reconstruct corpus from Pinecone at `RAGSystem.__init__` via `index.list()` → `index.fetch()` with disk cache (`artifacts/bm25_corpus.json`), fallback to local re-chunk.
- **Evaluation** (`evaluate.py`): compute **Precision@5 and Recall@5, macro-averaged over questions** on Golden Set with multiple relevant docs per question (matched by `document_id`). Print raw + ceiling + normalized precision; include ablation (vector-only/BM25-only/hybrid). Support `--offline`.
- **Golden Set** (`data/golden_set.json`): multiple relevant documents per question.
- **Config/env**: `.env.example` with `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME`, `PINECONE_NAMESPACE` (default `rag-docs`), ensemble weights. Config validation with actionable errors.
- **README.md**: Pinecone index replication steps, metrics summary (including redundancy/ceiling explanation), scaling notes.
- **Tests**: fake-based unit tests (credential-free), integration tests opt-in.

### Out of Scope
- **PDF ingest** unless trivially supported by existing pipeline (assignment states documents include PDF/Markdown/JSON but corpus is Markdown; do not expand scope now).
- **UI/hosted service**.
- **LLM answer generation beyond retrieval** — assignment focuses on RETRIEVAL and evaluation (metrics). No chat/QA generation step required.
- **Native Pinecone full_text_search/schema API** (blocked by SDK lock-in).
- **Per-category namespaces** (rejected; categories stay metadata).

## Approach

### Architecture (high level)
1. **Config** (`src/rag/config.py`): load `.env`, validate required keys, validate ensemble weights (exactly two, >0, sum≈1.0), produce `MissingConfigError` with remediation. Single source of truth for namespace.
2. **Chunking** (`src/rag/chunking.py`): use `RecursiveCharacterTextSplitter.from_tiktoken_encoder` (cl100k_base) for ~500-800 token chunks, build 9-field flat metadata, enforce 40KB guard, compute `document_id` from filename stem, `chunk_id` = vector id.
3. **Pinecone client** (`src/rag/pinecone_client.py`): isolate ALL Pinecone SDK calls. Use legacy API `pc.create_index(...)` with `ServerlessSpec(cloud="aws", region="us-east-1")`, `pc.has_index()`, `pc.Index()`. Pin `pinecone==7.3.0`.
4. **Ingest** (`scripts/ingest.py`): chunk → upsert via `PineconeVectorStore.add_texts` (batch_size safe) or batched upsert; assert dimension 1536 before upsert.
5. **Hydration** (`src/rag/hydration.py`): at `RAGSystem.__init__`, build BM25 corpus from Pinecone via `index.list()` → `index.fetch()` (1000 ids/request), cache to `artifacts/bm25_corpus.json` with cache key validation (`index_name`, namespace, pipeline_version, ceil(vector_count/1000)), fallback chain cache → Pinecone → local re-chunk → explicit error.
6. **Retrieval** (`src/rag/retrieval.py`): `RAGSystem` constructs BM25 + vector retrievers (per-retriever `k=20`), `EnsembleRetriever` with `weights=[0.4,0.6]`, `id_key="document_id"`, returns `results[:5]`. Namespace propagated everywhere.
7. **Metrics** (`src/rag/metrics.py`): pure functions `precision_at_k`, `recall_at_k`, macro averages, ceiling calculation, normalized precision. Doc-level matching by `document_id`.
8. **Evaluation** (`evaluate.py`): load Golden Set (multiple relevant docs), run retriever, compute metrics + ceiling + ablation, print table, write `artifacts/last_eval.json`, support `--offline`.

### Module layout (proposed)
```
pinecone-rag-hybrid/
├─ .env.example
├─ requirements.txt (pinned per SDK constraint)
├─ README.md
├─ data/
│  ├─ corpus/*.md (8 pytest files)
│  └─ golden_set.json
├─ src/rag/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ chunking.py
│  ├─ pinecone_client.py
│  ├─ hydration.py
│  ├─ retrieval.py
│  └─ metrics.py
├─ scripts/{init_index.py, ingest.py}
├─ evaluate.py
└─ tests/ (fake-based)
```

## Affected Files

| Area | Impact | Description |
|------|--------|-------------|
| `requirements.txt` | New | Pin `pinecone==7.3.0`, `langchain-pinecone==0.2.13`, `langchain-community==0.4.2`, `langchain-classic==1.0.8`, `langchain-core==1.7.0`, `langchain-openai==1.7.2`, `langchain-text-splitters==1.1.3`, `rank-bm25`, `python-dotenv`, `tiktoken`. |
| `.env.example` | New | `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME`, `PINECONE_NAMESPACE=rag-docs`, `ENSEMBLE_BM25_WEIGHT=0.4`, `ENSEMBLE_VECTOR_WEIGHT=0.6`. |
| `data/corpus/*.md` | New | 8 committed Markdown files on pytest (offline). |
| `data/golden_set.json` | New | Golden Set with multiple relevant docs per question, matched by `document_id`. |
| `src/rag/config.py` | New | Settings, validation, `MissingConfigError`. |
| `src/rag/chunking.py` | New | Token-based chunking, metadata schema, guards. |
| `src/rag/pinecone_client.py` | New | All Pinecone SDK calls (legacy API). |
| `src/rag/hydration.py` | New | Pinecone→BM25 corpus + cache + fallback. |
| `src/rag/retrieval.py` | New | `RAGSystem` with hybrid retrieval, doc-level fusion, top-5. |
| `src/rag/metrics.py` | New | Pure metric functions + ceiling/normalized precision. |
| `scripts/init_index.py` | New | Create index if not exists (Serverless). |
| `scripts/ingest.py` | New | Ingest pipeline. |
| `evaluate.py` | New | Evaluation + ablation + offline mode. |
| `README.md` | New/Modified | Replication steps, metrics summary, scaling notes. |
| `tests/*` | New | Fake-based unit tests + opt-in integration. |

## Replicability & No-Credential Grading

- **Preflight validation**: `config.load_settings()` validates all required env vars before constructing clients; missing vars raise `MissingConfigError` with actionable message listing missing names.
- **Graceful failure**: `ingest.py` and `evaluate.py` catch `MissingConfigError` and print remediation (copy `.env.example`, fill keys) with exit code 2. No tracebacks on missing creds.
- **Credential-free unit tests**: all tests use fakes (`FakeIndex`, `FakeEmbeddings`, `FakeVectorRetriever`). `uv run pytest -q` must pass without credentials/pyproject dependency on external services.
- **Offline evaluation**: `evaluate.py --offline` reads `artifacts/last_eval.json` and reprints metrics table.
- **Test doubles blueprint**: prototype verified end-to-end (ingest→hydrate→BM25→Ensemble→top-5→P@5/R@5) with zero credentials.
- **Pure metric functions**: `metrics.precision_at_k`/`recall_at_k` are pure and unit-testable.
- **Opt-in integration**: `@pytest.mark.integration` skipped unless keys present.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| SDK lock-in (pinecone<8, API 2025-04) | High | Pin versions explicitly; isolate all SDK calls in `pinecone_client.py` for future v10 migration. |
| BM25 hydration O(N), fetch returns values | High | Cache to disk with validation; document ceiling (~10^4-10^5 chunks); fallback chain. |
| Silent empty retrieval (namespace/text_key) | High | Single namespace constant used everywhere; name field `text` or pass `text_key` consistently; tests asserting propagation; explicit checks. |
| Upsert batch limit (batch_size=None sends all) | High | Use `PineconeVectorStore.add_texts` (batch_size=32) or explicit batching. |
| Metric misinterpretation (ceiling) | High | Compute and print ceiling + normalized precision; explain redundancy of P@5/R@5 when m=1 in README. |
| Doc-level fusion bias (long docs) | Med | Acceptable at course scale (24 chunks/8 docs); document tradeoff. |
| tiktoken cold download | Med | Keep out of credential-free unit tests via injected length function. |
| Corpus too small (<6 docs) | Med | Encode assertion `len(corpus_docs) >= 6` in tests. |
| Dimension/model mismatch | Med | Assert `len(vector)==1536`, set `model="text-embedding-3-small"`, `dimensions=1536` explicitly. |

## Rollback Plan

- Revert to previous commit if SDK issues arise; all Pinecone calls isolated so rollback is localized.
- If hydration fails, fallback to local re-chunk from committed corpus.
- If evaluation fails due to config, use `--offline` with cached results or run unit tests only.
- Disable integration tests via marker if credentials unavailable.

## Dependencies

- Pinecone Serverless (AWS us-east-1)
- OpenAI API (text-embedding-3-small)
- Python packages as pinned in requirements.txt
- Committed corpus and Golden Set (no network required for unit tests)

## Success Criteria

- [ ] `scripts/init_index.py` creates/verifies Serverless index with correct spec
- [ ] `scripts/ingest.py` ingests all corpus files with correct metadata schema
- [ ] `RAGSystem` returns top-5 distinct docs via hybrid retrieval with doc-level fusion
- [ ] `evaluate.py` prints Precision@5, Recall@5 (macro), ceiling, normalized precision, and ablation table
- [ ] Golden Set has multiple relevant docs per question; matching uses `document_id`
- [ ] README documents Pinecone replication steps and explains redundancy + ceiling
- [ ] `uv run pytest -q` passes with zero credentials (all fakes)
- [ ] Missing credentials produce actionable errors with exit code 2

## Review Workload Forecast

- **Estimated total changed lines**: ~800-900 lines (new files only; no existing code modified)
- **Breakdown**: config/chunking/pinecone_client/hydration/retrieval/metrics (~400), scripts/evaluate (~200), tests (~200), data/README/config files (~100)
- **400-line budget risk**: **High** (exceeds 400 lines)
- **Chained PRs recommended**: **Yes** (split by layers: foundation+tests, ingestion+hydration, retrieval+eval/README)
- **Decision needed before apply**: **Yes** (plan PR splits to respect review budget)
- **User review budget**: 800 lines (noted; still recommend chaining despite higher budget)
