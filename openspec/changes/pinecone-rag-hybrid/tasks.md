# Tasks: pinecone-rag-hybrid

> Phase: tasks · Change: `pinecone-rag-hybrid` · Store: hybrid (OpenSpec + Engram)  
> Scope guard: RETRIEVAL + METRICS ONLY. No LLM generation, no `ChatOpenAI`, no LCEL chain.  
> SDK: `pinecone==7.3.0` legacy API. Module layout: `src/rag/*`, `scripts/{init_index,ingest}.py`, `evaluate.py`, `tests/`.  
> TDD: STRICT — test task BEFORE implementation task for every behavior. Runner: `uv run pytest -q`.  
> Chain strategy: `stacked-to-main`. Review budget: 800 lines. Design slices: PR1 foundation (~330), PR2 ingest (~350), PR3 retrieval+eval (~360).

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~1040 (new files only; no existing code modified) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR1 (foundation+pure/tests) → PR2 (ingest/index) → PR3 (retrieval+eval+README/artifacts) |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

Decision needed before apply: Yes  
Chained PRs recommended: Yes  
Chain strategy: stacked-to-main  
400-line budget risk: High  

### Suggested Work Units

| Unit | Goal | Likely PR | Focused test command | Runtime harness | Rollback boundary |
|------|------|-----------|----------------------|-----------------|-------------------|
| 1 | Foundation: config/env/errors/metrics/eval-pure + credential-free tests | PR1 (base: main) | `uv run pytest -q tests/test_metrics.py tests/test_config.py tests/test_golden_set.py tests/test_errors_taxonomy.py -v` | Pure stdlib only; no Pinecone/OpenAI keys. Uses fakes/injected env. | `src/rag/errors.py`, `src/rag/config.py`, `src/rag/metrics.py`, `src/rag/evaluation.py` (pure parts), `tests/test_metrics.py`, `tests/test_config.py`, `tests/test_golden_set.py`, plus `pyproject.toml`, `.env.example`, `.gitignore`. Revert slice commit. |
| 2 | Ingest: chunking/embeddings/pinecone_client/init/ingest + tests | PR2 (base: PR1 branch) | `uv run pytest -q tests/test_chunking.py tests/test_ingest.py tests/test_index_init.py -v` | Credential-free via fakes (`FakeEmbeddings`, `FakeIndex`). No live Pinecone/OpenAI calls. | `src/rag/chunking.py`, `src/rag/embeddings.py`, `src/rag/pinecone_client.py`, `scripts/init_index.py`, `scripts/ingest.py`, `data/corpus/*.md`, tests above. Isolated to SDK boundary (`pinecone_client.py`). Revert slice. |
| 3 | Retrieval+Eval: hydration/RAGSystem/evaluate + CLI/tests/README/artifacts | PR3 (base: PR2 branch) | `uv run pytest -q tests/test_hydration.py tests/test_retrieval.py tests/test_cli_exit_codes.py tests/test_evaluation.py -v` | Credential-free via fakes; `evaluate.py --offline` reprints committed `artifacts/last_eval.json`. No live keys needed. | `src/rag/hydration.py`, `src/rag/retrieval.py`, `evaluate.py`, `data/golden_set.json`, `artifacts/last_eval.json`, `README.md`, tests above. Fallback/cache disposable. Revert slice. |

---

## Slice 1: PR1 — Foundation, Config/Env, Pure Metrics/Eval (stacked to main) 
**Goal:** Establish credential-free foundation. All pure logic testable with zero network/keys.  
**Estimated lines:** ~330  
**Creds required:** NONE (unit/pure). Integration never in this slice.

### Phase 1.1: Project scaffolding & packaging
- [x] 1.1.1 **[SCAFFOLD][TEST-FIRST]** Create `tests/conftest.py` with fakes.  
  **Files:** `tests/conftest.py` (new)  
  **Test asserts:** fakes exist (`FakeIndex` with `list`/`fetch`/`describe_index_stats`/`upsert` records; `FakeEmbeddings` deterministic hash, configurable dim; `FakeVectorStore` with `.as_retriever()` returning `FakeRetriever`; `FakeSplitter` char-based). No imports of `openai`/`pinecone` at module scope.  
  **Impl later:** N/A (test/support)  
  **Creds:** No
- [x] 1.1.2 **[SCAFFOLD][IMPL]** Create `pyproject.toml`.  
  **Files:** `pyproject.toml` (new)  
  **Implements:** `[tool.pytest.ini_options] pythonpath=["src","."]`, registers `integration` marker. No build backend, deps in requirements.  
  **Creds:** No
- [x] 1.1.3 **[SCAFFOLD][IMPL]** Create `requirements.txt` pinned per SDK constraints.  
  **Files:** `requirements.txt` (new)  
  **Implements:** `pinecone==7.3.0`, `langchain-pinecone==0.2.13`, `langchain-community==0.4.2`, `langchain-classic==1.0.8`, `langchain-core==1.7.0`, `langchain-openai==1.7.2`, `langchain-text-splitters==1.1.3`, `rank-bm25`, `python-dotenv`, `tiktoken` (and test deps `pytest>=8.0.0`).  
  **Creds:** No
- [x] 1.1.4 **[SCAFFOLD][IMPL]** Create `.env.example` and `.gitignore`.  
  **Files:** `.env.example` (new), `.gitignore` (new)  
  **Implements:** `.env.example` has `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME`, `PINECONE_NAMESPACE=rag-docs`, `ENSEMBLE_BM25_WEIGHT=0.4`, `ENSEMBLE_VECTOR_WEIGHT=0.6` plus optional commented block (`EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `UPSERT_BATCH_SIZE`, `PINECONE_CLOUD`, `PINECONE_REGION`, `PIPELINE_VERSION`). `.gitignore` ignores `.env`, `.venv/`, `artifacts/bm25_corpus.json` but **MUST NOT** ignore `artifacts/last_eval.json` (committed).  
  **Creds:** No
- [x] 1.1.5 **[SCAFFOLD][IMPL]** Create package marker.  
  **Files:** `src/rag/__init__.py` (new, **EMPTY**)  
  **Implements:** empty package marker; no re-exports.  
  **Creds:** No

### Phase 1.2: Errors taxonomy (pure)
- [x] 1.2.1 **[ERRORS][TEST-FIRST]** `tests/test_errors_taxonomy.py`  
  **Files:** `tests/test_errors_taxonomy.py` (new)  
  **Test asserts:** `RagError` root; subclasses `ConfigError`, `MissingConfigError`, `CorpusTooSmallError`, `MetadataTooLargeError`, `DimensionMismatchError`, `IndexConfigMismatchError`, `GoldenSetError`, `HydrationError`, `RetrievalEmptyError` exist and inherit correctly.  
  **Creds:** No
- [x] 1.2.2 **[ERRORS][IMPL]** `src/rag/errors.py`  
  **Files:** `src/rag/errors.py` (new)  
  **Implements:** exception taxonomy exactly as specified (stdlib only).  
  **Creds:** No

### Phase 1.3: Config & env validation (pure + dotenv lazy)
- [x] 1.3.1 **[CONFIG][TEST-FIRST]** `tests/test_config.py`  
  **Files:** `tests/test_config.py` (new)  
  **Test asserts:** defaults (`namespace=rag-docs`, weights 0.4/0.6, dims 1536, chunk 600/80); shell env wins over `.env` (`load_dotenv(override=False)`); weights `[4,6]` → `ConfigError` naming both values stating positive and sum 1.0±1e-6; `""` namespace → `ConfigError`; missing keys aggregated by `require_credentials` (lists ALL missing) with `.env.example` context; `load_settings(env={...})` is hermetic (no dotenv).  
  **Regression-proof:** first-missing-only would fail (test checks all missing).  
  **Creds:** No
- [x] 1.3.2 **[CONFIG][IMPL]** `src/rag/config.py`  
  **Files:** `src/rag/config.py` (new)  
  **Implements:** frozen `Settings` dataclass, `load_settings(env=None|Mapping)`, `require_credentials(settings)` raising `MissingConfigError` listing ALL missing. `dotenv` imported lazily inside function when `env is None`. Validate weights (two, >0, `abs(sum-1.0)<=1e-6`), namespace non-empty, chunk sizes/overlap, batch_size 1..100, `embedding_dimensions==1536`.  
  **Creds:** No

### Phase 1.4: Pure metrics (stdlib only) — REGRESSION-PROOF
- [x] 1.4.1 **[METRICS][TEST-FIRST]** `tests/test_metrics.py` (frozen calibration + traps)  
  **Files:** `tests/test_metrics.py` (new)  
  **Test asserts (must fail on broken impl):**  
  - **Frozen calibration:** `k=5`, `m_q=[1,1,1,3,6]`, `hits_q=[1,1,1,1,5]` → `P@5==0.36`, `R@5≈0.833333`, `ceiling==0.44`, `normalized_precision≈0.818182`. Per-question `(1.0,1.0),(0.2,1.0),(0.2,1.0),(0.2,0.333333),(1.0,0.833333)`. Proves `min(m,k)` clamping (`m=6` would break naive `sum(m)/k`).  
  - **Thin retrieval penalized:** `retrieved=[A], relevant={A,B}, k=5` → `P@5==0.2` (denom `k`, not `len(retrieved)`).  
  - **Duplicates deduped:** `retrieved=[A,A,B], relevant={A,B}, k=5` → `P@5==0.4`.  
  - **Empty relevant:** `recall_at_k` raises `ValueError`; `precision_at_k` still returns `0.0`.  
  - **Macro empty:** raises `ValueError`.  
  - **Ceiling==0:** `normalized_precision` returns `0.0`.  
  - **Redundancy identity:** when all `m_q==1`, `R@k == k*P@k` (assert equality).  
  **Creds:** No
- [x] 1.4.2 **[METRICS][IMPL]** `src/rag/metrics.py`  
  **Files:** `src/rag/metrics.py` (new)  
  **Implements:** `precision_at_k(retrieved_ids, relevant_ids, k=5)`, `recall_at_k(...)` (ValueError if `len(relevant_ids)==0`), `hits_at_k`, `macro_precision`, `macro_recall` (ValueError if N==0), `precision_ceiling(relevant_sizes, k=5)`, `normalized_precision(precision, ceiling)` (returns `0.0` if `ceiling==0`). Deduplicate preserving first occurrence, fixed denom `k`, `min(m_q,k)`. **Stdlib only.**  
  **Creds:** No

### Phase 1.5: Evaluation (pure parts) + Golden Set
- [x] 1.5.1 **[GOLDEN][TEST-FIRST]** `tests/test_golden_set.py`  
  **Files:** `tests/test_golden_set.py` (new)  
  **Test asserts:** valid entry accepted, exact match (no substring) against corpus ids, unknown id → `GoldenSetError` naming id, duplicate question_id → `GoldenSetError`, single relevant → `GoldenSetError` citing `R@k = k·P@k`, <10 entries → validation fails (committed must have >=10 and >=2 each).  
  **Creds:** No
- [x] 1.5.2 **[EVAL][TEST-FIRST]** `tests/test_evaluation.py`  
  **Files:** `tests/test_evaluation.py` (new)  
  **Test asserts:** `score_retrieval` produces per-question `QuestionScore` with correct values/order, aggregates `MetricsReport` includes `ceiling` and `normalized_precision`, `build_report_payload` produces shape matching `artifacts/last_eval.json` (schema_version, generated_at, index_name, namespace, embedding_model, pipeline_version, k, n_questions, golden_set_path, primary, ablations in order `vector,bm25,hybrid`, per_question).  
  **Creds:** No
- [x] 1.5.3 **[EVAL][IMPL]** `src/rag/evaluation.py` (pure parts)  
  **Files:** `src/rag/evaluation.py` (new)  
  **Implements:** dataclasses `GoldenQuestion`, `QuestionScore`, `MetricsReport`; `load_golden_set(path, corpus_ids=None, min_questions=0, min_relevant=2)` with validations (unique ids, >=2, exact membership, >=10 for committed set); `score_retrieval(golden, retrieved_ids: Sequence[Sequence[str]], k=5)` producing full report; `build_report_payload(...)`. Stdlib only.  
  **Creds:** No
- [x] 1.5.4 **[DATA][IMPL]** Create committed data (foundation artifacts). **DONE (PR1 + PR3):** `data/golden_set.json` is committed (12 questions, >=2 relevant docs each, ids matching the 8 corpus stems of task 2.1.3). `artifacts/last_eval.json` landed in PR3 per design §11 and is **generated by real code**: `python evaluate.py --calibration` scores the frozen worked example in `data/calibration_set.json` through `score_retrieval` + `build_report_payload`; `tests/test_evaluation.py::test_committed_artifact_aggregates_match_its_per_question_rows` recomputes the aggregates from the per-question rows and proves they match `primary`/`ablations`.  
  **Files:** `data/golden_set.json` (new, committed), `data/calibration_set.json` (new, committed), `artifacts/last_eval.json` (new, **COMMITTED**).  
  **Implements:** `data/golden_set.json` has >=10 questions, each >=2 relevant `document_id`s present in corpus. `artifacts/last_eval.json` matches frozen calibration shape: `primary.precision 0.36, recall 0.833333, ceiling 0.44, normalized_precision 0.818182`, ablations in order `vector,bm25,hybrid`, includes `per_question` as in design.  
  **Creds:** No

**Slice 1 acceptance (PR1):** `uv run pytest -q tests/test_metrics.py tests/test_config.py tests/test_golden_set.py tests/test_errors_taxonomy.py tests/test_evaluation.py -v` green with NO keys set. No network imports at top level.

---


## Slice 2: PR2 — Ingest Pipeline & Index Init (stacked to PR1)
**Goal:** Chunking, embeddings wiring (lazy), Pinecone client isolation (legacy API), init script, ingest script, corpus files. Credential-free via fakes.  
**Estimated lines:** ~350  
**Creds required:** NONE for unit tests (all faked). Integration only if explicitly run with keys.

### Phase 2.1: Chunking (token-based) — REGRESSION-PROOF
- [x] 2.1.1 **[CHUNK][TEST-FIRST]** `tests/test_chunking.py`  
  **Files:** `tests/test_chunking.py` (new)  
  **Test asserts (must fail on broken impl):**  
  - **Token sizing:** 1500-token doc with tiktoken cl100k_base → every chunk `0 < token_count <= 800`, count 2–5 chunks (would be ~10 if using `len`).  
  - **Schema:** 10 keys total (`document_id,chunk_id,chunk_index,source,page,section,category,tags,token_count,text`); `page==0` int; `tags` is `list[str]`; `chunk_id` format `doc#000`, `chunk_index` matches; `text` verbatim.  
  - **Section:** nearest heading captured, `""` if none.  
  - **Size guard:** chunk producing >40KB UTF-8 JSON metadata → `MetadataTooLargeError`, no upsert path touched.  
  - **Corpus floor:** `load_corpus` with <6 docs → `CorpusTooSmallError`.  
  - **Metadata validation:** exact key set enforced, no `None`, no nested objects/bools.  
  **Creds:** No
- [x] 2.1.2 **[CHUNK][IMPL]** `src/rag/chunking.py`  
  **Files:** `src/rag/chunking.py` (new)  
  **Implements:** `Chunk` dataclass, `load_corpus(corpus_dir, min_documents=6)` → `CorpusTooSmallError`, `make_splitter(chunk_size=600, chunk_overlap=80)` uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with `encoding_name="cl100k_base"`, `add_start_index=True` (lazy import), `split_documents(docs, splitter=None, count_tokens=None)`, `build_metadata(...)` with all 9 fields + `text`, `validate_metadata(metadata, max_bytes=40000)` checking exact key set/types and raising `MetadataTooLargeError`. `tiktoken` imported lazily.  
  **Creds:** No
- [x] 2.1.3 **[DATA][IMPL]** Commit 8 pytest Markdown corpus files.  
  **Files:** `data/corpus/pytest-basics.md`, `data/corpus/pytest-fixtures.md`, `data/corpus/pytest-conftest.md`, `data/corpus/pytest-markers.md`, `data/corpus/pytest-parametrize.md`, `data/corpus/pytest-mocking.md`, `data/corpus/pytest-plugins.md`, `data/corpus/pytest-coverage.md` (8 files, new)  
  **Implements:** offline Markdown corpus; `document_id` = stem, sources `data/corpus/*.md`. Sufficient for >=6 floor and Golden Set membership.  
  **Creds:** No

### Phase 2.2: Embeddings wiring (lazy)
- [x] 2.2.1 **[EMBED][IMPL]** `src/rag/embeddings.py`  
  **Files:** `src/rag/embeddings.py` (new)  
  **Implements:** `build_embeddings(settings)` lazily imports `langchain_openai.OpenAIEmbeddings` (model `text-embedding-3-small`, `dimensions=1536`), `embed_texts(embeddings, texts, expected_dim=1536)` asserts every vector length == expected_dim else `DimensionMismatchError` naming expected/actual before upsert.  
  **Creds:** No (uses fakes in tests)

### Phase 2.3: Pinecone client isolation (legacy API)
- [x] 2.3.1 **[PINECONE][IMPL]** `src/rag/pinecone_client.py`  
  **Files:** `src/rag/pinecone_client.py` (new)  
  **Implements:** ALL Pinecone SDK calls isolated. `build_client(settings)` lazy imports `pinecone.Pinecone`. `verify_or_create_index(client, settings)` uses `pc.has_index(INDEX_NAME)`; if absent calls `pc.create_index(name=..., dimension=1536, metric="cosine", spec=ServerlessSpec(cloud=settings.cloud, region=settings.region))` from `pinecone`. If exists checks config (dimension/metric) → `IndexConfigMismatchError` naming expected/actual with **no delete/recreate**. `build_index(client, settings)` returns `client.Index(settings.index_name)`. `build_vector_store(index, embeddings, settings)` constructs `PineconeVectorStore` with `text_key="text"`, namespace from settings. **NO `Pinecone.indexes` usage.**  
  **Creds:** No (fakes in tests)

### Phase 2.4: Index init script
- [x] 2.4.1 **[INIT][TEST-FIRST]** `tests/test_index_init.py`  
  **Files:** `tests/test_index_init.py` (new)  
  **Test asserts:** create kwargs exact (`dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1")`); no-op when exists (zero create calls); wrong dimension/metric → `IndexConfigMismatchError` with no delete/create; namespace printed from `describe_index_stats()`; exit code 2 with missing keys naming var + `.env.example` + `uv run pytest -q` (no traceback).  
  **Creds:** No
- [x] 2.4.2 **[INIT][IMPL]** `scripts/init_index.py`  
  **Files:** `scripts/init_index.py` (new)  
  **Implements:** 3-line bootstrap `sys.path` so `python scripts/init_index.py` works without install. Validate keys before constructing Pinecone; `main()` wrapped once for MissingConfigError → exit 2 with remediation. Use legacy API, print namespaces/stats.  
  **Creds:** Required for live run; tests use fakes.

### Phase 2.5: Ingest script
- [x] 2.5.1 **[INGEST][TEST-FIRST]** `tests/test_ingest.py`  
  **Files:** `tests/test_ingest.py` (new)  
  **Test asserts:** corpus floor before embed; dims asserted before upsert (`DimensionMismatchError` blocks upsert); 24 records total → one `add_texts` call with `batch_size=32` (single batch); vector ids == `chunk_id`; idempotent re-upsert (same count).  
  **Regression-proof:** dims check after upsert would fail; per-record upsert would fail batching assertion.  
  **Creds:** No
- [x] 2.5.2 **[INGEST][IMPL]** `scripts/ingest.py`  
  **Files:** `scripts/ingest.py` (new)  
  **Implements:** 3-line bootstrap. Preflight keys (exit 2 if missing). Load corpus, enforce floor, chunk with tiktoken, build/validate metadata, embed with dim guard, upsert via `PineconeVectorStore.add_texts(batch_size=32)` or batched API targeting single namespace. `chunk_id` is vector id.  
  **Creds:** Required for live run; tests use fakes.

**Slice 2 acceptance (PR2):** `uv run pytest -q tests/test_chunking.py tests/test_ingest.py tests/test_index_init.py -v` green with NO keys. Chunking uses tiktoken cl100k_base. Index init uses legacy API only.

---


## Slice 3: PR3 — Hydration, Hybrid Retrieval, Evaluation CLI, README, Artifacts (stacked to PR2)
**Goal:** BM25 hydration with cache/fallback, RAGSystem hybrid (EnsembleRetriever) with doc-level fusion and top-5 distinct, evaluate.py with ablation and --offline, CLI exit codes, README, committed artifacts.  
**Estimated lines:** ~360  
**Creds required:** NONE for unit tests and `--offline`. Integration only if live keys provided.

### Phase 3.1: Hydration (Pinecone → BM25 corpus)
- [x] 3.1.1 **[HYDRATE][TEST-FIRST]** `tests/test_hydration.py`  
  **Files:** `tests/test_hydration.py` (new)  
  **Test asserts (must fail on broken impl):**  
  - **Cache hit:** valid cache key matches → 0 `list`/`fetch` calls.  
  - **Paginated fetch:** 1500 ids → exactly 2 `fetch` calls (1000, 500), namespace passed explicitly (never `None`). `text` popped from metadata to `Document.page_content`, metadata has 9 keys.  
  - **Cache key:** includes `(index_name, namespace, pipeline_version, ceil(vector_count/1000))` and vector_count checked.  
  - **Fallback:** Pinecone error → re-chunk local corpus; both fail → `HydrationError` naming cache path and remediation.  
  **Regression-proof:** streaming per page would fail count; missing namespace would fail; cache ignoring vector_count would fail.  
  **Creds:** No
- [x] 3.1.2 **[HYDRATE][IMPL]** `src/rag/hydration.py`  
  **Files:** `src/rag/hydration.py` (new)  
  **Implements:** `build_corpus(*, index, settings, cache_path=None, local_corpus_dir=None)` — get `describe_index_stats()` for ns vector_count; compute key; if cache valid return docs (0 fetch); if vcount==0 go LOCAL; collect all ids via `index.list(namespace=ns)` generator; batch fetch 1000 ids/call; for each vector pop `text` to `Document.page_content`; write cache; on any exception log warning and fall back to `chunking.split_documents(chunking.load_corpus(...))` → if fails raise `HydrationError(cache_path + remediation)`.  
  **Creds:** No

### Phase 3.2: Hybrid retrieval (RAGSystem)
- [x] 3.2.1 **[RETR][TEST-FIRST]** `tests/test_retrieval.py`  
  **Files:** `tests/test_retrieval.py` (new)  
  **Test asserts (must fail on broken impl):**  
  - **Weights/id_key:** default `[0.4,0.6]`, both retrievers `k=20`, `EnsembleRetriever` has `id_key="document_id"`.  
  - **Over-return + dedupe:** ensemble yields 9 docs with 3 sharing same `document_id` → `retrieve()` returns 5 distinct `document_id`s in fused order (system slices to `[:5]`, defensive dedupe by `document_id`).  
  - **<5 distinct:** returns exactly 3.  
  - **Payload:** `RetrievedDocument` has `document_id`, `id` (chunk_id), `text`, metadata.  
  - **Mode branches:** `mode="vector"`/`"bm25"` use single retriever directly (no `[1.0]` ensemble hiding weight validation); `mode="hybrid"` uses ensemble.  
  - **Empty → error:** zero results → `RetrievalEmptyError` naming index and namespace.  
  **Regression-proof:** removing `[:5]` or dedupe fails tests.  
  **Creds:** No
- [x] 3.2.2 **[RETR][IMPL]** `src/rag/retrieval.py`  
  **Files:** `src/rag/retrieval.py` (new)  
  **Implements:** `RetrievedDocument` dataclass; `RAGSystem` `__init__(settings, index=None, embeddings=None, vector_store=None, corpus=None, mode="hybrid")`. Build BM25 from hydrated corpus, vector retriever from vector store (namespace propagated). `mode=="hybrid"` → `EnsembleRetriever([bm25, vec], weights=[bm25_w, vec_w], id_key="document_id")`; else single retriever. `retrieve(query, k=5)`: run retriever → defensive dedupe by `document_id` preserving fused order → `[:k]` → if empty raise `RetrievalEmptyError(f"index={index_name} namespace={namespace}: 0 documents ...")`. Lazy imports for LangChain.  
  **Creds:** No

### Phase 3.3: CLI exit codes
- [x] 3.3.1 **[CLI][TEST-FIRST]** `tests/test_cli_exit_codes.py`  
  **Files:** `tests/test_cli_exit_codes.py` (new)  
  **Test asserts:** scrubbed env for all three CLIs (`scripts/init_index.py`, `scripts/ingest.py`, `evaluate.py`) → exit code 2; message names the missing var + `.env.example` + `uv run pytest -q`; no traceback.  
  **Creds:** No
- [x] 3.3.2 **[CLI][IMPL]** Ensure exit code 2 contract in all CLIs  
  **Files:** `scripts/init_index.py`, `scripts/ingest.py`, `evaluate.py` (update)  
  **Implements:** each `main()` wrapped once: `except MissingConfigError as e: print(f"Missing configuration: {e}"); print("Copy .env.example to .env and set the listed values."); print("Credential-free checks: uv run pytest -q"); sys.exit(2)`. `--offline` never calls `load_settings` with credential requirement and never constructs a client.  
  **Creds:** No

### Phase 3.4: Evaluate script
- [x] 3.4.1 **[EVALCLI][IMPL]** `evaluate.py`  
  **Files:** `evaluate.py` (new)  
  **Implements:** 3-line bootstrap. Load golden set, build RAGSystem for modes `vector,bm25,hybrid` (ablations), score each, compute ceiling/normalized, print table (raw P@5,R@5, ceiling, normalized, 3-row ablation), write `artifacts/last_eval.json` via `build_report_payload`. Support `--offline`: read and reprint committed `artifacts/last_eval.json` with no keys/clients.  
  **Creds:** Required for live run; `--offline` is credential-free.

### Phase 3.5: Documentation
- [x] 3.5.1 **[DOCS][IMPL]** `README.md`  
  **Files:** `README.md` (new)  
  **Implements:** Prereqs (Python 3.12, uv, Pinecone, OpenAI), install steps with `uv venv`/`uv pip`, copy `.env.example` to `.env`, `uv run python scripts/init_index.py`, `uv run python scripts/ingest.py`, `uv run python evaluate.py`, `uv run python evaluate.py --offline`, `uv run pytest -q`. Explain metrics (fixed-k denom, `R@k=k·P@k` redundancy when m==1, ceiling/normalized precision), doc-level fusion tradeoff, scale notes (~10^4–10^5 chunks), troubleshooting (exit code 2, namespace/text_key, stale cache delete or bump PIPELINE_VERSION, SDK pin rationale).  
  **Creds:** N/A

**Slice 3 acceptance (PR3):** `uv run pytest -q tests/test_hydration.py tests/test_retrieval.py tests/test_cli_exit_codes.py tests/test_evaluation.py -v` green. `evaluate.py --offline` prints frozen calibration values. All three CLIs have exit code 2 contract.

---


## Verification & Closure (run after all slices)
- [x] V.1 **[VERIFY][INTEGRATION-CHECK]** Credential-free full suite  
  **Command:** `uv run pytest -q` with **NO** `PINECONE_API_KEY`/`OPENAI_API_KEY` set  
  **Acceptance:** exit 0, all tests pass (unit + fake-based). No network calls attempted.  
  **Creds:** No
- [x] V.2 **[VERIFY][CALIBRATION]** Frozen calibration example  
  **Check:** `tests/test_metrics.py` passes the exact case `m_q=[1,1,1,3,6], hits_q=[1,1,1,1,5]` → `P@5=0.36, R@5=0.833333, ceiling=0.44, norm=0.818182`.  
  **Creds:** No
- [x] V.3 **[VERIFY][OFFLINE]** Offline evaluation  
  **Command:** `uv run python evaluate.py --offline` (no keys)  
  **Acceptance:** prints table with stored values matching `artifacts/last_eval.json` (primary `hybrid` gives 0.36/0.833333/0.44/0.818182), ablations in order `vector,bm25,hybrid`, exits 0.  
  **Creds:** No
- [x] V.4 **[VERIFY][CONTRACTS]** Guardrails check  
  **Acceptance:** `src/rag/__init__.py` empty; no `pc.indexes`; namespace never passed as `None`; `text_key="text"` in vector store; 9+text schema enforced; `chunk_id` is vector id; dedupe by `document_id` + `[:5]`; exit code 2 for all CLIs on missing creds.  
  **Creds:** No
- [x] V.5 **[VERIFY][GITIGNORE]** Artifacts policy  
  **Acceptance:** `.gitignore` ignores `artifacts/bm25_corpus.json` but **commits** `artifacts/last_eval.json`.  
  **Creds:** No

--- 

## Implementation Notes (enforce during apply)
- **Lazy imports:** `dotenv`, `tiktoken`, `langchain_*`, `pinecone`, `openai` only inside functions. `__init__.py` empty.
- **pythonpath:** rely on `pyproject.toml` `pythonpath=["src","."]` for tests; scripts keep 3-line `sys.path` bootstrap.
- **Legacy Pinecone:** `Pinecone(api_key=...)`, `pc.create_index(..., spec=ServerlessSpec(cloud="aws",region="us-east-1"))`, `pc.Index(name)`. No `Pinecone.indexes`.
- **VectorStore:** `PineconeVectorStore(index=..., embedding=..., text_key='text', namespace=..., pinecone_api_key=..., index_name=..., host=...)` as per verified signature.
- **Hydration:** collect-all-ids then batched fetch (<=1000/call), exactly 2 fetches for 1500 ids; namespace explicit.
- **Retrieval:** `id_key="document_id"`, defensive dedupe + `[:5]`, single retriever for non-hybrid modes (no `[1.0]`).
- **Metrics:** stdlib only, integer-exact hits, `min(m,k)`, macro averaging, ceiling/normalized.
- **TDD discipline:** every behavior has test-first task; regression-proof assertions specified where critical.
- **Chaining:** PR1→PR2→PR3 stacked to main as above.
- **Creds:** unit tests MUST pass with no credentials. Integration marked `@pytest.mark.integration` and auto-skipped if keys missing.

**Total tasks:** 1.1.1-1.5.4 (15) + 2.1.1-2.5.2 (9) + 3.1.1-3.5.1 (8) + V.1-V.5 (5) = **37 tasks** across 3 slices + verification.
