# Design: pinecone-rag-hybrid

> Phase: design · Change: `pinecone-rag-hybrid` · Store: hybrid (OpenSpec + Engram)
> Scope guard: RETRIEVAL + METRICS ONLY. No LLM generation, no `ChatOpenAI`, no LCEL chain.
> SDK: `pinecone==7.3.0` **legacy** API (`pc.create_index(...)` + `pc.Index(name)`), verified installed.
> Size note: ~2.7k words, over the 800-word design budget on purpose — the four flagged risks require exact signatures, key lists, and shapes to be machine-actionable. The 800-line *code* review budget (§11) is unaffected.

## 1. Module layout

`src/rag/` package (not flat, not installed). Rationale: keeps pure logic (`metrics`, `errors`, `evaluation`) importable with zero network deps, and isolates every third-party import behind a factory. `__init__.py` stays EMPTY — no re-exports, so `import rag.metrics` never drags `langchain_openai`.

| Path | Purpose |
|---|---|
| `pyproject.toml` | `[tool.pytest.ini_options] pythonpath=["src","."]`, registers `integration` marker. No build backend, no deps (deps live in requirements.txt). |
| `.env.example`, `.gitignore` | 6 mandated vars + optional block; ignore `.env`, `.venv/`, `artifacts/bm25_corpus.json`. |
| `src/rag/__init__.py` | Empty package marker. |
| `src/rag/errors.py` | Exception taxonomy. Pure stdlib. |
| `src/rag/config.py` | `Settings` + `load_settings` + `require_credentials`. dotenv imported lazily. |
| `src/rag/chunking.py` | Corpus load, token split, metadata build/validate. tiktoken + splitters imported lazily. |
| `src/rag/embeddings.py` | `build_embeddings()` + `embed_texts()` dimension assertion. `langchain_openai` lazy. |
| `src/rag/pinecone_client.py` | ALL Pinecone SDK calls: client, verify-or-create, index handle, store factory. `pinecone` lazy. |
| `src/rag/hydration.py` | Pinecone → BM25 corpus + disk cache + local fallback. |
| `src/rag/retrieval.py` | `RAGSystem`, `RetrievedDocument`, fusion/dedup/slice. LangChain lazy in `__init__`. |
| `src/rag/metrics.py` | Pure metric math. Stdlib only. |
| `src/rag/evaluation.py` | Golden Set load/validate, scoring, report payload. Stdlib only. |
| `scripts/init_index.py`, `scripts/ingest.py`, `evaluate.py` | Thin CLIs; each bootstraps `sys.path` with 3 lines so `python scripts/x.py` works with no install. |
| `tests/` | See §7. |

## 2. Metadata schema — FINAL, verbatim

**10 keys total = 9 schema fields + `text` payload.** The proposal's "9 fields" counts only the non-payload fields; the explore's 10-row table is authoritative. After ingestion `text` is popped by `PineconeVectorStore`, so retrieved/hydrated docs carry 9 keys.

| # | Field | Type | Example |
|---|---|---|---|
| 1 | `document_id` | `str` | `"pytest-fixtures"` |
| 2 | `chunk_id` | `str` (**also the vector id**) | `"pytest-fixtures#002"` |
| 3 | `chunk_index` | `int` | `2` |
| 4 | `source` | `str` | `"data/corpus/pytest-fixtures.md"` |
| 5 | `page` | `int` (0 = not paginated) | `0` |
| 6 | `section` | `str` (nearest heading, `""` if none) | `"Fixture scopes"` |
| 7 | `category` | `str` (closed set) | `"fixtures"` |
| 8 | `tags` | `list[str]` | `["pytest","fixture","scope"]` |
| 9 | `token_count` | `int` | `612` |
| 10 | `text` | `str` (verbatim chunk) | `"Fixtures are..."` |

**`token_count` is KEPT.** Dropping it would remove the only observable evidence that chunking is token-based — the spec scenario `0 < token_count <= 800` becomes untestable, and the size-guard scenario loses its forensic field. Cost is 4 bytes.

`pipeline_version` is **not** a metadata field (would break the fixed schema); it is a `Settings` value used only in the cache key. No `None` values, no nested objects, no bools.

## 3. Interfaces & signatures

```python
# errors.py
class RagError(Exception): ...
class ConfigError(RagError): ...
class MissingConfigError(ConfigError): ...
class CorpusTooSmallError(RagError): ...
class MetadataTooLargeError(RagError): ...
class DimensionMismatchError(RagError): ...
class IndexConfigMismatchError(RagError): ...
class GoldenSetError(RagError): ...
class HydrationError(RagError): ...
class RetrievalEmptyError(RagError): ...

# config.py
@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str; openai_api_key: str; index_name: str
    namespace: str = "rag-docs"
    bm25_weight: float = 0.4; vector_weight: float = 0.6
    embedding_model: str = "text-embedding-3-small"; embedding_dimensions: int = 1536
    chunk_size: int = 600; chunk_overlap: int = 80; batch_size: int = 32
    cloud: str = "aws"; region: str = "us-east-1"; pipeline_version: str = "1"
    corpus_dir: Path = Path("data/corpus"); golden_set_path: Path = Path("data/golden_set.json")
    artifacts_dir: Path = Path("artifacts")

def load_settings(env: Mapping[str, str] | None = None) -> Settings
    # env=None -> load_dotenv(override=False) then read os.environ; env={...} -> hermetic, no dotenv
def require_credentials(settings: Settings) -> None   # raises MissingConfigError listing ALL missing

# chunking.py
@dataclass(frozen=True)
class Chunk: id: str; text: str; metadata: dict[str, object]

def load_corpus(corpus_dir: Path, *, min_documents: int = 6) -> list[Document]  # CorpusTooSmallError
def make_splitter(*, chunk_size: int = 600, chunk_overlap: int = 80) -> TextSplitterLike  # lazy tiktoken
def split_documents(docs, *, splitter=None, count_tokens: Callable[[str], int] | None = None) -> list[Chunk]
def build_metadata(*, document_id: str, chunk_index: int, text: str, source: str,
                   section: str, category: str, tags: list[str], token_count: int,
                   page: int = 0) -> dict[str, object]
def validate_metadata(metadata: Mapping[str, object], *, max_bytes: int = 40_000) -> None
    # MetadataTooLargeError on UTF-8 JSON > 40_000; also enforces exact key set + types

# embeddings.py
def build_embeddings(settings: Settings) -> EmbeddingsLike          # lazy langchain_openai
def embed_texts(embeddings, texts: Sequence[str], *, expected_dim: int = 1536) -> list[list[float]]
    # DimensionMismatchError naming expected/actual, before any upsert

# pinecone_client.py
def build_client(settings: Settings) -> PineconeLike
def verify_or_create_index(client, settings: Settings) -> dict   # IndexConfigMismatchError; no delete/recreate
def build_index(client, settings: Settings) -> IndexLike          # client.Index(settings.index_name)
def build_vector_store(index, embeddings, settings: Settings) -> VectorStoreLike  # text_key="text", namespace=ns

# hydration.py
def build_corpus(*, index, settings: Settings, cache_path: Path | None = None,
                 local_corpus_dir: Path | None = None) -> list[Document]   # HydrationError

# retrieval.py
@dataclass(frozen=True)
class RetrievedDocument:
    id: str; document_id: str; text: str; metadata: dict[str, object]; score: float | None = None

class RAGSystem:
    def __init__(self, *, settings: Settings, index=None, embeddings=None,
                 vector_store=None, corpus: list[Document] | None = None,
                 mode: Literal["hybrid","vector","bm25"] = "hybrid") -> None
    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]   # RetrievalEmptyError

# metrics.py  (pure, stdlib only)
def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int = 5) -> float
def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int = 5) -> float
def hits_at_k(retrieved_ids, relevant_ids, k: int = 5) -> int
def macro_precision(pairs: Sequence[tuple[Sequence[str], Collection[str]]], k: int = 5) -> float
def macro_recall(pairs, k: int = 5) -> float
def precision_ceiling(relevant_sizes: Sequence[int], k: int = 5) -> float
def normalized_precision(precision: float, ceiling: float) -> float      # ceiling==0 -> 0.0

# evaluation.py
@dataclass(frozen=True)
class GoldenQuestion: question_id: str; question: str; relevant_document_ids: tuple[str, ...]
@dataclass(frozen=True)
class QuestionScore: question_id: str; retrieved_document_ids: tuple[str,...]; hits: int
                     m: int; precision: float; recall: float
@dataclass(frozen=True)
class MetricsReport: k: int; n_questions: int; precision: float; recall: float
                     ceiling: float; normalized_precision: float; per_question: tuple[QuestionScore,...]

def load_golden_set(path: Path, *, corpus_ids: set[str] | None = None,
                    min_questions: int = 0, min_relevant: int = 2) -> list[GoldenQuestion]  # GoldenSetError
def score_retrieval(golden, retrieved_ids: Sequence[Sequence[str]], k: int = 5) -> MetricsReport
def build_report_payload(*, settings, golden, ablation: Mapping[str, MetricsReport],
                         generated_at: str) -> dict[str, object]      # artifacts/last_eval.json shape
```

`retrieve()` internals: run `mode` retriever → **dedupe by `document_id` preserving fused order** → `[:k]` → if empty raise `RetrievalEmptyError(f"index={index_name} namespace={namespace}: 0 documents ...")`. Both guards stay even though `id_key="document_id"` already dedupes, because fakes (and a mis-set `id_key`) can return duplicates; the spec forbids relying on the ensemble.

`mode` is decided at construction: `hybrid` → `EnsembleRetriever([bm25, vec], weights=[bm25_w, vec_w], id_key="document_id")`; `vector`/`bm25` → call that single retriever directly (no `[1.0]` ensemble, which would make the weight validation meaningless).

## 4. Data shapes

**Golden Set** (`data/golden_set.json`, array, ≥10 entries, each ≥2 ids):
```json
[{"question_id":"q1","question":"How do fixture scopes differ?",
  "relevant_document_ids":["pytest-fixtures","pytest-conftest"]}]
```
Validation: unique non-empty `question_id`, non-empty `question`, `len(set(ids))>=2`, exact membership against corpus stems (never substring), `>=10` entries for the committed file. Single-relevant rejection message cites `R@k = k·P@k`.

**Hydration cache** (`artifacts/bm25_corpus.json`, gitignored):
```json
{"cache_key":{"index_name":"rag-pytest","namespace":"rag-docs","pipeline_version":"1","page_count":1},
 "vector_count":24,
 "documents":[{"id":"pytest-fixtures#000","text":"...","metadata":{"document_id":"...","chunk_id":"..."}}]}
```

**`artifacts/last_eval.json`** (COMMITTED so a keyless grader can run `--offline`):
```json
{"schema_version":1,"generated_at":"2026-10-22T23:00:00Z","index_name":"rag-pytest",
 "namespace":"rag-docs","embedding_model":"text-embedding-3-small","pipeline_version":"1",
 "k":5,"n_questions":12,"golden_set_path":"data/golden_set.json",
 "primary":{"mode":"hybrid","precision":0.36,"recall":0.833333,"ceiling":0.44,"normalized_precision":0.818182},
 "ablations":{"vector":{"precision":0.0,"recall":0.0,"ceiling":0.44,"normalized_precision":0.0},
              "bm25":{"precision":0.0,"recall":0.0,"ceiling":0.44,"normalized_precision":0.0},
              "hybrid":{"precision":0.36,"recall":0.833333,"ceiling":0.44,"normalized_precision":0.818182}},
 "per_question":[{"question_id":"q1","retrieved_document_ids":["..."],"hits":2,"m":3,
                  "precision":0.4,"recall":0.666667}]}
```

## 5. Hydration algorithm (exact)

```
stats   = index.describe_index_stats()
vcount  = stats.namespaces.get(ns).vector_count if ns in stats.namespaces else 0
key     = (index_name, ns, pipeline_version, ceil(vcount/1000))
if cache exists and cache.cache_key == key and cache.vector_count == vcount: return docs   # 0 list/fetch
if vcount == 0: goto LOCAL
try:
    ids = [vid for page in index.list(namespace=ns) for vid in page]      # generator, ~100/page
    docs = []
    for i in range(0, len(ids), 1000):                                   # <=1000 ids/call
        res = index.fetch(ids=ids[i:i+1000], namespace=ns)
        for vid, vec in res.vectors.items():
            md = dict(vec.metadata); docs.append(Document(id=vid, page_content=md.pop("text"), metadata=md))
    write cache(key, vcount, docs); return docs
except Exception: log warning; goto LOCAL
LOCAL: docs = chunking.split_documents(chunking.load_corpus(corpus_dir))  # source of truth
       return docs   # failure here -> HydrationError(cache path + remediation)
```

**Collect-all-ids-then-fetch is required**, not streamed per page: streaming would emit ~15 fetches for 1,500 ids; the spec demands exactly 2 (`1000 + 500`). Namespace is always passed explicitly — never `None`.

**Scale ceiling (README honesty):** O(N) id listings + one 1,000-id fetch per batch, each returning unused 1536-dim values (~1.8 MB per 300 chunks). Correct to ~10⁴–10⁵ chunks. Beyond that, the lexical side needs Pinecone native `full_text_search` (SDK v10, blocked by the `langchain-pinecone` pin) or an external BM25 service.

## 6. Metrics & reporting

Pure, stdlib-only, integer-exact hit counting:

```
hits(retrieved, relevant, k) = |set(retrieved)[:k] ∩ relevant|     # dedupe first-occurrence, then truncate
precision_at_k = hits / k                                          # fixed denominator, thin retrieval penalized
recall_at_k    = hits / len(relevant)                              # ValueError if relevant empty
P@k = (1/N)Σ P@k(q);  R@k = (1/N)Σ R@k(q)                          # macro, ValueError if N==0
ceiling = (1/N)Σ min(m_q,k)/k
normalized_precision = P@k / ceiling                               # 0.0 when ceiling == 0
```

**Frozen calibration case** (`tests/test_metrics.py`, single fixture): `k=5`, `m_q=[1,1,1,3,6]`, `hits_q=[1,1,1,1,5]` → `P@5=0.36`, `R@5=0.833333`, `ceiling=0.44`, `normalized=0.818182`, per-question `(0.2,1.0),(0.2,1.0),(0.2,1.0),(0.2,0.333333),(1.0,0.833333)`. Note `m=6, hits=5` proves `min(m,k)` clamping; a naive `sum(m)/k` ceiling would give 0.48 and fail.

Ablation = same Golden Set scored three times through `RAGSystem(mode=...)`. Report rows in fixed order `vector, bm25, hybrid`, plus the redundancy identity `R@k == k·P@k` when every `m_q==1` (asserted as a test, not printed).

## 7. Credential-free grading & lazy imports

Three import tiers, enforced by test placement:

| Tier | Modules imported | Needs creds/network |
|---|---|---|
| A — pure | `errors`, `metrics`, `evaluation`, `config` (env injected), `chunking` (metadata only, injected `count_tokens`) | No. No third-party at import. |
| B — wiring with fakes | `hydration`, `retrieval`, `pinecone_client` (typed protocols only) | No. LangChain imported at call time; fakes used. |
| C — integration | real Pinecone/OpenAI | Yes; `@pytest.mark.integration`, auto-skipped by `conftest.py` when either key is absent. |

Import guards: `__init__.py` empty; `dotenv`/`tiktoken`/`langchain_*`/`pinecone`/`openai` imported **inside functions**, never at module top; `pyproject.toml` `pythonpath=["src","."]` so no `sys.path` hacks in tests (scripts do their own 3-line bootstrap). CI command: `uv run pytest -q` with both keys unset — must be green.

**Exit-code-2 contract** (all three CLIs): `main()` wrapped once —
`except MissingConfigError as e: print(f"Missing configuration: {e}"); print("Copy .env.example to .env and set the listed values."); print("Credential-free checks: uv run pytest -q"); sys.exit(2)`. `--offline` never calls `load_settings` with credential requirement and never constructs a client → exit 0.

## 8. Config & `.env`

Mandated (`.env.example`, all 6 present): `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME`, `PINECONE_NAMESPACE=rag-docs`, `ENSEMBLE_BM25_WEIGHT=0.4`, `ENSEMBLE_VECTOR_WEIGHT=0.6`.
Added (optional, commented block): `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `UPSERT_BATCH_SIZE`, `PINECONE_CLOUD`, `PINECONE_REGION`, `PIPELINE_VERSION`.

| Rule | Failure |
|---|---|
| Weights: exactly two, each `>0`, `abs(sum-1.0) <= 1e-6`, no NaN/inf | `ConfigError("ENSEMBLE_BM25_WEIGHT=4.0 and ENSEMBLE_VECTOR_WEIGHT=6.0: weights must be positive and sum to 1.0 (±1e-6), got 10.0")` |
| `namespace` non-empty (never `""`, never `None`) | `ConfigError("PINECONE_NAMESPACE must be a non-empty string (empty means the default namespace and breaks every read/write)")` |
| `chunk_size>0`, `0 <= chunk_overlap < chunk_size` | `ConfigError` naming both values |
| `batch_size` in `1..100`, `embedding_dimensions==1536` | `ConfigError` |
| Keys: `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME` for scripts; `require_credentials` at first client build | `MissingConfigError` listing every missing name + `.env.example` pointer |

`load_dotenv(..., override=False)` so process env wins (spec scenario), and `.env` is gitignored.

## 9. Testing strategy (strict TDD, tests first)

| File | Covers | Tier | Fails on broken impl when… |
|---|---|---|---|
| `test_metrics.py` | frozen case, thin, duplicates, empty relevant, identity | A | denominator uses `len(retrieved)` (0.2→1.0); dupes not removed (0.4→0.6); ceiling without `min` (0.44→0.48); micro recall (0.8333→0.75) |
| `test_chunking.py` | exact 10-key schema, `page==0:int`, `tags:list[str]`, `chunk_id` format, token sizing, 40 KB guard, corpus floor, `document_id` stem | A | `length_function=len` yields ~10 chunks for a 1500-token doc (assert 2–5); allowing `None`/extra key; guard removed → no raise |
| `test_config.py` | defaults, shell-wins-over-file, weights `[4,6]`, `""` namespace, missing-key aggregation | A | weight check only asserts positivity; `.env` overrides shell; first-missing-only message |
| `test_golden_set.py` | valid, duplicate id, 1 relevant, unknown id, <10 entries | A | `min_relevant` check dropped; substring matching accepts `pytest-fixtures-advanced` |
| `test_evaluation.py` | per-question + aggregates, payload shape, offline reprint | A | report drops `ceiling`/`normalized_precision`; per_question order lost |
| `test_hydration.py` | cache hit (0 calls), 1,500 ids → exactly 2 fetches (1000/500), `text` popped, cache key/format, namespace arg, Pinecone error → local, both fail → `HydrationError` | B | streaming per page → 15 fetches; `namespace=None`; cache key ignores `vector_count`; silent empty corpus instead of fallback |
| `test_retrieval.py` | ensemble weights/`k=20`/`id_key`, over-return 9→5 distinct, <5 docs, payload, `RetrievalEmptyError`, mode branches | B | `[:5]` removed → 9 docs; dedupe removed → repeated `document_id`; `id_key=None`; empty list returned silently |
| `test_ingest.py` | floor before embed, dims asserted before upsert, 24 records in one `batch_size=32` call, idempotent re-upsert | B | dims check after upsert; per-record upsert calls; id generated by uuid instead of `chunk_id` |
| `test_index_init.py` | create kwargs exact, no-op when present, `IndexConfigMismatchError` with no delete/create, namespace printed | B | schema API used; recreate on mismatch; `create_index` without `ServerlessSpec` |
| `test_cli_exit_codes.py` | 3 scripts, scrubbed env → exit 2, message names var + `.env.example` + `uv run pytest -q`, no traceback | B | `KeyError`/`PineconeConfigurationError` traceback; exit 1 |
| `tests/integration/test_live_roundtrip.py` | init → ingest → hydrate → retrieve → evaluate | C | opt-in only |

Fakes (`tests/conftest.py`): `FakeIndex` (records `list`/`fetch`/`describe_index_stats` args, honours 1000-id slicing, returns `vectors[id] = SimpleNamespace(metadata=...)`), `FakeEmbeddings` (deterministic hash vectors, configurable dim), `FakeVectorStore` (`.as_retriever()` → `FakeRetriever`, emits queued docs), `FakeSplitter` (char-based, no tiktoken). No test imports `openai` or `pinecone`.

## 10. Decisions & tradeoffs

| Fork | Chosen | Rejected | Why |
|---|---|---|---|
| Layout | `src/rag/` + `pyproject` pythonpath | flat `rag/` | Keeps pure modules importable without an install step and without path hacks; matches proposal |
| Metadata count | 9 + `text` = 10 keys | drop `token_count` (9 keys) | Spec's token-sizing scenario needs an observable field; 4 bytes |
| Hydration | collect ids → batched fetch → disk cache | stream fetch per page; local-only | Exactly `ceil(N/1000)` fetches per spec; cache makes init cheap; local-only hides the metadata-text requirement |
| Cache key | `(index, ns, pipeline_version, ceil(count/1000))` + exact count | mtime-based | Cheap stats call catches re-ingest; no crystal-ball invalidation |
| Errors | One `RagError` root, 9 subclasses, exit 2 only for config | bare `ValueError`/`RuntimeError` | Named, testable failure modes; graders get a stable contract |
| Ablation | `RAGSystem(mode=...)`, single retriever for single modes | `[1.0]`/`[0,1]` ensembles | Degenerate weights would bypass the weight-validation logic under test |
| `metrics` deps | stdlib only | numpy | Guarantees the credential-free path; integer hits are exact |
| Retrieval fusion | `id_key="document_id"` + defensive dedupe + `[:5]` | trust ensemble's `k` | Verified: ensemble returns the union, never truncated |

## 11. Slice boundaries (`stacked-to-main`, ~800-line budget)

| Slice | Contents | ~lines |
|---|---|---|
| PR1 foundation | `pyproject`, `requirements`, `.env.example`, `.gitignore`, `errors.py`, `config.py`, `metrics.py`, `evaluation.py` (pure part), `test_metrics/config/golden_set/errors` | ~330 |
| PR2 ingest | `chunking.py`, `embeddings.py`, `pinecone_client.py`, `scripts/ingest.py`, `scripts/init_index.py`, `data/corpus/*.md`, `test_chunking/ingest/index_init` | ~350 |
| PR3 retrieval+eval | `hydration.py`, `retrieval.py`, `evaluate.py`, `data/golden_set.json`, `artifacts/last_eval.json`, `test_hydration/retrieval/cli`, README | ~360 |

PR2 targets PR1's branch; PR3 targets PR2's. Each slice is independently revertable and testable.

## 12. Rollback

- Revert the offending slice commit; `pinecone_client.py` is the only module touching the SDK, so an SDK break is a one-file revert.
- No destructive Pinecone call exists in code (no delete/recreate). Rollback of data = manual namespace delete in the Pinecone console (documented in README troubleshooting).
- Hydration cache is disposable: delete `artifacts/bm25_corpus.json` (or bump `PIPELINE_VERSION`) to force re-hydration; local re-chunk is the built-in fallback.
- `evaluate.py --offline` reprints `artifacts/last_eval.json`, so metrics survive any credential/API failure.

## 13. README replication outline

1. Prereqs: Python 3.12, `uv`, Pinecone account, OpenAI key.
2. `uv venv --python 3.12 .venv`; `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`.
3. `Copy-Item .env.example .env`, fill keys.
4. `uv run python scripts/init_index.py` → create/verify, print dimensions, metric, namespaces.
5. `uv run python scripts/ingest.py` → 8 docs / ~24 chunks, print stats.
6. `uv run python evaluate.py` → P@5, R@5, ceiling, normalized, 3-row ablation; writes `artifacts/last_eval.json`.
7. `uv run python evaluate.py --offline` → same table, no keys (grader path).
8. `uv run pytest -q` → green with zero credentials.
9. Metrics explained: fixed-`k` denominator, `R@k = k·P@k` redundancy, ceiling and normalized precision, doc-level fusion bias for long docs.
10. Scale notes + troubleshooting (exit code 2, namespace/`text_key` traps, stale cache, SDK pin rationale).

## Open Questions

- [ ] None blocking. (`PineconeVectorStore(index=..., namespace=..., text_key=...)` is the intended single construction site; if 0.2.13's constructor differs, only `build_vector_store` changes — verify at first apply step.)
