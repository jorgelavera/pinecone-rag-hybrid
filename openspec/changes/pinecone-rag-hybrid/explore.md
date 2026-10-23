# Exploration: pinecone-rag-hybrid

> Phase: explore · Change: `pinecone-rag-hybrid` · Store: hybrid (OpenSpec + Engram)
> All SDK claims below were verified by installing packages and introspecting them, not from memory.

## Verdict table

| # | Problem | Recommendation |
|---|---------|----------------|
| 0 | Pinecone SDK API surface | Target `pinecone==7.3.0` **legacy** API (`create_index(name, dimension, metric, spec=ServerlessSpec(...))`) + `langchain-pinecone==0.2.13`. The schema API is unreachable — see below. |
| 1 | BM25 hydration | Hydrate the corpus from Pinecone at `RAGSystem` init (`index.list` → `index.fetch`, ≤1000 ids/call), cache to `artifacts/bm25_corpus.json`, fall back to local re-chunk. |
| 2 | Precision@5 / Recall@5 | Macro-averaged, **doc-level** matching on `document_id`, fixed denominator `k=5`; report the achievable ceiling and normalized precision. |
| 3 | Ensemble weights | `[0.4 BM25, 0.6 vector]`, overridable via `.env`, validated at load time; prove it with an ablation table. |
| 4 | Namespaces | One explicit namespace `rag-docs` (env `PINECONE_NAMESPACE`) used by every read and write; categories live only in metadata. |
| 5 | Corpus | 8 committed Markdown files on `pytest`, fully offline and deterministic. |
| 6 | Grader without keys | Preflight config validation → typed error → actionable message + exit code; unit suite is 100% fake-based; integration tests opt-in. |
| 7 | Metadata schema | 9 flat fields, chunk id as vector id, text stored under the name `text`, 40 KB guard enforced by a test. |

---

## Decision 0 (prerequisite): which Pinecone API to target

### Verified facts

I created a clean Python 3.12 venv and installed `pinecone` + `langchain-pinecone`. The resolver installed **pinecone 7.3.0**, even though PyPI's latest is **10.1.0** (released 2026-10-20). Reason — `langchain-pinecone 0.2.13` metadata:

```
pinecone[asyncio]<8.0.0,>=6.0.0
```

That is a hard upper bound. Consequences, confirmed by introspection:

- `pinecone 7.3.0` has **no `pc.indexes`** namespace — the schema-based `pc.indexes.create(...)` + `SchemaBuilder` API **does not exist** in any installable combination with `langchain-pinecone`.
- Pinecone's own docs state document/schema indexes require *Python SDK v10 or later*. That path is therefore blocked for us.
- `pinecone 7.3.0` **does** expose the still-documented "bring your own vectors" path:
  `pc.create_index(name, vector_type="dense", dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))`
  plus `pc.has_index(name)`, `pc.describe_index(name)`, `pc.Index(name)`.
- `langchain-pinecone 0.2.13` exposes `PineconeVectorStore` (MRO: `PineconeVectorStore → VectorStore → ABC`), with `from_texts`, `add_texts`, `similarity_search`, `as_retriever`, `delete`.

### Options

| Option | Pros | Cons | Effort |
|---|---|---|---|
| **A. `pinecone<8` + `langchain-pinecone` + `PineconeVectorStore`** | Only combination that installs cleanly; matches documented course examples; least custom code | Locked to a 2025-04-era API version; no access to FTS/schema features | Low |
| B. `pinecone==10.x` + hand-written `BaseRetriever` over `pc.indexes` | Modern API, native `full_text_search` | Cannot use `PineconeVectorStore`; whole custom data-plane layer; diverges from what a grader expects | High |
| C. `pinecone==10.x` + `langchain-pinecone` unpinned | — | **Does not install.** Contradicts a declared dependency constraint | N/A |

**Recommendation: A.** Pin it explicitly in `requirements.txt` (not left implicit) so a future `pinecone` release can't silently break the install:

```
pinecone==7.3.0
langchain-pinecone==0.2.13
langchain-community==0.4.2
langchain-classic==1.0.8
langchain-core==1.7.0
langchain-openai==1.7.2
langchain-text-splitters==1.1.3
rank-bm25==0.2.2
python-dotenv==1.2.4
```

`rank-bm25` is **mandatory and not pulled transitively** — `BM25Retriever.from_documents` raises
`ImportError: Could not import rank_bm25, please install with 'pip install rank_bm25'` without it. Verified.

Import paths that actually work:

```python
from langchain_community.retrievers import BM25Retriever      # works; langchain_classic path is deprecated
from langchain_classic.retrievers import EnsembleRetriever    # EnsembleRetriever only lives here
```

`langchain` (the meta package) is **not installed** and is not needed. `langchain-classic` arrives transitively via `langchain-community`, but pin it explicitly.

**Risk carried by this choice:** `pinecone 7.3.0` targets API version `2025-04`. It works today but will age. Mitigation: isolate every Pinecone call behind `pinecone_client.py` so a future v10 migration touches one module.

---

## 1. BM25 hydration (biggest architectural risk)

### The problem

`BM25Retriever.from_documents(...)` builds an in-memory `rank_bm25` index. Pinecone cannot host it. Since chunk text lives in Pinecone metadata, the corpus that BM25 needs is *inside* the index — so `RAGSystem.__init__` must reconstruct it.

### Verified mechanics

- `index.list(**kwargs)` is a **generator** that yields `list[str]` of ids, ~100 per page, following `pagination_token` internally.
- `index.fetch(ids=..., namespace=...)` accepts **max 1,000 ids per request** (documented limit) and returns `{id: Vector(values, metadata)}`.
- `fetch` **always** returns vector values — there is no `include_values=False`. At 1536 floats/chunk that is ~1.8 MB of wasted bandwidth per 300 chunks. Acceptable at course scale, not at production scale.

### Options

| Option | Pros | Cons |
|---|---|---|
| A. Re-chunk local source files at init | No API calls, instant, no bandwidth waste | Fails if source files absent; duplicates pipeline logic; a grader cannot see the metadata trick working |
| B. `list` + `fetch` all vectors from Pinecone | Demonstrates the stored-text requirement; self-contained; source files optional | O(N) reads; wastes bandwidth on values; does not scale to 10^6 vectors |
| C. `query` with a dummy vector, `top_k=10000` | One call | Ordering is arbitrary and undocumented; 4 MB response cap; fragile. Reject |
| D. B + disk cache | Read cost paid once; fast repeated init; cache doubles as offline corpus | Cache can go stale if the index changes |

**Recommendation: B + D.**

```
hydration.build_corpus(index, namespace, cache_path)
  cache key = (index_name, namespace, pipeline_version, ceil(vector_count/1000))
  if cache hit  -> load JSON, return
  else          -> for id_batch in index.list(namespace=ns):
                     for slice in chunks(id_batch, 1000):
                       for vid, v in index.fetch(slice, namespace=ns).vectors.items():
                         md = dict(v.metadata); text = md.pop("text")
                         Document(id=vid, page_content=text, metadata=md)
                   write cache; return
```

Cache validation uses `describe_index_stats().namespaces[ns].vector_count` — cheap, and catches the "index changed, cache stale" case. `pipeline_version` in metadata catches chunking-config changes.

**Fallback chain:** cache → Pinecone → local re-chunk → explicit error with remediation text. This single function is what makes the offline/no-credentials story work.

**Scale honesty for the README:** hydration is O(N) and is correct up to roughly 10^4–10^5 chunks. Beyond that, hybrid retrieval must move the lexical side into Pinecone's native `full_text_search` (requires SDK v10, blocked here) or an external BM25 service (OpenSearch). Documenting this limit is stronger than pretending it scales.

---

## 2. Precision@5 and Recall@5 with multiple relevant docs

### Definitions (macro-averaged over questions)

For question `q`, with relevant doc set `R_q` (`m_q = |R_q| ≥ 1`) and the **deduplicated, ordered** top-5 doc list `T_q`:

```
hits_q       = |set(T_q) ∩ R_q|
Precision@5  = hits_q / 5           # fixed denominator = k
Recall@5     = hits_q / m_q
P@5          = (1/N) · Σ_q Precision@5(q)
R@5          = (1/N) · Σ_q Recall@5(q)
```

### Worked example

| Q | relevant `R_q` | m | hits | P@5 | R@5 |
|---|---|---|---|---|---|
| q1 | {A,B,C} | 3 | 3 | 0.60 | 1.00 |
| q2 | {A,B} | 2 | 1 | 0.20 | 0.50 |
| q3 | {C,D} | 2 | 2 | 0.40 | 1.00 |
| q4 | {E} | 1 | 1 | 0.20 | 1.00 |
| q5 | {A,B,C} | 3 | 2 | 0.40 | 0.667 |

```
macro P@5 = (0.60+0.20+0.40+0.20+0.40)/5 = 0.36
macro R@5 = (1.00+0.50+1.00+1.00+0.667)/5 = 0.833
```

### Why a single expected doc makes the two metrics redundant (precise version)

The orchestrator's premise is directionally right but not literally correct — worth fixing before it lands in the README. With `m_q = 1` every question:

```
hits ∈ {0,1}  →  P@5 = hits/5  and  R@5 = hits  →  R@5 = 5·P@5  (exact identity)
```

So `P@5` is not "constant 0.2"; it is binary-valued `{0, 0.2}`, and `R@5` is `{0, 1}`. The real defect is that **one metric is a linear rescaling of the other** — you compute two numbers and learn one fact. Multiple relevant docs break that identity, which is the actual justification for the decision.

### Two non-obvious reporting requirements

1. **Ceiling.** Precision@5 cannot reach 1.0 when `m_q < 5`. The best attainable score is
   `ceiling = (1/N)·Σ_q min(m_q, 5)/5`. For the table above, `ceiling = 0.44`.
   Reporting `P@5 = 0.36` alone looks mediocre; `0.36 / 0.44 = 0.82` normalized is honest.
   `evaluate.py` must print both raw and normalized.
2. **Denominator.** Use `k=5` even if the retriever returned fewer than 5 docs — that is the standard definition and it penalizes thin retrieval correctly.

### Matching key

Use a doc-level `document_id` = source filename stem (`pytest-fixtures`). Match Golden Set entries against it exactly.

**Matching on `source` alone is dangerous** because `source` is a path-ish string (`data/corpus/pytest-fixtures.md`). If the Golden Set says `pytest-fixtures` and metadata says `data/corpus/pytest-fixtures.md`, exact matching scores 0 everywhere; if you "fix" it with substring matching, then `pytest-fixtures` also matches `pytest-fixtures-advanced`, silently inflating precision. A dedicated `document_id` that is *defined* as the canonical key removes both failure modes.

### Doc-level fusion is required for a clean denominator

Verified: `EnsembleRetriever.weighted_reciprocal_rank` returns the **union** of all retriever lists, deduplicated — it does **not** truncate to `k`. With two retrievers at `k=20` it can return dozens of documents. Two consequences:

- `RAGSystem` **must slice `[:5]` itself**.
- With `id_key=None` (default) dedup is by `page_content`, so 5 results may contain several chunks of the *same* document → fewer than 5 distinct docs → ambiguous precision.

`EnsembleRetriever(retrievers=[bm25, vec], weights=[0.4, 0.6], id_key="document_id")` dedups at **document level**, so `[:5]` yields exactly 5 distinct documents. Verified in a live prototype: 10 chunks over 8 docs → 4–5 distinct docs returned.

**Recommendation:** `id_key="document_id"`, over-fetch `k=20` per retriever, slice `[:5]`.

Tradeoff to document: the document object returned for a key is the *first* occurrence in concatenation order (BM25 list first), so the chunk text shown may not be the best-matching chunk. Acceptable — we are returning documents, not passages.

---

## 3. Ensemble weights

`EnsembleRetriever` uses weighted reciprocal rank fusion:

```
score(d) = Σ_i weight_i / (rank_i(d) + c),   c = 60 (default, not configurable via .env)
```

Verified numeric consequence — because `c=60` dominates, raw ranks matter less than *agreement between retrievers*:

- doc at BM25 rank 1 only: `0.4/61 = 0.00656`
- doc at rank 10 in **both** lists: `0.4/70 + 0.6/70 = 0.01429` → outranks it

This is why per-retriever `k` must be well above 5: fusion needs overlapping candidates to find consensus.

### Recommendation

`weights = [0.4 /* BM25 */, 0.6 /* vector */]`

Rationale: vector search carries paraphrase and synonym recall; BM25 is there to win the exact technical tokens (`conftest.py`, `monkeypatch.setenv`, `tmp_path`, `--cov-report=term-missing`) that embeddings smear together. Keeping vector above 0.5 preserves generalization on questions that do not reuse the corpus vocabulary, while 0.4 is enough to reorder on exact matches. Equal weights (0.5/0.5) was rejected: it lets lexical noise promote chunks for questions phrased differently from the corpus.

Configurable via `.env` (`ENSEMBLE_BM25_WEIGHT=0.4`, `ENSEMBLE_VECTOR_WEIGHT=0.6`) so the ablation in `evaluate.py` can sweep them. `config.py` must validate: exactly two values, each `> 0`, sum ≈ 1.0 (±1e-6) — the class only checks that *some* weight is positive, so a typo like `[4, 6]` would pass silently.

**Payoff for the assignment:** `evaluate.py` should print a 3-row ablation — vector-only, BM25-only, hybrid — over the same Golden Set. That is the evidence that the hybrid architecture is justified rather than decorative, and it directly demonstrates "improves precision on technical terms". Cost: a few lines, three retriever configurations.

---

## 4. Namespace strategy

The assignment warns about "ignoring namespace". The concrete failure it refers to is verified in `langchain-pinecone` source: `add_texts` and `similarity_search` pass `namespace` straight through, and `None` means the **default namespace `""`**. So writing with `namespace=None` and querying with `namespace="docs"` returns **zero results with no error**. Silent, and exactly the trap.

Also verified: `PineconeVectorStore` will **silently skip** any match whose metadata lacks the `text_key` (`logger.warning("Found document with no `text` key. Skipping.")`). A namespace or text-key mistake therefore surfaces as "retrieval returns nothing", not as an exception.

### Options

| Option | Pros | Cons |
|---|---|---|
| Single default namespace (`""`) | Zero config | Cannot demonstrate namespace awareness; collides with any other data in the index |
| **Single explicit namespace `rag-docs`** | Idiomatic, explicit, one corpus for BM25, trivially consistent | Category isolation must use metadata filters |
| One namespace per category | Natural isolation; cheaper scans | Fragments the BM25 corpus (must hydrate N namespaces and merge), requires N vector queries per question, and defeats `EnsembleRetriever`'s single-retriever model |

**Recommendation: single explicit namespace `rag-docs`**, from `PINECONE_NAMESPACE` (default `rag-docs`).

Consistency enforcement:
- One module-level source of truth (`config.settings.namespace`); no literal strings anywhere else.
- `init_index.py` prints `describe_index_stats().namespaces` so the namespace is *visible* during setup.
- A unit test asserts the namespace passed to `add_texts`, to `similarity_search`, and to hydration is the same object (fake index records the argument).

`category` stays a **metadata** field for optional filtering (`filter={"category": "retrieval"}`), never a namespace. This keeps namespaces as a deployment boundary and metadata as the query-time dimension — the intended split.

---

## 5. Corpus choice

Requirement: reproducible at grade time without network.

| Option | Pros | Cons |
|---|---|---|
| Fetch a library's live docs | "Authentic" | Network + version drift at grade time; the grader's run may differ from yours. Reject |
| Vendor a library's docs into the repo | Authentic + offline | License/attribution handling; large diff |
| **8 curated Markdown files on `pytest`** | Fully offline, deterministic, tiny diff, terms chosen to stress BM25 | Author-written, not upstream text |

**Recommendation: 8 committed Markdown files under `data/corpus/`**, ~100–200 lines each, covering `pytest`:

`pytest-basics.md`, `pytest-fixtures.md`, `pytest-conftest.md`, `pytest-parametrize.md`,
`pytest-markers.md`, `pytest-monkeypatch.md`, `pytest-tmp-path.md`, `pytest-coverage.md`

Why `pytest` specifically: it is dense in tokens that embeddings handle poorly and BM25 handles perfectly — `conftest.py`, `monkeypatch.setenv`, `tmp_path`, `parametrize`, `--cov-report=term-missing`, `PytestUnknownMarkWarning`. That makes the hybrid-vs-vector ablation show a *real* difference instead of noise. It also matches the project's own stack, so the content is easy to write accurately.

Sizing: ~1,500 tokens/file → ~3 chunks/file at 600 tokens → **~24 chunks, 8 documents**. Eight documents is the minimum that makes doc-level `P@5` meaningful; encode `assert len(corpus_docs) >= 6` as a test so the corpus can't silently shrink.

---

## 6. Grader without Pinecone/OpenAI credentials

This is the top **grading** risk (not a technical one). Verified failure modes:

| Trigger | Exception | When it fires |
|---|---|---|
| `Pinecone()` with no key | `pinecone.PineconeConfigurationError` | constructor — fails fast |
| `OpenAIEmbeddings(...).embed_documents()` with no key | `openai.OpenAIError: Missing credentials` | first embed call — fails late |

Because one fails at construction and the other at first use, a naive script dies mid-ingest with a traceback. Design around it:

1. **Preflight** — `config.load_settings()` reads `.env` and validates `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME` *before* any client is constructed. Missing vars → `MissingConfigError` listing every missing name.
2. **Single catch point** — `evaluate.py` and `ingest.py` wrap `main()`:
   ```
   except MissingConfigError as e:
       print("Missing configuration: ...")
       print("1) copy .env.example to .env   2) fill PINECONE_API_KEY / OPENAI_API_KEY / INDEX_NAME")
       print("To run without credentials: uv run pytest -q   (unit suite uses fakes)")
       sys.exit(2)
   ```
   Exit code 2 = configuration problem, distinguishable from 1 = real failure.
3. **Offline mode** — `evaluate.py --offline` reads `artifacts/last_eval.json` (written on every successful run) and reprints the metrics table. A grader sees real numbers with no keys.
4. **Test suite is the real deliverable** — `uv run pytest -q` must pass on a bare machine. Verified feasible: I built a prototype with `FakeIndex` (mimicking `upsert`/`list`/`fetch`/`describe_index_stats`), `FakeEmbeddings` (deterministic hash-based vectors) and `FakeVectorRetriever`. It ran the complete flow — ingest → hydrate → BM25 → `EnsembleRetriever` → top-5 → `P@5`/`R@5` — with **zero credentials**. That prototype is the blueprint for the test doubles.

   Unit coverage (all fake-based): chunking sizes, metadata schema validity, `document_id` construction, cache hit/miss, hydration pagination (including >1000 ids), namespace consistency, ensemble weight validation, and `metrics.precision_at_k` / `recall_at_k` against hand-computed values.
5. **Integration tests opt-in** — `@pytest.mark.integration`, skipped unless both keys are present. `conftest.py` adds the marker and the skip rule so `-q` stays green.

Verified environment detail: `uv run pytest -q` works without a `pyproject.toml`, using the local `.venv` (tested: pytest 9.1.1, exit 0). Keep `.venv` at the project root or `uv run` may climb to a parent project.

---

## 7. Metadata schema

### Field set (flat only — Pinecone rejects nested objects and `null` values; both documented)

| Field | Type | Example | Purpose |
|---|---|---|---|
| `text` | str | `"Fixtures are functions..."` | **Original chunk text.** Name is fixed by `text_key`; see below |
| `document_id` | str | `"pytest-fixtures"` | Doc-level identity: Golden Set matching + `id_key` for fusion |
| `chunk_id` | str | `"pytest-fixtures#002"` | Chunk identity; **also the vector id** |
| `chunk_index` | int | `2` | Ordering / debugging |
| `source` | str | `"data/corpus/pytest-fixtures.md"` | Provenance trail |
| `page` | int | `0` | PDF page (1-based); `0` = not paginated |
| `section` | str | `"Fixture scopes"` | Markdown heading at chunk start; `""` for PDF |
| `category` | str | `"fixtures"` | Single closed-set tag, used for `filter=` |
| `tags` | list[str] | `["pytest","fixture","scope"]` | Extra filterable tags (list[str] is a supported metadata type) |
| `token_count` | int | `612` | Chunk-size guard / observability |

`page` is `int` for type stability (`0` for Markdown), with the real location carried in `section`; `line_start`/`line_end` can be added from the splitter's `add_start_index=True` if a line range is wanted. Do not make `page` a string for Markdown — mixed types in one field break filtering.

### The `text` name is not cosmetic

`PineconeVectorStore` reads `text_key` (default `"text"`) from metadata, **pops it**, and uses it as `page_content`. If the field is named `original_text` and `text_key` is left at its default, every result is **silently dropped** with only a warning. Two safe paths: name the field `text`, or set `text_key="original_text"` on *every* construction site (`add_texts` and query both use it).

**Recommendation:** name it `text`, so the assignment's "store the original text in metadata" is satisfied with zero custom config. Add a test asserting the ingest call and the retrieval constructor agree on `text_key`.

A useful consequence: after retrieval, `Document.id` is the Pinecone vector id (`chunk_id`) and `document_id` survives; only `text` is removed. So `id_key="document_id"` works on retrieved docs.

### Size limits

- Documented hard limit: **40 KB of filterable metadata per record**.
- Our chunk ceiling: 800 tokens ≈ 3.2 KB of UTF-8 English. Two orders of magnitude of headroom.
- Enforce anyway: a unit test asserts `len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) < 40_000` for every chunk. This is a cheap guard against a splitter misconfiguration silently producing a 3,000-token chunk — and a 40 KB violation is a hard upsert failure, not a warning.

### Chunking

`RecursiveCharacterTextSplitter` is **character-based by default** (`length_function=len`). `chunk_size=600` means 600 *characters*, not tokens — a classic mistake that yields ~150-token chunks and wrecks retrieval quality. To honor "~500–800 tokens":

```python
RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",   # same family as text-embedding-3-small
    chunk_size=600, chunk_overlap=80, add_start_index=True,
)
```

Two notes:
- `tiktoken` downloads the `cl100k_base` BPE on first use (cached afterwards). Fine for ingest, but keep it out of the credential-free unit-test path — inject the length function rather than calling tiktoken in tests.
- A single indivisible block (long code fence) can exceed `chunk_size`. Guard with the token-count assertion via `token_count` metadata.

Also verified: `OpenAIEmbeddings` has **its own `chunk_size` (default 1000)** meaning *embedding batch size*, completely unrelated to the splitter's `chunk_size`. Passing one for the other is a silent bug. Set `model="text-embedding-3-small"` and `dimensions=1536` explicitly — the class default is `text-embedding-ada-002`, and `dimensions` defaults to `None`.

Every ingest run must assert `len(vector) == 1536` before upsert: this is the assignment's #1 named error (dimension mismatch) and it is cheap to catch.

---

## Proposed layout (for the proposal/design phases)

```
pinecone-rag-hybrid/
├─ .env.example                 # committed; .env gitignored
├─ requirements.txt             # fully pinned (see Decision 0)
├─ README.md                    # index replication steps + metrics summary
├─ data/
│  ├─ corpus/*.md               # 8 files, offline
│  └─ golden_set.json           # {pregunta, documentos_esperados[]} x ~12
├─ src/rag/
│  ├─ config.py                 # settings + validation + MissingConfigError
│  ├─ chunking.py               # load -> split -> metadata (pure, deterministic)
│  ├─ pinecone_client.py        # ALL SDK calls isolated here
│  ├─ hydration.py              # Pinecone -> BM25 corpus + JSON cache
│  ├─ retrieval.py              # RAGSystem: hybrid, id_key, slice [:5]
│  └─ metrics.py                # precision_at_k / recall_at_k (pure)
├─ scripts/{init_index.py,ingest.py}
├─ evaluate.py                  # P@5 / R@5 + ceiling + ablation + --offline
└─ tests/                       # all fake-based
```

`metrics.py` and `chunking.py` are pure functions — write them test-first. `pinecone_client.py` holds every SDK call so the eventual v10 migration is one file.

## Affected areas

Everything is new; nothing exists yet except `openspec/`. The files that carry architectural risk:

- `src/rag/hydration.py` — pagination, cache invalidation, fallback chain (highest risk)
- `src/rag/retrieval.py` — `id_key`, `[:5]` slicing, namespace propagation
- `src/rag/metrics.py` — metric definitions, ceiling, matching key
- `src/rag/chunking.py` — token-based sizing, metadata construction, 40 KB guard
- `src/rag/config.py` — fail-fast validation, weight validation
- `evaluate.py` — degradation path, ablation table
- `tests/*` — the actual credential-free proof

## Risks

1. **SDK lock-in.** `pinecone<8` targets API version `2025-04`. Works today; will age. Isolate in one module.
2. **Hydration cost/scale.** O(N) reads, and `fetch` returns unused 1536-dim values. Correct to ~10^4–10^5 chunks, not beyond. Document the ceiling honestly.
3. **Silent-empty retrieval.** Namespace mismatch or wrong `text_key` yields zero results with no exception. Mitigated by the single namespace constant, the text-key test, and an explicit `assert len(results) > 0` in the smoke path.
4. **Doc-level fusion bias.** `id_key="document_id"` accumulates RRF score per matching chunk, so long documents with many chunks get a structural boost. Acceptable at 24 chunks; would need normalization at scale.
5. **tiktoken first-use download.** Needs network on a cold cache. Keep it out of the unit-test path via injected length function.
6. **Upsert batching.** Verified: `index.upsert(vectors)` with `batch_size=None` sends **everything in one request**, violating the documented 1,000-record / 2 MB limit. Use `PineconeVectorStore.add_texts` (default `batch_size=32`) for ingest, or pass `batch_size=100` explicitly.
7. **Metric misinterpretation.** Raw `P@5` is capped by `min(m,5)/5`; reporting it without the ceiling makes a good system look mediocre. Report normalized precision.
8. **Corpus size floor.** Fewer than 6 documents makes doc-level `P@5` degenerate. Encode as a test.

## Ready for proposal

**Yes.** Every fork is resolved to one recommendation, the riskiest assumption (SDK compatibility) is verified by installation rather than assumed, and the end-to-end architecture has been executed against fakes with zero credentials. Nothing here blocks the proposal phase.
