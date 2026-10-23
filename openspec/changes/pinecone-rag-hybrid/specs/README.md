# Delta Index: pinecone-rag-hybrid

All four capabilities are **NEW** — there is no pre-existing behavior in `openspec/specs/`.
Each delta is therefore an ADDED-only delta. No MODIFIED, REMOVED, or RENAMED requirements
exist for this change.

Deltas live in the OpenSpec directory form so the archive step can consume them:

| Capability | Type | Requirements | Scenarios | Delta spec |
|------------|------|--------------|-----------|------------|
| `hybrid-retrieval` | New (ADDED) | 6 | 11 | [hybrid-retrieval/spec.md](hybrid-retrieval/spec.md) |
| `ingest-pipeline` | New (ADDED) | 6 | 10 | [ingest-pipeline/spec.md](ingest-pipeline/spec.md) |
| `evaluation-metrics` | New (ADDED) | 7 | 11 | [evaluation-metrics/spec.md](evaluation-metrics/spec.md) |
| `index-init` | New (ADDED) | 6 | 8 | [index-init/spec.md](index-init/spec.md) |

Totals: **25 requirements, 40 scenarios**.

## Scope guard

Retrieval and metrics only. No LLM answer generation: `RAGSystem` returns the top-5 retrieved
documents; `evaluate.py` measures Precision@5 and Recall@5. Any requirement mentioning an LLM
chat model is out of scope by decision.

## Archive note

At archive time these ADDED-only deltas are copied to `openspec/specs/<capability>/spec.md`
(the source of truth), since no main spec exists yet.
