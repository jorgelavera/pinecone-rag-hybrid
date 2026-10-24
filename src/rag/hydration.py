"""Rebuild the BM25 corpus from Pinecone, with a disk cache and a local fallback.

Algorithm (design §5), deliberately *not* streamed per page:

1. read ``describe_index_stats()`` for the configured namespace;
2. return the on-disk cache when key **and** exact vector count still match;
3. otherwise collect every id with ``index.list(namespace=...)`` and fetch them
   in ``<= 1000``-id batches — 1,500 ids are exactly two calls (1000 + 500);
4. pop ``text`` from each vector's metadata into ``Document.page_content``;
5. on any Pinecone failure (or an empty namespace) re-chunk the local corpus;
6. if that also fails, raise :class:`HydrationError` naming the cache path and
   the remediation.

``langchain_core`` is imported inside the helpers so ``import rag.hydration``
stays credential-free and network-free.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag import chunking
from rag.config import Settings
from rag.errors import HydrationError

CACHE_FILENAME = "bm25_corpus.json"
FETCH_BATCH_SIZE = 1_000
METADATA_TEXT_KEY = "text"

logger = logging.getLogger(__name__)


def build_corpus(
    *,
    index,
    settings: Settings,
    cache_path: Path | None = None,
    local_corpus_dir: Path | None = None,
) -> list[object]:
    """Return the BM25 corpus as ``Document`` objects (``text`` popped into content).

    ``namespace`` is always passed explicitly: ``None`` means Pinecone's default
    empty namespace and would silently hydrate an empty corpus.
    """
    namespace = settings.namespace
    resolved_cache = Path(cache_path) if cache_path is not None else _default_cache_path(settings)
    resolved_corpus = Path(local_corpus_dir) if local_corpus_dir is not None else Path(settings.corpus_dir)

    try:
        vector_count = _vector_count(index.describe_index_stats(), namespace)
    except Exception as exc:  # noqa: BLE001 - any stats failure must fall back
        logger.warning("Pinecone stats failed for namespace %s (%s); using the local corpus", namespace, exc)
        return _local_documents(resolved_corpus, resolved_cache)

    cache_key = {
        "index_name": settings.index_name,
        "namespace": namespace,
        "pipeline_version": settings.pipeline_version,
        "page_count": _page_count(vector_count),
    }

    cached = _read_cache(resolved_cache)
    if cached is not None and _cache_matches(cached, cache_key, vector_count):
        return _documents_from_cache(cached)

    if vector_count == 0:
        logger.warning("Namespace %s holds no vectors; re-chunking %s", namespace, resolved_corpus)
        return _local_documents(resolved_corpus, resolved_cache)

    try:
        ids = [vector_id for page in index.list(namespace=namespace) for vector_id in page]
        if not ids:
            raise LookupError(f"namespace {namespace!r} listed zero ids despite a non-zero vector count")

        documents: list[object] = []
        for start in range(0, len(ids), FETCH_BATCH_SIZE):
            batch = ids[start : start + FETCH_BATCH_SIZE]
            response = index.fetch(ids=batch, namespace=namespace)
            for vector_id, record in response.vectors.items():
                metadata = dict(record.metadata or {})
                text = metadata.pop(METADATA_TEXT_KEY, "")
                documents.append(_document(vector_id, text, metadata))
    except Exception as exc:  # noqa: BLE001 - any Pinecone failure must fall back
        logger.warning("Pinecone hydration failed for namespace %s (%s); using the local corpus", namespace, exc)
        return _local_documents(resolved_corpus, resolved_cache)

    if not documents:
        logger.warning("Hydration returned no documents for namespace %s; using the local corpus", namespace)
        return _local_documents(resolved_corpus, resolved_cache)

    _write_cache(resolved_cache, cache_key, vector_count, documents)
    return documents


def _vector_count(stats, namespace: str) -> int:
    namespaces = getattr(stats, "namespaces", None) or {}
    entry = namespaces.get(namespace) if hasattr(namespaces, "get") else None
    if entry is None:
        return 0
    count = getattr(entry, "vector_count", None)
    if count is None and isinstance(entry, dict):
        count = entry.get("vector_count")
    return int(count or 0)


def _page_count(vector_count: int) -> int:
    return (vector_count + FETCH_BATCH_SIZE - 1) // FETCH_BATCH_SIZE


def _default_cache_path(settings: Settings) -> Path:
    return Path(settings.artifacts_dir) / CACHE_FILENAME


def _document(vector_id: str, text: str, metadata: dict[str, object]):
    from langchain_core.documents import Document

    return Document(id=vector_id, page_content=text, metadata=metadata)


def _read_cache(cache_path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("cache_key"), dict) or "vector_count" not in payload:
        return None
    if not isinstance(payload.get("documents"), list):
        return None
    return payload


def _cache_matches(cached: dict[str, object], cache_key: dict[str, object], vector_count: int) -> bool:
    return cached.get("cache_key") == cache_key and cached.get("vector_count") == vector_count


def _documents_from_cache(cached: dict[str, object]) -> list[object]:
    return [
        _document(
            str(entry.get("id", "")),
            str(entry.get("text", "")),
            dict(entry.get("metadata") or {}),
        )
        for entry in cached["documents"]
    ]


def _write_cache(
    cache_path: Path,
    cache_key: dict[str, object],
    vector_count: int,
    documents: list[object],
) -> None:
    payload = {
        "cache_key": cache_key,
        "vector_count": vector_count,
        "documents": [
            {"id": document.id, "text": document.page_content, "metadata": dict(document.metadata)}
            for document in documents
        ],
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        # The cache is disposable: a read-only artifacts directory must not
        # break hydration, it only makes the next run slower.
        logger.warning("Could not write the hydration cache at %s (%s)", cache_path, exc)


def _local_documents(corpus_dir: Path, cache_path: Path) -> list[object]:
    """Re-chunk the local corpus; the source of truth when Pinecone is unusable."""
    try:
        documents = chunking.load_corpus(corpus_dir)
        chunks = chunking.split_documents(documents)
    except Exception as exc:
        raise HydrationError(
            f"BM25 hydration failed: Pinecone was unusable and {corpus_dir} could not be re-chunked. "
            f"Delete the stale cache at {cache_path} (or bump PIPELINE_VERSION), run "
            f"scripts/ingest.py, and retry. Original error: {exc}"
        ) from exc

    return [
        _document(chunk.id, chunk.text, {key: value for key, value in chunk.metadata.items() if key != METADATA_TEXT_KEY})
        for chunk in chunks
    ]
