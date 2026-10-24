"""Hybrid retrieval via ``RAGSystem`` with lazy LangChain imports.

Ensemble rules (design/spec):
* ``mode == "hybrid"`` uses ``langchain_classic.retrievers.EnsembleRetriever``
  with ``weights=[bm25_weight, vector_weight]`` (defaults [0.4, 0.6]),
  ``id_key="document_id"``, and ``k=20`` on both retrievers.
* ``mode == "vector"`` or ``"bm25"`` call the single retriever directly (no
  ``[1.0]`` ensemble so weight validation is not bypassed).
* ``retrieve`` runs the retriever, **deduplicates by ``document_id`` preserving
  fused order**, slices to ``[:k]`` itself, and raises
  :class:`RetrievalEmptyError` naming the index and namespace when the result
  is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag.config import Settings
from rag.errors import ConfigError, RetrievalEmptyError
from rag.hydration import build_corpus


@dataclass(frozen=True)
class RetrievedDocument:
    id: str
    document_id: str
    text: str
    metadata: dict[str, object]
    score: float | None = None


class RAGSystem:
    def __init__(
        self,
        *,
        settings: Settings,
        index=None,
        embeddings=None,
        vector_store=None,
        corpus: list[object] | None = None,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> None:
        self.settings = settings
        self.index = index
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.mode = mode

        self.corpus = list(corpus) if corpus is not None else build_corpus(index=index, settings=settings)
        self.bm25_retriever = _build_bm25_retriever(self.corpus)
        self.vector_retriever = _build_vector_retriever(vector_store, index)

        if mode in ("vector", "hybrid") and self.vector_retriever is None:
            raise ConfigError(
                f"mode={mode!r} needs a vector retriever but none could be built: pass "
                "vector_store= (e.g. rag.pinecone_client.build_vector_store(index, embeddings, "
                "settings)) or an index exposing similarity_search. mode='bm25' works with the "
                "corpus alone."
            )

        if mode == "hybrid":
            from langchain_classic.retrievers import EnsembleRetriever

            self.retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[settings.bm25_weight, settings.vector_weight],
                id_key="document_id",
            )
        elif mode == "vector":
            self.retriever = self.vector_retriever
        elif mode == "bm25":
            self.retriever = self.bm25_retriever
        else:
            raise ValueError(f"Unknown retrieval mode: {mode!r}")

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDocument]:
        documents = self._invoke_retriever(query)
        deduplicated: list[object] = []
        seen: set[str] = set()
        for document in documents:
            document_id = _document_id(document)
            if document_id in seen:
                continue
            seen.add(document_id)
            deduplicated.append(document)
            if len(deduplicated) == k:
                break

        if not deduplicated:
            raise RetrievalEmptyError(
                f"index={self.settings.index_name} namespace={self.settings.namespace}: "
                f"0 documents returned for query {query!r}; check the namespace, ingestion, or index configuration"
            )

        return [_to_retrieved_document(document) for document in deduplicated]

    def _invoke_retriever(self, query: str):
        retriever = self.retriever
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        if hasattr(retriever, "get_relevant_documents"):
            return retriever.get_relevant_documents(query)
        raise TypeError(f"Retriever {type(retriever)!r} has no invoke() or get_relevant_documents()")


def _build_bm25_retriever(corpus: list[object]):
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(corpus)
    retriever.k = 20
    return retriever


def _build_vector_retriever(vector_store, index):
    """Return a real vector retriever, or ``None`` when no vector source exists.

    ``None`` (handled fail-fast in ``RAGSystem.__init__``) replaces the old
    ``_EmptyVectorRetriever`` placeholder, whose empty results were later
    swallowed and hid the missing wiring.
    """
    if vector_store is not None and hasattr(vector_store, "as_retriever"):
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        except Exception:
            retriever = vector_store.as_retriever()
        if hasattr(retriever, "search_kwargs"):
            current = getattr(retriever, "search_kwargs") or {}
            retriever.search_kwargs = {**current, "k": 20}
        try:
            retriever.k = 20
        except Exception:
            pass
        return retriever

    if index is not None and hasattr(index, "similarity_search"):
        return index

    return None


def _document_id(document) -> str:
    metadata = getattr(document, "metadata", None) or {}
    if isinstance(metadata, dict):
        document_id = metadata.get("document_id")
        if document_id:
            return str(document_id)
    return str(getattr(document, "id", ""))


def _to_retrieved_document(document) -> RetrievedDocument:
    metadata = dict(getattr(document, "metadata", None) or {})
    document_id = _document_id(document)
    chunk_id = metadata.get("chunk_id") or getattr(document, "id", "")
    text = getattr(document, "page_content", None)
    if text is None and isinstance(metadata, dict):
        text = metadata.pop("text", "")
    return RetrievedDocument(
        id=str(chunk_id),
        document_id=str(document_id),
        text="" if text is None else str(text),
        metadata=metadata,
    )
