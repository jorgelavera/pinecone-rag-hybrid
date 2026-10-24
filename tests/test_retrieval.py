"""Hybrid retrieval contract for RAGSystem.

Regression-proof cases:
* ``[:5]`` removed would return all 9; duplicates removed would leak repeats.
* relying on EnsembleRetriever's k would be insufficient (we slice ourselves);
* a ``[1.0]`` ensemble would hide weight validation in single-mode branches
  (we test direct retriever usage instead);
* empty results must raise ``RetrievalEmptyError`` naming index and namespace.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from conftest import FakeEmbeddings, FakeIndex, FakeRetriever, FakeVectorStore
from rag.config import load_settings
from rag.errors import ConfigError, RetrievalEmptyError
from rag.retrieval import RAGSystem, RetrievedDocument


def settings(**overrides):
    environment = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "INDEX_NAME": "rag-pytest",
        **overrides,
    }
    return load_settings(env=environment)


def chunk(document_id: str, chunk_index: int = 0, text: str | None = None) -> Document:
    chunk_id = f"{document_id}#{chunk_index:03d}"
    return Document(
        id=chunk_id,
        page_content=text or f"{document_id} body",
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "source": f"data/corpus/{document_id}.md",
            "page": 0,
            "section": "x",
            "category": "fixtures",
            "tags": ["pytest"],
            "token_count": 10,
        },
    )


class RecordingEnsemble:
    """Stand-in for ``EnsembleRetriever`` that records construction args and yields queued docs."""

    instances: list["RecordingEnsemble"] = []

    def __init__(self, retrievers, weights=None, c=None, id_key=None, **kwargs):
        self.retrievers = retrievers
        self.weights = weights
        self.c = c
        self.id_key = id_key
        self.kwargs = kwargs
        self.documents: list[Document] = []
        RecordingEnsemble.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()

    def invoke(self, query: str):
        return list(self.documents)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


def hydrate(monkeypatch, corpus):
    def fake_build_corpus(**kwargs):
        return list(corpus)

    monkeypatch.setattr("rag.retrieval.build_corpus", fake_build_corpus)


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def test_hybrid_builds_ensemble_with_defaults_and_k20(tmp_path, monkeypatch):
    RecordingEnsemble.reset()
    hydrate(monkeypatch, [chunk("pytest-fixtures")])
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    index = FakeIndex()
    store = FakeVectorStore()
    system = RAGSystem(
        settings=settings(),
        index=index,
        embeddings=FakeEmbeddings(),
        vector_store=store,
        corpus=[chunk("pytest-fixtures")],
        mode="hybrid",
    )

    assert len(RecordingEnsemble.instances) == 1
    ensemble = RecordingEnsemble.instances[0]
    assert [getattr(r, "k", None) for r in ensemble.retrievers] == [20, 20]
    assert ensemble.weights == [0.4, 0.6]
    assert ensemble.id_key == "document_id"
    assert system is not None


def test_invalid_weights_still_rejected_by_settings():
    with pytest.raises(Exception) as excinfo:
        load_settings(env={"PINECONE_API_KEY": "k", "OPENAI_API_KEY": "k", "INDEX_NAME": "idx", "ENSEMBLE_BM25_WEIGHT": "4", "ENSEMBLE_VECTOR_WEIGHT": "6"})
    assert "sum to 1.0" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Fusion, dedup, slicing
# --------------------------------------------------------------------------- #


def test_over_return_with_duplicate_document_ids_is_deduplicated_to_five_in_fused_order(monkeypatch):
    RecordingEnsemble.reset()
    corpus = [chunk("doc-a"), chunk("doc-b"), chunk("doc-c"), chunk("doc-d"), chunk("doc-e"), chunk("doc-f")]
    hydrate(monkeypatch, corpus)
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    system = RAGSystem(settings=settings(), index=FakeIndex(), embeddings=FakeEmbeddings(), vector_store=FakeVectorStore(), corpus=corpus, mode="hybrid")
    ensemble = RecordingEnsemble.instances[0]
    # 9 documents: doc-a appears three times, order is the fused ranking.
    ensemble.documents = [
        chunk("doc-a", 0),
        chunk("doc-b", 0),
        chunk("doc-a", 1),
        chunk("doc-c", 0),
        chunk("doc-d", 0),
        chunk("doc-a", 2),
        chunk("doc-e", 0),
        chunk("doc-f", 0),
        chunk("doc-g", 0),
    ]

    results = system.retrieve("fixture scope", k=5)

    assert len(results) == 5
    assert [result.document_id for result in results] == ["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"]
    assert len({result.document_id for result in results}) == 5
    assert all(isinstance(result, RetrievedDocument) for result in results)
    assert results[0].id == "doc-a#000"
    assert results[0].text == "doc-a body"


def test_fewer_than_five_distinct_documents_returns_them_without_padding(monkeypatch):
    RecordingEnsemble.reset()
    corpus = [chunk("doc-a"), chunk("doc-b"), chunk("doc-c")]
    hydrate(monkeypatch, corpus)
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    system = RAGSystem(settings=settings(), index=FakeIndex(), embeddings=FakeEmbeddings(), vector_store=FakeVectorStore(), corpus=corpus, mode="hybrid")
    ensemble = RecordingEnsemble.instances[0]
    ensemble.documents = [chunk("doc-a", 0), chunk("doc-b", 0), chunk("doc-a", 1), chunk("doc-c", 0)]

    results = system.retrieve("test", k=5)

    assert len(results) == 3
    assert [result.document_id for result in results] == ["doc-a", "doc-b", "doc-c"]


# --------------------------------------------------------------------------- #
# Modes and errors
# --------------------------------------------------------------------------- #


def test_vector_mode_uses_a_single_retriever_without_a_one_weight_ensemble(monkeypatch):
    RecordingEnsemble.reset()
    corpus = [chunk("doc-a")]
    hydrate(monkeypatch, corpus)
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    store = FakeVectorStore(documents=[SimpleNamespace(page_content="x", metadata={"document_id": "doc-a", "chunk_id": "doc-a#000"})])
    store.retriever.documents = [chunk("doc-a", 0)]

    system = RAGSystem(settings=settings(), index=FakeIndex(), embeddings=FakeEmbeddings(), vector_store=store, corpus=corpus, mode="vector")

    assert RecordingEnsemble.instances == []
    results = system.retrieve("x", k=5)
    assert [result.document_id for result in results] == ["doc-a"]


def test_bm25_mode_uses_a_single_retriever_without_a_one_weight_ensemble(monkeypatch):
    RecordingEnsemble.reset()
    corpus = [chunk("doc-b")]
    hydrate(monkeypatch, corpus)
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    system = RAGSystem(settings=settings(), index=FakeIndex(), embeddings=FakeEmbeddings(), vector_store=FakeVectorStore(), corpus=corpus, mode="bm25")

    assert RecordingEnsemble.instances == []
    results = system.retrieve("body", k=5)
    assert [result.document_id for result in results] == ["doc-b"]


def test_zero_results_raise_retrieval_empty_error_naming_index_and_namespace(monkeypatch):
    RecordingEnsemble.reset()
    hydrate(monkeypatch, [chunk("doc-a")])
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    system = RAGSystem(settings=settings(), index=FakeIndex(), embeddings=FakeEmbeddings(), vector_store=FakeVectorStore(), corpus=[chunk("doc-a")], mode="hybrid")
    ensemble = RecordingEnsemble.instances[0]
    ensemble.documents = []

    with pytest.raises(RetrievalEmptyError) as excinfo:
        system.retrieve("conftest", k=5)

    message = str(excinfo.value)
    assert "rag-pytest" in message
    assert "rag-docs" in message


# --------------------------------------------------------------------------- #
# Fail-fast: vector/hybrid modes need a real vector retriever
# --------------------------------------------------------------------------- #


def test_hybrid_mode_without_any_vector_source_raises_config_error(monkeypatch):
    RecordingEnsemble.reset()
    hydrate(monkeypatch, [chunk("doc-a")])
    monkeypatch.setattr("langchain_classic.retrievers.EnsembleRetriever", RecordingEnsemble)

    # FakeIndex has no similarity_search and no vector_store is passed: the old
    # _EmptyVectorRetriever placeholder made this construction silently "work".
    with pytest.raises(ConfigError) as excinfo:
        RAGSystem(settings=settings(), index=FakeIndex(), corpus=[chunk("doc-a")], mode="hybrid")

    message = str(excinfo.value)
    assert "vector_store" in message
    assert "hybrid" in message


def test_vector_mode_without_any_vector_source_raises_config_error(monkeypatch):
    hydrate(monkeypatch, [chunk("doc-a")])

    with pytest.raises(ConfigError) as excinfo:
        RAGSystem(settings=settings(), corpus=[chunk("doc-a")], mode="vector")

    assert "vector_store" in str(excinfo.value)


def test_bm25_mode_works_with_corpus_only(monkeypatch):
    hydrate(monkeypatch, [chunk("doc-b")])

    system = RAGSystem(settings=settings(), corpus=[chunk("doc-b")], mode="bm25")

    results = system.retrieve("body", k=5)
    assert [result.document_id for result in results] == ["doc-b"]
