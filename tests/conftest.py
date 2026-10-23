"""Shared pytest configuration and credential-free test doubles.

Nothing here imports `openai`, `pinecone` or any `langchain` package, so the whole
suite can run with both API keys unset.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

INTEGRATION_KEYS = ("PINECONE_API_KEY", "OPENAI_API_KEY")


def pytest_collection_modifyitems(config, items):
    """Skip `@pytest.mark.integration` tests unless both live keys are present."""
    if all(os.environ.get(name) for name in INTEGRATION_KEYS):
        return
    skip = pytest.mark.skip(reason="integration tests need PINECONE_API_KEY and OPENAI_API_KEY")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


class FakeEmbeddings:
    """Deterministic hash embeddings with a configurable dimension."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255.0 for index in range(self.dimensions)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeRetriever:
    """Returns a queued list of documents and records every query."""

    def __init__(self, documents: list[SimpleNamespace] | None = None):
        self.documents = list(documents or [])
        self.queries: list[str] = []

    def invoke(self, query: str):
        self.queries.append(query)
        return list(self.documents)

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


class FakeVectorStore:
    """Minimal vector store: `add_texts` batching plus `as_retriever`."""

    def __init__(self, documents: list[SimpleNamespace] | None = None):
        self.documents = list(documents or [])
        self.retriever = FakeRetriever(self.documents)
        self.retriever_kwargs: dict[str, object] = {}
        self.add_texts_calls: list[dict[str, object]] = []

    def as_retriever(self, **kwargs):
        self.retriever_kwargs = dict(kwargs)
        return self.retriever

    def add_texts(self, texts, metadatas=None, ids=None, batch_size=None, namespace=None):
        self.add_texts_calls.append(
            {
                "texts": list(texts),
                "metadatas": list(metadatas or []),
                "ids": list(ids or []),
                "batch_size": batch_size,
                "namespace": namespace,
            }
        )
        return list(ids or [])


@dataclass
class FakeIndex:
    """Records `list` / `fetch` / `describe_index_stats` calls.

    `fetch` honours Pinecone's 1000-id request ceiling the way the real client
    does: it returns everything asked for in that single call.
    """

    vectors: dict[str, SimpleNamespace] = field(default_factory=dict)
    namespace_counts: dict[str, int] = field(default_factory=dict)
    page_size: int = 100
    list_calls: list[dict[str, object]] = field(default_factory=list)
    fetch_calls: list[dict[str, object]] = field(default_factory=list)
    stats_calls: int = 0

    def describe_index_stats(self, *args, **kwargs):
        self.stats_calls += 1
        namespaces = {
            namespace: SimpleNamespace(vector_count=count)
            for namespace, count in self.namespace_counts.items()
        }
        return SimpleNamespace(namespaces=namespaces, total_vector_count=sum(self.namespace_counts.values()))

    def list(self, namespace: str | None = None, **kwargs):
        self.list_calls.append({"namespace": namespace, **kwargs})
        ids = list(self.vectors)
        for start in range(0, len(ids), self.page_size):
            yield ids[start : start + self.page_size]

    def fetch(self, ids, namespace: str | None = None, **kwargs):
        requested = list(ids)
        self.fetch_calls.append({"ids": requested, "namespace": namespace, **kwargs})
        found = {vector_id: self.vectors[vector_id] for vector_id in requested if vector_id in self.vectors}
        return SimpleNamespace(vectors=found)


class FakeSplitter:
    """Character-based splitter so tests never need tiktoken."""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        step = max(self.chunk_size - self.chunk_overlap, 1)
        return [text[start : start + self.chunk_size] for start in range(0, len(text), step)]

    def split_documents(self, documents):
        chunks = []
        for document in documents:
            for text in self.split_text(document.page_content):
                chunks.append(SimpleNamespace(page_content=text, metadata=dict(document.metadata)))
        return chunks


@pytest.fixture
def fake_index() -> FakeIndex:
    return FakeIndex()


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()
