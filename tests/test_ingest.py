"""Ingest pipeline tests with fakes (Tier B).

Covers: corpus floor enforced before embedding, dimension assertion before
upsert, batching, chunk_id used as the vector id, and idempotent re-upsert.
No imports of `openai` or `pinecone`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.config import load_settings
from rag.errors import CorpusTooSmallError, DimensionMismatchError
from conftest import FakeEmbeddings, FakeIndex, FakeVectorStore


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ingest.py"


@pytest.fixture(scope="module")
def ingest():
    spec = importlib.util.spec_from_file_location("rag_script_ingest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrackingEmbeddings(FakeEmbeddings):
    def __init__(self, dimensions: int = 1536, bad_index: int | None = None):
        super().__init__(dimensions=dimensions)
        self.bad_index = bad_index
        self.embed_calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        vectors = super().embed_documents(texts)
        if self.bad_index is not None and 0 <= self.bad_index < len(vectors):
            vectors = [list(vector) for vector in vectors]
            vectors[self.bad_index] = vectors[self.bad_index][:1024]
        return vectors


def settings(**overrides):
    environment = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "INDEX_NAME": "rag-pytest",
        **overrides,
    }
    return load_settings(env=environment)


class FakePineconeClientForIngest:
    def __init__(self):
        self.index = FakeIndex(namespace_counts={"rag-docs": 0})
        self.index_calls: list[str] = []

    def Index(self, name: str):
        self.index_calls.append(name)
        return self.index


def test_ingest_enforces_corpus_floor_before_embedding(tmp_path, ingest):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for stem in ["pytest-basics", "pytest-conftest", "pytest-fixtures"]:
        (corpus_dir / f"{stem}.md").write_text("# Heading\n\nbody", encoding="utf-8")

    embedding_model = TrackingEmbeddings()
    client = FakePineconeClientForIngest()

    with pytest.raises(CorpusTooSmallError):
        ingest.main(
            client=client,
            settings=settings(),
            corpus_dir=corpus_dir,
            embedding_model=embedding_model,
        )

    assert embedding_model.embed_calls == []
    assert client.index_calls == []


def test_ingest_asserts_embedding_dimensions_before_upsert(tmp_path, ingest, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for stem in [
        "pytest-basics",
        "pytest-conftest",
        "pytest-coverage",
        "pytest-fixtures",
        "pytest-markers",
        "pytest-mocking",
    ]:
        (corpus_dir / f"{stem}.md").write_text("# Heading\n\n" + ("word " * 40), encoding="utf-8")

    embedding_model = TrackingEmbeddings(bad_index=0)
    client = FakePineconeClientForIngest()
    store = FakeVectorStore()

    def fake_build_embeddings(_settings):
        return embedding_model

    def fake_build_vector_store(_index, _embeddings, _settings):
        return store

    monkeypatch.setattr("rag.embeddings.build_embeddings", fake_build_embeddings)
    monkeypatch.setattr("rag.pinecone_client.build_vector_store", fake_build_vector_store)

    with pytest.raises(DimensionMismatchError) as excinfo:
        ingest.main(client=client, settings=settings(), corpus_dir=corpus_dir)

    message = str(excinfo.value)
    assert "1536" in message
    assert "1024" in message
    assert store.add_texts_calls == []


def test_ingest_batches_and_uses_chunk_id_as_vector_id(tmp_path, ingest, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for stem in [
        "pytest-basics",
        "pytest-conftest",
        "pytest-coverage",
        "pytest-fixtures",
        "pytest-markers",
        "pytest-mocking",
    ]:
        (corpus_dir / f"{stem}.md").write_text("# Heading\n\n" + ("word " * 40), encoding="utf-8")

    embedding_model = TrackingEmbeddings()
    client = FakePineconeClientForIngest()
    store = FakeVectorStore()

    def fake_build_embeddings(_settings):
        return embedding_model

    def fake_build_vector_store(_index, _embeddings, _settings):
        return store

    monkeypatch.setattr("rag.embeddings.build_embeddings", fake_build_embeddings)
    monkeypatch.setattr("rag.pinecone_client.build_vector_store", fake_build_vector_store)

    exit_code = ingest.main(client=client, settings=settings(), corpus_dir=corpus_dir)

    assert exit_code == 0
    assert len(store.add_texts_calls) == 1
    call = store.add_texts_calls[0]
    assert call["batch_size"] == 32
    assert call["namespace"] == "rag-docs"
    assert len(call["texts"]) == len(call["metadatas"]) == len(call["ids"])
    assert len(call["ids"]) == 6
    assert call["ids"][0].endswith("#000")
    assert all(id_ == metadata["chunk_id"] for id_, metadata in zip(call["ids"], call["metadatas"]))
    for metadata in call["metadatas"]:
        assert set(metadata) == {
            "document_id",
            "chunk_id",
            "chunk_index",
            "source",
            "page",
            "section",
            "category",
            "tags",
            "token_count",
        }
        assert "text" not in metadata


def test_ingest_is_idempotent_on_re_upsert(tmp_path, ingest, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for stem in [
        "pytest-basics",
        "pytest-conftest",
        "pytest-coverage",
        "pytest-fixtures",
        "pytest-markers",
        "pytest-mocking",
    ]:
        (corpus_dir / f"{stem}.md").write_text("# Heading\n\n" + ("word " * 40), encoding="utf-8")

    embedding_model = TrackingEmbeddings()
    client = FakePineconeClientForIngest()
    store = FakeVectorStore()

    def fake_build_embeddings(_settings):
        return embedding_model

    def fake_build_vector_store(_index, _embeddings, _settings):
        return store

    monkeypatch.setattr("rag.embeddings.build_embeddings", fake_build_embeddings)
    monkeypatch.setattr("rag.pinecone_client.build_vector_store", fake_build_vector_store)

    ingest.main(client=client, settings=settings(), corpus_dir=corpus_dir)
    ingest.main(client=client, settings=settings(), corpus_dir=corpus_dir)

    assert len(store.vectors) == 6
    assert len(store.add_texts_calls) == 2
    assert store.add_texts_calls[0]["ids"] == store.add_texts_calls[1]["ids"]
