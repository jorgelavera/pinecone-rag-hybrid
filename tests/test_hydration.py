"""BM25 corpus hydration: collect all ids, then fetch in exact 1000-id batches.

Tier B: the Pinecone index is faked (``tests/conftest.py``) and the local
fallback uses a character splitter, so nothing here touches the network and no
credentials are required.

Regression-proof cases:
* streaming one fetch per page would produce 15 fetches for 1,500 ids (we pin 2);
* omitting ``namespace=`` would fail the recorded-argument assertions;
* a cache key that ignores ``vector_count`` would return the stale one-document
  cache and fail the 1,200-document assertion.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeIndex, FakeSplitter
from rag.chunking import METADATA_KEYS
from rag.config import load_settings
from rag.errors import HydrationError
from rag.hydration import build_corpus

CACHE_KEY_FIELDS = {"index_name", "namespace", "pipeline_version", "page_count"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def settings(tmp_path: Path, **overrides):
    environment = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "INDEX_NAME": "rag-pytest",
        "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "CORPUS_DIR": str(tmp_path / "corpus"),
        **overrides,
    }
    return load_settings(env=environment)


def vector(chunk_id: str, document_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_index": int(chunk_id.rsplit("#", 1)[1]),
            "source": f"data/corpus/{document_id}.md",
            "page": 0,
            "section": "Fixture scopes",
            "category": "fixtures",
            "tags": ["pytest", "fixtures"],
            "token_count": 42,
            "text": text,
        }
    )


def populate(index: FakeIndex, count: int, *, prefix: str = "pytest-fixtures") -> FakeIndex:
    """Fill the fake namespace with ``count`` chunk vectors in insertion order."""
    for position in range(count):
        chunk_id = f"{prefix}#{position:04d}"
        index.vectors[chunk_id] = vector(chunk_id, prefix, f"text-{position:04d}")
    index.namespace_counts["rag-docs"] = count
    return index


def write_cache(path: Path, *, page_count: int, vector_count: int, documents) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cache_key": {
                    "index_name": "rag-pytest",
                    "namespace": "rag-docs",
                    "pipeline_version": "1",
                    "page_count": page_count,
                },
                "vector_count": vector_count,
                "documents": [
                    {"id": chunk_id, "text": text, "metadata": {"document_id": document_id}}
                    for chunk_id, document_id, text in documents
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def write_local_corpus(directory: Path, count: int = 6) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stems = [
        "pytest-basics",
        "pytest-conftest",
        "pytest-coverage",
        "pytest-fixtures",
        "pytest-markers",
        "pytest-mocking",
        "pytest-parametrize",
        "pytest-plugins",
    ][:count]
    for stem in stems:
        (directory / f"{stem}.md").write_text(
            f"# {stem}\n\n" + ("body text " * 40), encoding="utf-8"
        )
    return directory


def use_character_splitter(monkeypatch) -> None:
    """Keep the local fallback offline-deterministic: no tiktoken splitter."""
    monkeypatch.setattr(
        "rag.chunking.make_splitter",
        lambda **kwargs: FakeSplitter(chunk_size=200, chunk_overlap=0),
    )


class ExplodingOnListIndex(FakeIndex):
    """Records nothing and fails like a dropped Pinecone connection."""

    def list(self, namespace=None, **kwargs):
        raise ConnectionError("pinecone is unreachable")


class ExplodingOnStatsIndex(FakeIndex):
    def describe_index_stats(self, *args, **kwargs):
        raise ConnectionError("stats endpoint is unreachable")


# --------------------------------------------------------------------------- #
# Batched fetch
# --------------------------------------------------------------------------- #


def test_collects_all_ids_then_fetches_in_exact_1000_id_batches(tmp_path):
    index = populate(FakeIndex(), 1500)

    documents = build_corpus(index=index, settings=settings(tmp_path))

    assert len(documents) == 1500
    # 1,500 ids => ceil(1500/1000) = 2 calls: 1000 then 500.
    assert [len(call["ids"]) for call in index.fetch_calls] == [1000, 500]
    assert [call["namespace"] for call in index.fetch_calls] == ["rag-docs", "rag-docs"]
    assert [call["namespace"] for call in index.list_calls] == ["rag-docs"]

    first = documents[0]
    assert first.id == "pytest-fixtures#0000"
    assert first.page_content == "text-0000"
    assert "text" not in first.metadata
    assert set(first.metadata) == set(METADATA_KEYS) - {"text"}


def test_hydration_writes_a_cache_file_keyed_by_page_count_and_exact_count(tmp_path):
    index = populate(FakeIndex(), 1500)

    build_corpus(index=index, settings=settings(tmp_path))

    cache_path = tmp_path / "artifacts" / "bm25_corpus.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(payload["cache_key"]) == CACHE_KEY_FIELDS
    assert payload["cache_key"] == {
        "index_name": "rag-pytest",
        "namespace": "rag-docs",
        "pipeline_version": "1",
        "page_count": 2,
    }
    assert payload["vector_count"] == 1500
    assert len(payload["documents"]) == 1500
    assert set(payload["documents"][0]) == {"id", "text", "metadata"}


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def test_matching_cache_skips_list_and_fetch_entirely(tmp_path):
    index = populate(FakeIndex(), 40)
    cache_path = write_cache(
        tmp_path / "artifacts" / "bm25_corpus.json",
        page_count=1,
        vector_count=40,
        documents=[("pytest-fixtures#0000", "pytest-fixtures", "cached chunk text")],
    )

    documents = build_corpus(index=index, settings=settings(tmp_path), cache_path=cache_path)

    assert index.list_calls == []
    assert index.fetch_calls == []
    assert index.stats_calls == 1
    assert [document.id for document in documents] == ["pytest-fixtures#0000"]
    assert documents[0].page_content == "cached chunk text"


def test_cache_is_rejected_when_the_exact_vector_count_changed(tmp_path):
    # page_count is 2 both before and after: only the exact count exposes staleness.
    index = populate(FakeIndex(), 1200)
    cache_path = write_cache(
        tmp_path / "artifacts" / "bm25_corpus.json",
        page_count=2,
        vector_count=1150,
        documents=[("stale#0000", "stale", "stale chunk text")],
    )

    documents = build_corpus(index=index, settings=settings(tmp_path), cache_path=cache_path)

    assert len(index.list_calls) == 1
    assert [len(call["ids"]) for call in index.fetch_calls] == [1000, 200]
    assert len(documents) == 1200
    assert documents[0].id == "pytest-fixtures#0000"


def test_cache_is_rejected_when_the_page_count_in_the_key_differs(tmp_path):
    index = populate(FakeIndex(), 1000)
    cache_path = write_cache(
        tmp_path / "artifacts" / "bm25_corpus.json",
        page_count=2,
        vector_count=1000,
        documents=[("stale#0000", "stale", "stale chunk text")],
    )

    documents = build_corpus(index=index, settings=settings(tmp_path), cache_path=cache_path)

    assert [len(call["ids"]) for call in index.fetch_calls] == [1000]
    assert len(documents) == 1000


# --------------------------------------------------------------------------- #
# Local fallback chain
# --------------------------------------------------------------------------- #


def test_empty_namespace_skips_pinecone_and_rechunks_the_local_corpus(tmp_path, monkeypatch):
    use_character_splitter(monkeypatch)
    corpus_dir = write_local_corpus(tmp_path / "corpus")
    index = FakeIndex(namespace_counts={"rag-docs": 0})

    documents = build_corpus(
        index=index,
        settings=settings(tmp_path),
        local_corpus_dir=corpus_dir,
    )

    assert index.list_calls == []
    assert index.fetch_calls == []
    assert {document.metadata["document_id"] for document in documents} == {
        "pytest-basics",
        "pytest-conftest",
        "pytest-coverage",
        "pytest-fixtures",
        "pytest-markers",
        "pytest-mocking",
    }
    assert all(
        document.id == f"{document.metadata['document_id']}#{index:03d}"
        for document in documents
        for index in [int(document.id.rsplit("#", 1)[1])]
    )
    assert {document.id for document in documents if document.id.endswith("#000")} == {
        "pytest-basics#000",
        "pytest-conftest#000",
        "pytest-coverage#000",
        "pytest-fixtures#000",
        "pytest-markers#000",
        "pytest-mocking#000",
    }
    assert all("text" not in document.metadata for document in documents)


def test_a_pinecone_listing_error_falls_back_to_the_local_corpus(tmp_path, monkeypatch):
    use_character_splitter(monkeypatch)
    corpus_dir = write_local_corpus(tmp_path / "corpus")
    index = populate(ExplodingOnListIndex(), 10)

    documents = build_corpus(
        index=index,
        settings=settings(tmp_path),
        local_corpus_dir=corpus_dir,
    )

    assert index.fetch_calls == []
    assert len(documents) >= 6
    assert documents[0].page_content.startswith("# pytest-basics")


def test_a_stats_error_falls_back_to_the_local_corpus(tmp_path, monkeypatch):
    use_character_splitter(monkeypatch)
    corpus_dir = write_local_corpus(tmp_path / "corpus")
    index = populate(ExplodingOnStatsIndex(), 10)

    documents = build_corpus(
        index=index,
        settings=settings(tmp_path),
        local_corpus_dir=corpus_dir,
    )

    assert len(documents) >= 6


def test_double_failure_raises_hydration_error_naming_the_cache_and_remediation(tmp_path):
    cache_path = tmp_path / "artifacts" / "bm25_corpus.json"
    index = populate(ExplodingOnListIndex(), 10)

    with pytest.raises(HydrationError) as excinfo:
        build_corpus(
            index=index,
            settings=settings(tmp_path),
            cache_path=cache_path,
            local_corpus_dir=tmp_path / "missing-corpus",
        )

    message = str(excinfo.value)
    assert str(cache_path) in message
    assert "PIPELINE_VERSION" in message
    assert "scripts/ingest.py" in message
