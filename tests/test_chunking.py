"""Corpus loading, token-based splitting and the 10-key metadata contract.

Tier A: every test here runs with zero credentials and zero network. The real
splitter is only constructed in the two tests that are explicitly about token
sizing; everything else injects `FakeSplitter` and a width-neutral token counter.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import chunking
from rag.errors import CorpusTooSmallError, MetadataTooLargeError

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
GOLDEN_SET = ROOT / "data" / "golden_set.json"

STEMS = [
    "pytest-basics",
    "pytest-conftest",
    "pytest-coverage",
    "pytest-fixtures",
    "pytest-markers",
    "pytest-mocking",
    "pytest-parametrize",
    "pytest-plugins",
]

EXPECTED_KEYS = {
    "document_id",
    "chunk_id",
    "chunk_index",
    "source",
    "page",
    "section",
    "category",
    "tags",
    "token_count",
    "text",
}


def write_corpus(directory: Path, stems: list[str], body: str = "Body for {stem}.\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (directory / f"{stem}.md").write_text(f"# {stem}\n\n" + body.format(stem=stem), encoding="utf-8")
    return directory


def counted_tokens(text: str) -> int:
    """Deterministic, dependency-free token counter for the injected path."""
    return max(1, len(text.split()))


@pytest.fixture
def valid_metadata() -> dict[str, object]:
    return chunking.build_metadata(
        document_id="pytest-fixtures",
        chunk_index=0,
        text="Fixtures provide reusable setup code.",
        source="data/corpus/pytest-fixtures.md",
        section="Fixture scopes",
        category="fixtures",
        tags=["pytest", "fixtures"],
        token_count=7,
    )


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #


def test_load_corpus_derives_document_ids_from_filename_stems(tmp_path):
    directory = write_corpus(tmp_path / "corpus", STEMS)

    documents = chunking.load_corpus(directory)

    assert [document.metadata["document_id"] for document in documents] == STEMS
    assert all(document.page_content.startswith("# pytest-") for document in documents)


def test_load_corpus_records_posix_source_category_and_tags(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"])

    (document,) = chunking.load_corpus(directory, min_documents=1)

    assert document.metadata["source"] == (directory / "pytest-fixtures.md").as_posix()
    assert document.metadata["category"] == "fixtures"
    assert document.metadata["tags"] == ["pytest", "fixtures"]


def test_corpus_floor_rejects_five_documents(tmp_path):
    directory = write_corpus(tmp_path / "corpus", STEMS[:5])

    with pytest.raises(CorpusTooSmallError) as excinfo:
        chunking.load_corpus(directory)

    message = str(excinfo.value)
    assert "5" in message
    assert "6" in message


def test_corpus_floor_message_names_the_directory(tmp_path):
    directory = write_corpus(tmp_path / "corpus", STEMS[:2])

    with pytest.raises(CorpusTooSmallError) as excinfo:
        chunking.load_corpus(directory, min_documents=8)

    assert str(directory) in str(excinfo.value)


def test_committed_corpus_contains_every_golden_set_document():
    golden_ids = {
        document_id
        for entry in json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
        for document_id in entry["relevant_document_ids"]
    }

    documents = chunking.load_corpus(CORPUS_DIR)

    assert len(documents) == 8
    assert {document.metadata["document_id"] for document in documents} == golden_ids


# --------------------------------------------------------------------------- #
# Metadata schema
# --------------------------------------------------------------------------- #


def test_metadata_schema_is_exactly_ten_flat_keys(valid_metadata):
    assert set(valid_metadata) == EXPECTED_KEYS
    assert valid_metadata["chunk_id"] == "pytest-fixtures#000"
    assert valid_metadata["chunk_index"] == 0
    assert valid_metadata["page"] == 0
    assert type(valid_metadata["page"]) is int
    assert valid_metadata["tags"] == ["pytest", "fixtures"]
    assert all(type(tag) is str for tag in valid_metadata["tags"])
    assert valid_metadata["text"] == "Fixtures provide reusable setup code."
    assert valid_metadata["token_count"] == 7


def test_page_defaults_to_integer_zero_for_markdown(valid_metadata):
    assert valid_metadata["page"] == 0
    assert isinstance(valid_metadata["page"], int)
    assert not isinstance(valid_metadata["page"], bool)


def test_chunk_ids_are_document_scoped_and_sequential(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"], body="x" * 300)
    (document,) = chunking.load_corpus(directory, min_documents=1)

    chunks = chunking.split_documents(
        [document],
        splitter=SimpleNamespace(
            split_documents=lambda docs: [
                SimpleNamespace(page_content=text, metadata={"start_index": start})
                for start, text in [(0, "a" * 100), (100, "b" * 100), (200, "c" * 100)]
            ]
        ),
        count_tokens=counted_tokens,
    )

    assert [chunk.id for chunk in chunks] == [
        "pytest-fixtures#000",
        "pytest-fixtures#001",
        "pytest-fixtures#002",
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == [chunk.id for chunk in chunks]
    assert [chunk.text for chunk in chunks] == ["a" * 100, "b" * 100, "c" * 100]


def test_split_documents_keeps_document_metadata_on_every_chunk(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-markers"], body="y" * 250)
    (document,) = chunking.load_corpus(directory, min_documents=1)

    chunks = chunking.split_documents(
        [document], splitter=_fake_splitter(100, 0), count_tokens=counted_tokens
    )

    assert len(chunks) == 3
    for chunk in chunks:
        assert set(chunk.metadata) == EXPECTED_KEYS
        assert chunk.metadata["document_id"] == "pytest-markers"
        assert chunk.metadata["source"] == (directory / "pytest-markers.md").as_posix()
        assert chunk.metadata["category"] == "markers"
        assert chunk.metadata["token_count"] == counted_tokens(chunk.text)


def _fake_splitter(chunk_size: int, chunk_overlap: int):
    """Local char splitter: keeps this file independent of conftest internals."""

    class _Splitter:
        def split_documents(self, documents):
            step = max(chunk_size - chunk_overlap, 1)
            pieces = []
            for document in documents:
                for start in range(0, len(document.page_content), step):
                    text = document.page_content[start : start + chunk_size]
                    pieces.append(
                        SimpleNamespace(
                            page_content=text,
                            metadata={**document.metadata, "start_index": start},
                        )
                    )
            return pieces

    return _Splitter()


def _paragraph_splitter():
    """One chunk per blank-line-separated paragraph, so markers never straddle."""

    class _Splitter:
        def split_documents(self, documents):
            pieces = []
            for document in documents:
                offset = 0
                for paragraph in document.page_content.split("\n\n"):
                    pieces.append(
                        SimpleNamespace(
                            page_content=paragraph,
                            metadata={**document.metadata, "start_index": offset},
                        )
                    )
                    offset += len(paragraph) + 2
            return pieces

    return _Splitter()


# --------------------------------------------------------------------------- #
# Token-based sizing
# --------------------------------------------------------------------------- #


def test_splitter_uses_the_cl100k_base_encoder(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    requested: list[str] = []

    def fake_get_encoding(name, *args, **kwargs):
        requested.append(name)
        return SimpleNamespace(encode=lambda text, **kw: ["a", "b", "c"])

    monkeypatch.setattr(tiktoken, "get_encoding", fake_get_encoding)

    splitter = chunking.make_splitter()

    assert requested == ["cl100k_base"]
    assert splitter._chunk_size == 600
    assert splitter._chunk_overlap == 80
    assert splitter._add_start_index is True
    measured = splitter._length_function("abcdefghij")
    assert measured == 3
    assert measured != len("abcdefghij")


def test_chunks_are_sized_in_tokens_not_characters(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - environment without encoder data
        pytest.skip(f"tiktoken cl100k_base data unavailable offline: {exc}")

    paragraph = "Pytest fixtures provide reusable setup and teardown code for tests.\n\n"
    text = encoding.decode(encoding.encode(paragraph * 200)[:1500])
    assert len(encoding.encode(text)) == 1500

    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"], body="")
    (directory / "pytest-fixtures.md").write_text(text, encoding="utf-8")
    (document,) = chunking.load_corpus(directory, min_documents=1)

    chunks = chunking.split_documents([document])

    assert 2 <= len(chunks) <= 5
    assert all(0 < chunk.metadata["token_count"] <= 800 for chunk in chunks)
    assert sum(chunk.metadata["token_count"] for chunk in chunks) >= 1400
    assert len(chunks) < math.ceil(len(text) / chunking.DEFAULT_CHUNK_SIZE)


# --------------------------------------------------------------------------- #
# Section resolution
# --------------------------------------------------------------------------- #


MARKED_DOCUMENT = (
    "Preamble paragraph without any heading.\n\n"
    "# pytest fixtures\n\n"
    "Intro paragraph about fixtures.\n\n"
    "## Fixture scopes\n\n"
    "SCOPE MARKER: function scope is the default.\n\n"
    "## Yield fixtures\n\n"
    "YIELD MARKER: teardown runs after the test.\n"
)


def test_section_is_the_nearest_preceding_heading(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"], body="")
    (directory / "pytest-fixtures.md").write_text(MARKED_DOCUMENT, encoding="utf-8")
    (document,) = chunking.load_corpus(directory, min_documents=1)

    chunks = chunking.split_documents(
        [document], splitter=_paragraph_splitter(), count_tokens=counted_tokens
    )

    scope_chunk = next(chunk for chunk in chunks if "SCOPE MARKER" in chunk.text)
    yield_chunk = next(chunk for chunk in chunks if "YIELD MARKER" in chunk.text)
    assert scope_chunk.metadata["section"] == "Fixture scopes"
    assert yield_chunk.metadata["section"] == "Yield fixtures"


def test_section_is_empty_before_the_first_heading(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"], body="")
    (directory / "pytest-fixtures.md").write_text(MARKED_DOCUMENT, encoding="utf-8")
    (document,) = chunking.load_corpus(directory, min_documents=1)

    chunks = chunking.split_documents(
        [document], splitter=_paragraph_splitter(), count_tokens=counted_tokens
    )

    preamble = next(chunk for chunk in chunks if "Preamble" in chunk.text)
    assert preamble.metadata["section"] == ""


def test_section_is_empty_for_a_document_without_headings(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-basics"], body="")
    (directory / "pytest-basics.md").write_text("Plain prose, no headings at all.", encoding="utf-8")
    (document,) = chunking.load_corpus(directory, min_documents=1)

    (chunk,) = chunking.split_documents(
        [document], splitter=_fake_splitter(500, 0), count_tokens=counted_tokens
    )

    assert chunk.metadata["section"] == ""


# --------------------------------------------------------------------------- #
# 40 KB guard
# --------------------------------------------------------------------------- #


def test_metadata_size_guard_rejects_oversized_record(valid_metadata):
    valid_metadata["text"] = "x" * 41_000

    with pytest.raises(MetadataTooLargeError) as excinfo:
        chunking.validate_metadata(valid_metadata)

    message = str(excinfo.value)
    assert "40000" in message
    assert "pytest-fixtures#000" in message


def test_validate_metadata_accepts_a_record_within_the_limit(valid_metadata):
    payload = json.dumps(valid_metadata, ensure_ascii=False).encode("utf-8")

    assert len(payload) < chunking.MAX_METADATA_BYTES
    chunking.validate_metadata(valid_metadata)


def test_split_documents_rejects_an_oversized_chunk(tmp_path):
    directory = write_corpus(tmp_path / "corpus", ["pytest-fixtures"], body="z" * 41_000)
    (document,) = chunking.load_corpus(directory, min_documents=1)

    with pytest.raises(MetadataTooLargeError):
        chunking.split_documents([document], splitter=_fake_splitter(50_000, 0), count_tokens=counted_tokens)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda metadata: metadata.update(extra=1), "extra"),
        (lambda metadata: metadata.pop("page"), "page"),
        (lambda metadata: metadata.update(section=None), "section"),
        (lambda metadata: metadata.update(page=False), "page"),
        (lambda metadata: metadata.update(tags=[1, 2]), "tags"),
        (lambda metadata: metadata.update(tags="pytest"), "tags"),
        (lambda metadata: metadata.update(section={"title": "nested"}), "section"),
        (lambda metadata: metadata.update(token_count="7"), "token_count"),
    ],
)
def test_validate_metadata_rejects_schema_violations(valid_metadata, mutation, field):
    mutation(valid_metadata)

    with pytest.raises(MetadataTooLargeError) as excinfo:
        chunking.validate_metadata(valid_metadata)

    assert field in str(excinfo.value)


# --------------------------------------------------------------------------- #
# Import tier
# --------------------------------------------------------------------------- #


def test_importing_chunking_pulls_in_no_third_party_modules():
    code = (
        "import sys, rag.chunking;"
        "banned={'tiktoken','langchain_text_splitters','langchain_core','langchain_openai',"
        "'pinecone','openai','dotenv'};"
        "print(sorted(banned & set(sys.modules)))"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
