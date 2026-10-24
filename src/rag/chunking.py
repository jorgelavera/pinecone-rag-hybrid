"""Corpus loading, token-based splitting, and the flat Pinecone metadata contract.

Tier A of the import map: nothing third-party is imported at module scope, so
``import rag.chunking`` works with no credentials and no network. ``tiktoken``,
``langchain_text_splitters`` and ``langchain_core`` are imported inside the
functions that actually need them.

The metadata schema is deliberately fixed at ten keys: nine flat fields plus the
verbatim ``text`` payload that ``PineconeVectorStore`` pops on retrieval.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.errors import CorpusTooSmallError, MetadataTooLargeError

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_MIN_DOCUMENTS = 6
MAX_CHUNK_TOKENS = 800
MAX_METADATA_BYTES = 40_000
ENCODING_NAME = "cl100k_base"
GENERAL_CATEGORY = "general"

METADATA_KEYS = (
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
)

#: Closed set used by the committed corpus. `load_corpus` derives it from the
#: `pytest-<topic>.md` stem, so a new file needs no registration; the constant
#: documents the intended vocabulary instead of enforcing it (a hard membership
#: check would reject any future corpus file).
CATEGORIES = frozenset(
    {"basics", "conftest", "coverage", "fixtures", "markers", "mocking", "parametrize", "plugins"}
)


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit. ``id`` is the Pinecone vector id (the ``chunk_id``)."""

    id: str
    text: str
    metadata: dict[str, object]


def load_corpus(corpus_dir: Path, *, min_documents: int = DEFAULT_MIN_DOCUMENTS) -> list[object]:
    """Load every ``*.md`` file in ``corpus_dir`` as one ``Document`` per file.

    ``document_id`` is the filename stem, so corpus files and Golden Set ids stay
    in lockstep. Fewer than ``min_documents`` files raises before any embedding
    or upsert can happen.
    """
    from langchain_core.documents import Document

    directory = Path(corpus_dir)
    paths = sorted(directory.glob("*.md")) if directory.is_dir() else []
    if len(paths) < min_documents:
        raise CorpusTooSmallError(
            f"{directory} holds {len(paths)} Markdown document(s); at least {min_documents} are required "
            "(the corpus floor keeps evaluation statistically meaningful)."
        )

    documents: list[object] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        document_id = path.stem
        category = _category_for(document_id)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "source": (directory / path.name).as_posix(),
                    "category": category,
                    "tags": ["pytest", category],
                    "headings": _heading_index(text),
                },
            )
        )
    return documents


def make_splitter(
    *, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
):
    """Build the token-counting splitter.

    ``from_tiktoken_encoder`` is what makes ``chunk_size`` a token budget: the
    default ``length_function`` is ``len``, which would split ~6x earlier.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def split_documents(
    documents: Sequence[object],
    *,
    splitter: object | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Split documents into validated :class:`Chunk` records, numbered per document."""
    resolved_splitter = splitter if splitter is not None else make_splitter()
    counter = count_tokens if count_tokens is not None else _tiktoken_counter()

    chunks: list[Chunk] = []
    for document in documents:
        metadata = document.metadata
        document_id = str(metadata["document_id"])
        headings = metadata.get("headings") or ()
        source = str(metadata.get("source", ""))
        category = str(metadata.get("category", ""))
        tags = list(metadata.get("tags") or [])

        for chunk_index, piece in enumerate(resolved_splitter.split_documents([document])):
            text = piece.page_content
            chunk_metadata = build_metadata(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
                source=source,
                section=_section_for(headings, piece.metadata.get("start_index")),
                category=category,
                tags=tags,
                token_count=counter(text),
            )
            validate_metadata(chunk_metadata)
            chunks.append(
                Chunk(id=str(chunk_metadata["chunk_id"]), text=text, metadata=chunk_metadata)
            )
    return chunks


def build_metadata(
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    source: str,
    section: str,
    category: str,
    tags: list[str],
    token_count: int,
    page: int = 0,
) -> dict[str, object]:
    """Build the exact ten-key record: nine schema fields plus the ``text`` payload."""
    return {
        "document_id": document_id,
        "chunk_id": f"{document_id}#{chunk_index:03d}",
        "chunk_index": chunk_index,
        "source": source,
        "page": page,
        "section": section,
        "category": category,
        "tags": list(tags),
        "token_count": token_count,
        "text": text,
    }


def validate_metadata(metadata: Mapping[str, object], *, max_bytes: int = MAX_METADATA_BYTES) -> None:
    """Enforce the schema and Pinecone's 40,000-byte per-record metadata limit.

    A record that cannot be stored is rejected here, before any embedding or
    upsert: wrong key set, ``None``, nested values, or a non-string tag are all
    metadata-contract violations and surface as :class:`MetadataTooLargeError`
    (the taxonomy has no separate schema error, and a bare ``ValueError`` would
    bypass the single ``RagError`` handler in the CLIs).
    """
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise MetadataTooLargeError(
            f"metadata for {metadata.get('chunk_id', '<unknown>')} is missing field(s): "
            + ", ".join(missing)
        )
    extra = sorted(set(metadata) - set(METADATA_KEYS))
    if extra:
        raise MetadataTooLargeError(
            f"metadata for {metadata.get('chunk_id', '<unknown>')} has unexpected field(s): "
            + ", ".join(extra)
        )

    for field_name in ("document_id", "chunk_id", "source", "section", "category", "text"):
        value = metadata[field_name]
        if type(value) is not str:
            raise MetadataTooLargeError(
                f"metadata field {field_name} must be a string, got {type(value).__name__}"
            )
    for field_name in ("chunk_index", "page", "token_count"):
        if type(metadata[field_name]) is not int:
            raise MetadataTooLargeError(
                f"metadata field {field_name} must be an int, got {type(metadata[field_name]).__name__}"
            )
    tags = metadata["tags"]
    if type(tags) is not list or any(type(tag) is not str for tag in tags):
        raise MetadataTooLargeError("metadata field tags must be a list[str]")

    payload = json.dumps(dict(metadata), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > max_bytes:
        raise MetadataTooLargeError(
            f"metadata for {metadata['chunk_id']} is {len(payload)} bytes, above the "
            f"{max_bytes}-byte Pinecone metadata limit; reduce CHUNK_SIZE"
        )


def _category_for(document_id: str) -> str:
    prefix = "pytest-"
    if document_id.startswith(prefix) and len(document_id) > len(prefix):
        return document_id[len(prefix) :]
    return GENERAL_CATEGORY


def _heading_index(text: str) -> tuple[tuple[int, str], ...]:
    """Map character offsets to Markdown heading titles, in document order."""
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                headings.append((offset, title))
        offset += len(line)
    return tuple(headings)


def _section_for(headings: Iterable[tuple[int, str]], start_index: object) -> str:
    """Return the nearest heading at or before the chunk start; ``""`` when none."""
    if not isinstance(start_index, int):
        return ""
    section = ""
    for offset, title in headings:
        if offset > start_index:
            break
        section = title
    return section


def _tiktoken_counter() -> Callable[[str], int]:
    import tiktoken

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return lambda text: len(encoding.encode(text))
