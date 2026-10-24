"""Embed documents with the OpenAI embeddings model (lazy imports).

``langchain_openai`` is only imported inside ``build_embeddings``; no top-level
OpenAI or LangChain imports keep this module credential-free to import.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag.config import Settings, require_credentials
from rag.errors import DimensionMismatchError


def build_embeddings(settings: Settings):
    """Build the embeddings client with the pinned model and dimensions."""
    require_credentials(settings)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.openai_api_key,
    )


def embed_texts(embeddings, texts: Sequence[str], *, expected_dim: int = 1536) -> list[list[float]]:
    """Embed texts and assert every vector has the expected dimension.

    The assertion runs before any upsert: a dimension mismatch must block
    ingestion rather than writing malformed vectors to Pinecone.
    """
    texts = list(texts)
    vectors = embeddings.embed_documents(texts)
    for index, vector in enumerate(vectors):
        length = len(vector)
        if length != expected_dim:
            raise DimensionMismatchError(
                f"Embedding at index {index} has dimension {length}, expected {expected_dim}"
            )
    return vectors
