"""Pinecone wiring isolated to this module (legacy API).

``pinecone`` is imported lazily inside every function that talks to the SDK, so
``import rag.pinecone_client`` carries no credentials, no network calls, and
does not import ``openai`` or ``langchain_openai``. ``Pinecone.indexes`` does
not exist in ``pinecone==7.3.0`` (the test fakes assert this).
"""

from __future__ import annotations

from rag.config import Settings, require_credentials
from rag.errors import IndexConfigMismatchError


def build_client(settings: Settings):
    """Create the legacy Pinecone client. Credentials must be present."""
    require_credentials(settings)
    from pinecone import Pinecone

    return Pinecone(api_key=settings.pinecone_api_key)


def verify_or_create_index(client, settings: Settings) -> dict:
    """Verify the index exists and matches the expected config, else create it.

    The index is never deleted or recreated on mismatch: that would risk data
    loss, and the test for a wrong dimension/metric explicitly forbids both
    operations.
    """
    name = settings.index_name
    if client.has_index(name):
        describe_index = getattr(client, "describe_index", None)
        if callable(describe_index):
            # No try/except around the describe call: a network/API failure must
            # surface as itself, never silently bypass the config guard below.
            record = describe_index(name)
            dimension = getattr(record, "dimension", None)
            metric = getattr(record, "metric", None)
            if dimension is not None and dimension != settings.embedding_dimensions:
                raise IndexConfigMismatchError(
                    f"Index {name!r} has dimension {dimension}, "
                    f"expected {settings.embedding_dimensions}; "
                    "refusing to delete or recreate the index"
                )
            if metric is not None and metric != "cosine":
                raise IndexConfigMismatchError(
                    f"Index {name!r} has metric {metric!r}, expected 'cosine'; "
                    "refusing to delete or recreate the index"
                )
        return {"name": name, "created": False, "dimension": settings.embedding_dimensions, "metric": "cosine"}

    from pinecone import ServerlessSpec

    client.create_index(
        name=name,
        dimension=settings.embedding_dimensions,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
    )
    return {"name": name, "created": True, "dimension": settings.embedding_dimensions, "metric": "cosine"}


def build_index(client, settings: Settings):
    """Return a handle to the index via the legacy ``client.Index(name)`` API."""
    return client.Index(settings.index_name)


def build_vector_store(index, embeddings, settings: Settings):
    """Wire PineconeVectorStore with the project contract."""
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(index=index, embedding=embeddings, text_key="text", namespace=settings.namespace)
