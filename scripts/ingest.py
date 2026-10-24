"""Ingest the Markdown corpus into Pinecone with batching, dim guards and idempotency.

Runs with no install step. Missing credentials cause exit code 2 with a
remediation message and no traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rag import chunking, embeddings, pinecone_client
from rag.config import load_settings, require_credentials
from rag.errors import MissingConfigError


def main(client=None, settings=None, env=None, corpus_dir=None, embedding_model=None):
    if settings is None:
        try:
            settings = load_settings(env=env)
            require_credentials(settings)
        except MissingConfigError as exc:
            print(f"Missing configuration: {exc}")
            print("Copy .env.example to .env and set the listed values.")
            print("Credential-free checks: uv run pytest -q")
            return 2
    else:
        require_credentials(settings)

    documents = chunking.load_corpus(corpus_dir or settings.corpus_dir)
    chunks = chunking.split_documents(documents)

    if client is None:
        client = pinecone_client.build_client(settings)
    index = pinecone_client.build_index(client, settings)
    if embedding_model is None:
        embedding_model = embeddings.build_embeddings(settings)

    vectors = embeddings.embed_texts(embedding_model, [chunk.text for chunk in chunks], expected_dim=settings.embedding_dimensions)
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    metadatas = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.pop("text", None)
        metadatas.append(metadata)
    ids = [chunk.id for chunk in chunks]

    store = pinecone_client.build_vector_store(index, embedding_model, settings)
    store.add_texts([chunk.text for chunk in chunks], metadatas=metadatas, ids=ids, batch_size=settings.batch_size, namespace=settings.namespace)

    print(f"Ingested {len(chunks)} chunks from {len(documents)} documents into namespace {settings.namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
