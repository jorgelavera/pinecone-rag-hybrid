"""Initialize the Pinecone index using the legacy SDK API.

Runs with no install step: the three-line bootstrap adds ``src`` to
``sys.path`` so ``python scripts/init_index.py`` works from the repo root.
Missing configuration prints a remediation message and exits with code 2 with
no traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from rag.config import load_settings, require_credentials
from rag.errors import MissingConfigError
from rag import pinecone_client


def main(client=None, settings=None, env=None):
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

    if client is None:
        client = pinecone_client.build_client(settings)

    result = pinecone_client.verify_or_create_index(client, settings)
    index = pinecone_client.build_index(client, settings)

    stats = index.describe_index_stats()
    namespaces = getattr(stats, "namespaces", {}) or {}
    if result.get("created"):
        print(f"Created index {settings.index_name} (dimension={settings.embedding_dimensions}, metric=cosine)")
    else:
        print(f"Index {settings.index_name} already exists (dimension={settings.embedding_dimensions}, metric=cosine)")
    print("Namespaces:")
    if not namespaces:
        print(f"  {settings.namespace}: 0 vectors (empty)")
    else:
        for namespace_name, namespace_info in namespaces.items():
            count = getattr(namespace_info, "vector_count", None)
            if count is None and isinstance(namespace_info, dict):
                count = namespace_info.get("vector_count")
            print(f"  {namespace_name}: {count} vectors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
