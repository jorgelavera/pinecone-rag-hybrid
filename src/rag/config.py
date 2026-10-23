"""Environment-driven settings with actionable validation.

Only the standard library is imported at module scope. ``python-dotenv`` is
imported lazily inside :func:`load_settings`, and only when the caller did not
inject an explicit environment mapping — that keeps the pure tier testable with
zero credentials, zero network, and not even the dotenv package installed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from rag.errors import ConfigError, MissingConfigError

DEFAULT_NAMESPACE = "rag-docs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EXPECTED_EMBEDDING_DIMENSIONS = 1536
WEIGHT_TOLERANCE = 1e-6
WEIGHT_TOLERANCE_TEXT = "1e-6"

# (environment variable, Settings attribute) pairs required by every live script.
CREDENTIAL_FIELDS = (
    ("PINECONE_API_KEY", "pinecone_api_key"),
    ("OPENAI_API_KEY", "openai_api_key"),
    ("INDEX_NAME", "index_name"),
)


@dataclass(frozen=True)
class Settings:
    """Single source of truth for every tunable value in the pipeline."""

    pinecone_api_key: str
    openai_api_key: str
    index_name: str
    namespace: str = DEFAULT_NAMESPACE
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = EXPECTED_EMBEDDING_DIMENSIONS
    chunk_size: int = 600
    chunk_overlap: int = 80
    batch_size: int = 32
    cloud: str = "aws"
    region: str = "us-east-1"
    pipeline_version: str = "1"
    corpus_dir: Path = Path("data/corpus")
    golden_set_path: Path = Path("data/golden_set.json")
    artifacts_dir: Path = Path("artifacts")


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build and validate :class:`Settings`.

    ``env=None`` loads ``.env`` with ``override=False`` (real environment wins)
    and then reads ``os.environ``. Passing a mapping is hermetic: no dotenv call,
    no process environment — which is how the tests stay credential-free.
    """
    if env is None:
        _load_dotenv_into_environment()
        env = os.environ

    settings = Settings(
        pinecone_api_key=_text(env, "PINECONE_API_KEY"),
        openai_api_key=_text(env, "OPENAI_API_KEY"),
        index_name=_text(env, "INDEX_NAME"),
        namespace=_text(env, "PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        bm25_weight=_as_float(env, "ENSEMBLE_BM25_WEIGHT", 0.4),
        vector_weight=_as_float(env, "ENSEMBLE_VECTOR_WEIGHT", 0.6),
        embedding_model=_text(env, "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_dimensions=_as_int(env, "EMBEDDING_DIMENSIONS", EXPECTED_EMBEDDING_DIMENSIONS),
        chunk_size=_as_int(env, "CHUNK_SIZE", 600),
        chunk_overlap=_as_int(env, "CHUNK_OVERLAP", 80),
        batch_size=_as_int(env, "UPSERT_BATCH_SIZE", 32),
        cloud=_text(env, "PINECONE_CLOUD", "aws"),
        region=_text(env, "PINECONE_REGION", "us-east-1"),
        pipeline_version=_text(env, "PIPELINE_VERSION", "1"),
        corpus_dir=Path(_text(env, "CORPUS_DIR", "data/corpus")),
        golden_set_path=Path(_text(env, "GOLDEN_SET_PATH", "data/golden_set.json")),
        artifacts_dir=Path(_text(env, "ARTIFACTS_DIR", "artifacts")),
    )
    return _validate(settings)


def require_credentials(settings: Settings) -> None:
    """Raise once, listing every missing variable, so one run fixes them all."""
    missing = [
        env_name for env_name, attribute in CREDENTIAL_FIELDS if not getattr(settings, attribute).strip()
    ]
    if missing:
        raise MissingConfigError(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and set these values."
        )


def _load_dotenv_into_environment() -> None:
    """Load `.env` without overriding the process environment; tolerate its absence."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _text(env: Mapping[str, str], name: str, default: str = "") -> str:
    value = env.get(name)
    return default if value is None else value


def _as_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid number") from exc


def _as_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid integer") from exc


def _validate(settings: Settings) -> Settings:
    _validate_weights(settings)
    _validate_namespace(settings)
    _validate_chunking(settings)

    if not 1 <= settings.batch_size <= 100:
        raise ConfigError(
            f"UPSERT_BATCH_SIZE={settings.batch_size}: must be between 1 and 100 "
            "(one bounded upsert request per batch)"
        )
    if settings.embedding_dimensions != EXPECTED_EMBEDDING_DIMENSIONS:
        raise ConfigError(
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions}: must be "
            f"{EXPECTED_EMBEDDING_DIMENSIONS} to match {DEFAULT_EMBEDDING_MODEL} and the index dimension"
        )
    return settings


def _validate_weights(settings: Settings) -> None:
    bm25, vector = settings.bm25_weight, settings.vector_weight
    total = bm25 + vector
    sums_correctly = abs(total - 1.0) <= WEIGHT_TOLERANCE
    if math.isfinite(total) and bm25 > 0 and vector > 0 and sums_correctly:
        return
    raise ConfigError(
        f"ENSEMBLE_BM25_WEIGHT={bm25} and ENSEMBLE_VECTOR_WEIGHT={vector}: "
        f"weights must be positive and sum to 1.0 (±{WEIGHT_TOLERANCE_TEXT}), got {round(total, 10)}"
    )


def _validate_namespace(settings: Settings) -> None:
    if not settings.namespace.strip():
        raise ConfigError(
            "PINECONE_NAMESPACE must be a non-empty string "
            "(empty means the default namespace and breaks every read/write)"
        )


def _validate_chunking(settings: Settings) -> None:
    if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ConfigError(
            f"CHUNK_SIZE={settings.chunk_size} and CHUNK_OVERLAP={settings.chunk_overlap}: "
            "chunk_size must be > 0 and chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size"
        )
