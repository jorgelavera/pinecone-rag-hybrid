"""Settings loading and validation, all credential-free and hermetic."""

import sys
import types
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rag.config import Settings, load_settings, require_credentials
from rag.errors import ConfigError, MissingConfigError

CREDENTIAL_VARS = ("PINECONE_API_KEY", "OPENAI_API_KEY", "INDEX_NAME")


@pytest.fixture
def fake_dotenv(monkeypatch):
    """Install a stub `dotenv` that mirrors real load_dotenv(override=...) semantics."""
    calls: list[dict[str, object]] = []
    module = types.ModuleType("dotenv")

    def load_dotenv(path=None, override=False, **kwargs):
        calls.append({"path": path, "override": override})
        if override:  # a real `override=True` clobbers the process environment
            monkeypatch.setenv("INDEX_NAME", "from-file")
        return True

    module.load_dotenv = load_dotenv
    monkeypatch.setitem(sys.modules, "dotenv", module)
    return calls


@pytest.fixture(autouse=True)
def _clear_project_env(monkeypatch):
    """Keep the caller's shell from leaking into assertions."""
    for name in (
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
        "INDEX_NAME",
        "PINECONE_NAMESPACE",
        "ENSEMBLE_BM25_WEIGHT",
        "ENSEMBLE_VECTOR_WEIGHT",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "UPSERT_BATCH_SIZE",
        "EMBEDDING_DIMENSIONS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_the_documented_values():
    settings = load_settings(env={})

    assert settings.namespace == "rag-docs"
    assert (settings.bm25_weight, settings.vector_weight) == (0.4, 0.6)
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert (settings.chunk_size, settings.chunk_overlap) == (600, 80)
    assert settings.batch_size == 32
    assert (settings.cloud, settings.region) == ("aws", "us-east-1")
    assert settings.pipeline_version == "1"
    assert settings.corpus_dir == Path("data/corpus")
    assert settings.golden_set_path == Path("data/golden_set.json")
    assert settings.artifacts_dir == Path("artifacts")
    assert (settings.pinecone_api_key, settings.openai_api_key, settings.index_name) == ("", "", "")


def test_environment_mapping_overrides_defaults():
    settings = load_settings(
        env={
            "PINECONE_API_KEY": "pc-key",
            "OPENAI_API_KEY": "oa-key",
            "INDEX_NAME": "rag-pytest",
            "PINECONE_NAMESPACE": "custom-ns",
            "ENSEMBLE_BM25_WEIGHT": "0.3",
            "ENSEMBLE_VECTOR_WEIGHT": "0.7",
        }
    )

    assert (settings.pinecone_api_key, settings.openai_api_key, settings.index_name) == (
        "pc-key",
        "oa-key",
        "rag-pytest",
    )
    assert settings.namespace == "custom-ns"
    assert (settings.bm25_weight, settings.vector_weight) == (0.3, 0.7)


def test_settings_are_frozen():
    settings = load_settings(env={"INDEX_NAME": "rag-pytest"})

    with pytest.raises(FrozenInstanceError):
        settings.index_name = "other"


def test_injected_environment_never_touches_dotenv(fake_dotenv):
    load_settings(env={"INDEX_NAME": "rag-pytest"})

    assert fake_dotenv == []


def test_shell_environment_wins_over_the_dotenv_file(fake_dotenv, monkeypatch):
    monkeypatch.setenv("INDEX_NAME", "from-shell")

    settings = load_settings()

    assert settings.index_name == "from-shell"
    assert fake_dotenv and fake_dotenv[0]["override"] is False


def test_missing_dotenv_falls_back_to_the_process_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", None)  # makes `import dotenv` raise ImportError
    monkeypatch.setenv("INDEX_NAME", "from-process")

    assert load_settings().index_name == "from-process"


def test_weights_must_be_positive_and_sum_to_one():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env={"ENSEMBLE_BM25_WEIGHT": "4.0", "ENSEMBLE_VECTOR_WEIGHT": "6.0"})

    message = str(excinfo.value)
    assert "ENSEMBLE_BM25_WEIGHT=4.0" in message
    assert "ENSEMBLE_VECTOR_WEIGHT=6.0" in message
    assert "positive" in message
    assert "sum to 1.0" in message
    assert "1e-6" in message
    assert "10.0" in message


def test_negative_weight_is_rejected_even_when_the_sum_is_one():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env={"ENSEMBLE_BM25_WEIGHT": "-0.5", "ENSEMBLE_VECTOR_WEIGHT": "1.5"})

    assert "positive" in str(excinfo.value)


def test_empty_namespace_is_rejected():
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env={"PINECONE_NAMESPACE": ""})

    message = str(excinfo.value)
    assert "PINECONE_NAMESPACE" in message
    assert "non-empty" in message


@pytest.mark.parametrize(
    ("env", "expected_fragment"),
    [
        ({"CHUNK_SIZE": "80", "CHUNK_OVERLAP": "80"}, "chunk_overlap"),
        ({"CHUNK_SIZE": "600", "CHUNK_OVERLAP": "-1"}, "chunk_overlap"),
        ({"UPSERT_BATCH_SIZE": "0"}, "UPSERT_BATCH_SIZE"),
        ({"UPSERT_BATCH_SIZE": "101"}, "UPSERT_BATCH_SIZE"),
        ({"EMBEDDING_DIMENSIONS": "1024"}, "1536"),
        ({"CHUNK_SIZE": "not-a-number"}, "CHUNK_SIZE"),
    ],
)
def test_invalid_numeric_settings_raise_config_error(env, expected_fragment):
    with pytest.raises(ConfigError) as excinfo:
        load_settings(env=env)

    assert expected_fragment in str(excinfo.value)


def test_require_credentials_lists_every_missing_name():
    with pytest.raises(MissingConfigError) as excinfo:
        require_credentials(load_settings(env={}))

    message = str(excinfo.value)
    for name in CREDENTIAL_VARS:
        assert name in message
    assert ".env.example" in message


def test_require_credentials_reports_only_the_missing_ones():
    with pytest.raises(MissingConfigError) as excinfo:
        require_credentials(load_settings(env={"INDEX_NAME": "rag-pytest"}))

    message = str(excinfo.value)
    assert "PINECONE_API_KEY" in message
    assert "OPENAI_API_KEY" in message
    assert "INDEX_NAME" not in message


def test_require_credentials_passes_when_all_keys_are_present():
    settings = load_settings(
        env={"PINECONE_API_KEY": "pc-key", "OPENAI_API_KEY": "oa-key", "INDEX_NAME": "rag-pytest"}
    )

    require_credentials(settings)


def test_settings_type_is_used_by_callers():
    assert isinstance(load_settings(env={}), Settings)
