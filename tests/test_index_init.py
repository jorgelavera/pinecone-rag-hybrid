"""Index bootstrap: legacy SDK surface, ServerlessSpec, mismatch guard, no recreate.

Tier B: the Pinecone client is faked, so nothing here imports `pinecone` or
touches the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag import pinecone_client
from rag.config import load_settings
from rag.errors import IndexConfigMismatchError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "init_index.py"


class StatsIndex:
    """Only the surface `init_index` needs: index stats."""

    def __init__(self, namespace_counts: dict[str, int] | None = None):
        self.namespace_counts = dict(namespace_counts or {})
        self.stats_calls = 0

    def describe_index_stats(self, *args, **kwargs):
        self.stats_calls += 1
        return SimpleNamespace(
            namespaces={
                namespace: SimpleNamespace(vector_count=count)
                for namespace, count in self.namespace_counts.items()
            },
            total_vector_count=sum(self.namespace_counts.values()),
        )


@pytest.fixture(scope="module")
def init_index():
    spec = importlib.util.spec_from_file_location("rag_script_init_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(**overrides):
    environment = {
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "INDEX_NAME": "rag-pytest",
        **overrides,
    }
    return load_settings(env=environment)


def existing_index(**overrides):
    values = {"name": "rag-pytest", "dimension": 1536, "metric": "cosine", **overrides}
    return SimpleNamespace(**values)


# --------------------------------------------------------------------------- #
# Verify then create
# --------------------------------------------------------------------------- #


def test_absent_index_is_created_with_the_legacy_call_and_serverless_spec(fake_pinecone_client):
    result = pinecone_client.verify_or_create_index(fake_pinecone_client, settings())

    assert len(fake_pinecone_client.create_calls) == 1
    call = fake_pinecone_client.create_calls[0]
    assert call["name"] == "rag-pytest"
    assert call["dimension"] == 1536
    assert call["metric"] == "cosine"
    spec = call["spec"]
    assert type(spec).__name__ == "ServerlessSpec"
    assert spec.cloud == "aws"
    assert spec.region == "us-east-1"
    assert result["created"] is True


def test_create_uses_the_configured_cloud_and_region(fake_pinecone_client):
    configured = settings(PINECONE_CLOUD="gcp", PINECONE_REGION="europe-west4")

    pinecone_client.verify_or_create_index(fake_pinecone_client, configured)

    spec = fake_pinecone_client.create_calls[0]["spec"]
    assert (spec.cloud, spec.region) == ("gcp", "europe-west4")


def test_existing_index_is_a_no_op(fake_pinecone_client):
    fake_pinecone_client.existing["rag-pytest"] = existing_index()

    result = pinecone_client.verify_or_create_index(fake_pinecone_client, settings())

    assert fake_pinecone_client.create_calls == []
    assert fake_pinecone_client.describe_calls == ["rag-pytest"]
    assert result["created"] is False
    assert result["dimension"] == 1536


def test_wrong_dimension_aborts_without_deleting_or_recreating(fake_pinecone_client):
    fake_pinecone_client.existing["rag-pytest"] = existing_index(dimension=768)

    with pytest.raises(IndexConfigMismatchError) as excinfo:
        pinecone_client.verify_or_create_index(fake_pinecone_client, settings())

    message = str(excinfo.value)
    assert "1536" in message
    assert "768" in message
    assert fake_pinecone_client.create_calls == []
    assert fake_pinecone_client.delete_calls == []


def test_wrong_metric_aborts_without_deleting_or_recreating(fake_pinecone_client):
    fake_pinecone_client.existing["rag-pytest"] = existing_index(metric="euclidean")

    with pytest.raises(IndexConfigMismatchError) as excinfo:
        pinecone_client.verify_or_create_index(fake_pinecone_client, settings())

    message = str(excinfo.value)
    assert "cosine" in message
    assert "euclidean" in message
    assert fake_pinecone_client.create_calls == []
    assert fake_pinecone_client.delete_calls == []


def test_schema_api_attribute_is_absent_in_this_sdk(fake_pinecone_client):
    with pytest.raises(AssertionError, match="no Pinecone.indexes"):
        fake_pinecone_client.indexes


class DescribeFailsClient:
    """has_index says yes, but describe_index dies with an API/network error."""

    def __init__(self):
        self.create_calls: list[dict[str, object]] = []
        self.delete_calls: list[str] = []

    def has_index(self, name: str) -> bool:
        return True

    def describe_index(self, name: str):
        raise RuntimeError("pinecone API unavailable")

    def create_index(self, **kwargs):
        self.create_calls.append(kwargs)

    def delete_index(self, name: str):
        self.delete_calls.append(name)


def test_describe_index_api_error_surfaces_instead_of_bypassing_the_guard():
    # A swallowed describe error would skip the dimension/metric guard and
    # report a healthy no-op: network/API failures must surface as themselves.
    client = DescribeFailsClient()

    with pytest.raises(RuntimeError, match="pinecone API unavailable"):
        pinecone_client.verify_or_create_index(client, settings())

    assert client.create_calls == []
    assert client.delete_calls == []


# --------------------------------------------------------------------------- #
# Handles and vector store
# --------------------------------------------------------------------------- #


def test_build_index_uses_the_legacy_index_lookup(fake_pinecone_client):
    stats_index = StatsIndex()
    fake_pinecone_client.index_factory = lambda: stats_index

    index = pinecone_client.build_index(fake_pinecone_client, settings())

    assert fake_pinecone_client.index_calls == ["rag-pytest"]
    assert index is stats_index


def test_build_vector_store_pins_text_key_and_namespace(fake_embeddings):
    index = SimpleNamespace(config=SimpleNamespace(host="https://rag-pytest.fake", api_key="key"))

    store = pinecone_client.build_vector_store(index, fake_embeddings, settings())

    assert store._text_key == "text"
    assert store._namespace == "rag-docs"
    assert store._index is index


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


def test_script_prints_the_namespace_with_its_vector_count(init_index, fake_pinecone_client, capsys):
    stats_index = StatsIndex({"rag-docs": 24})
    fake_pinecone_client.index_factory = lambda: stats_index

    exit_code = init_index.main(client=fake_pinecone_client, settings=settings())

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "rag-docs" in output
    assert "24" in output
    assert stats_index.stats_calls == 1


def test_script_second_run_creates_at_most_one_index(init_index, fake_pinecone_client, capsys):
    fake_pinecone_client.index_factory = lambda: StatsIndex({"rag-docs": 24})

    first = init_index.main(client=fake_pinecone_client, settings=settings())
    second = init_index.main(client=fake_pinecone_client, settings=settings())

    assert (first, second) == (0, 0)
    assert len(fake_pinecone_client.create_calls) == 1
    assert "already exists" in capsys.readouterr().out


def test_script_aborts_on_a_configuration_mismatch(init_index, fake_pinecone_client):
    fake_pinecone_client.existing["rag-pytest"] = existing_index(dimension=768)

    with pytest.raises(IndexConfigMismatchError, match="768"):
        init_index.main(client=fake_pinecone_client, settings=settings())

    assert fake_pinecone_client.create_calls == []
    assert fake_pinecone_client.delete_calls == []
