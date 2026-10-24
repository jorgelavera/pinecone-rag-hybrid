"""evaluate.py behavior: live client wiring, k=5 retrieval, error surfacing,
and the credential-free ``--calibration`` regenerator with honest provenance.

No test here imports ``openai``/``pinecone`` or touches the network: the live
path is exercised through injected fakes (``clients=``/``system_factory=``),
exactly like ``scripts/ingest.py`` tests inject ``client=``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.errors import RetrievalEmptyError

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("rag_script_evaluate", ROOT / "evaluate.py")
EVALUATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EVALUATE)

FROZEN = {"precision": 0.36, "recall": 0.833333, "ceiling": 0.44, "normalized_precision": 0.818182}


def live_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PINECONE_API_KEY": "test-pinecone-key",
        "OPENAI_API_KEY": "test-openai-key",
        "INDEX_NAME": "rag-pytest",
        "ARTIFACTS_DIR": str(tmp_path),
    }


class RecordingSystem:
    """RAGSystem stand-in recording construction kwargs and retrieve calls."""

    instances: list["RecordingSystem"] = []

    def __init__(self, *, settings, index=None, embeddings=None, vector_store=None, corpus=None, mode="hybrid"):
        self.mode = mode
        self.index = index
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.calls: list[tuple[str, int]] = []
        RecordingSystem.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances.clear()

    def retrieve(self, query: str, k: int = 5):
        self.calls.append((query, k))
        return [SimpleNamespace(document_id="pytest-fixtures")]


class FailingSystem:
    def __init__(self, **kwargs):
        pass

    def retrieve(self, query: str, k: int = 5):
        raise RetrievalEmptyError("index=rag-pytest namespace=rag-docs: 0 documents returned")


@pytest.fixture(autouse=True)
def _chdir_and_reset(monkeypatch):
    RecordingSystem.reset()
    monkeypatch.chdir(ROOT)


# --------------------------------------------------------------------------- #
# Live path
# --------------------------------------------------------------------------- #


def test_live_main_builds_rag_systems_with_the_real_clients_and_retrieves_at_k5(tmp_path):
    clients = EVALUATE.LiveClients(
        index=SimpleNamespace(name="fake-index"),
        embeddings=SimpleNamespace(name="fake-embeddings"),
        vector_store=SimpleNamespace(name="fake-vector-store"),
    )

    exit_code = EVALUATE.main(argv=[], env=live_env(tmp_path), clients=clients, system_factory=RecordingSystem)

    assert exit_code == 0
    assert [system.mode for system in RecordingSystem.instances] == ["vector", "bm25", "hybrid"]
    for system in RecordingSystem.instances:
        assert system.index is clients.index
        assert system.embeddings is clients.embeddings
        assert system.vector_store is clients.vector_store
        assert len(system.calls) == 12  # the committed Golden Set
        assert {k for _, k in system.calls} == {5}  # evaluation k, never batch_size

    payload = json.loads((tmp_path / "last_eval.json").read_text(encoding="utf-8"))
    assert payload["n_questions"] == 12
    assert payload["k"] == 5
    per_question = payload["per_question"]
    assert len(per_question) == 12
    assert per_question[0]["retrieved_document_ids"] == ["pytest-fixtures"]
    # per-question rows aggregate exactly to the reported primary mode
    assert round(sum(row["precision"] for row in per_question) / len(per_question), 6) == payload["primary"]["precision"]
    assert round(sum(row["recall"] for row in per_question) / len(per_question), 6) == payload["primary"]["recall"]


def test_live_main_surfaces_retrieval_errors_instead_of_swallowing_them(tmp_path):
    clients = EVALUATE.LiveClients(index=object(), embeddings=object(), vector_store=object())

    with pytest.raises(RetrievalEmptyError):
        EVALUATE.main(argv=[], env=live_env(tmp_path), clients=clients, system_factory=FailingSystem)

    assert not (tmp_path / "last_eval.json").exists()


# --------------------------------------------------------------------------- #
# --calibration: deterministic, credential-free regenerator
# --------------------------------------------------------------------------- #


def test_calibration_regenerates_the_committed_artifact_without_credentials(tmp_path, capsys):
    exit_code = EVALUATE.main(argv=["--calibration"], env={"ARTIFACTS_DIR": str(tmp_path)})

    assert exit_code == 0
    regenerated = json.loads((tmp_path / "last_eval.json").read_text(encoding="utf-8"))
    committed = json.loads((ROOT / "artifacts" / "last_eval.json").read_text(encoding="utf-8"))
    assert regenerated == committed

    assert {key: regenerated["primary"][key] for key in FROZEN} == FROZEN
    assert regenerated["n_questions"] == 5
    assert regenerated["golden_set_path"] == "data/calibration_set.json"
    for mode in ("vector", "bm25", "hybrid"):
        assert {key: regenerated["ablations"][mode][key] for key in FROZEN} == FROZEN
    for row in regenerated["per_question"]:
        assert all("#" not in doc_id for doc_id in row["retrieved_document_ids"])

    provenance = capsys.readouterr().out
    assert "calibration" in provenance.lower()


def test_offline_reprint_states_calibration_provenance(capsys):
    exit_code = EVALUATE.main(argv=["--offline"], env={})

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "calibration" in output.lower()
    assert "0.36" in output
    assert "0.833333" in output
