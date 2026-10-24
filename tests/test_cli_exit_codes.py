"""CLI exit code 2 contract on missing configuration.

All three entry points must print a remediation message that names the missing
variable, points to ``.env.example``, and suggests ``uv run pytest -q`` — with
no traceback leaked to the user. ``evaluate.py --offline`` is exempt from the
credential requirement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"rag_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INIT = load_script("init_index", ROOT / "scripts" / "init_index.py")
INGEST = load_script("ingest", ROOT / "scripts" / "ingest.py")
EVALUATE = load_script("evaluate", ROOT / "evaluate.py")


def scrubbed_env(monkeypatch):
    for key in ("PINECONE_API_KEY", "OPENAI_API_KEY", "INDEX_NAME"):
        monkeypatch.delenv(key, raising=False)


def assert_remediation(message: str) -> None:
    assert "Missing configuration" in message or "missing" in message.lower()
    assert "PINECONE_API_KEY" in message or "OPENAI_API_KEY" in message or "INDEX_NAME" in message
    assert ".env.example" in message
    assert "uv run pytest -q" in message


def test_init_index_exits_2_with_remediation_when_keys_missing(monkeypatch, capsys):
    scrubbed_env(monkeypatch)
    monkeypatch.chdir(ROOT)

    exit_code = INIT.main(env={})

    output = capsys.readouterr().out
    assert exit_code == 2
    assert_remediation(output)
    assert "Traceback" not in output


def test_ingest_exits_2_with_remediation_when_keys_missing(monkeypatch, capsys):
    scrubbed_env(monkeypatch)
    monkeypatch.chdir(ROOT)

    exit_code = INGEST.main(env={})

    output = capsys.readouterr().out
    assert exit_code == 2
    assert_remediation(output)
    assert "Traceback" not in output


def test_evaluate_exits_2_with_remediation_when_keys_missing_and_not_offline(monkeypatch, capsys):
    scrubbed_env(monkeypatch)
    monkeypatch.chdir(ROOT)

    exit_code = EVALUATE.main(argv=[], env={})

    output = capsys.readouterr().out
    assert exit_code == 2
    assert_remediation(output)
    assert "Traceback" not in output


def test_evaluate_offline_does_not_require_credentials(monkeypatch, tmp_path, capsys):
    scrubbed_env(monkeypatch)
    monkeypatch.chdir(ROOT)
    # The committed artifact must be readable from cwd/artifacts/last_eval.json.
    output = capsys.readouterr().out  # clear

    exit_code = EVALUATE.main(argv=["--offline"], env={})

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Precision@5" in output or "ceiling" in output.lower() or "hybrid" in output
    assert "Traceback" not in output
