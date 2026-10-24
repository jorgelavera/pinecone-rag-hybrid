"""Offline evaluation CLI with ablations, reporting, and exit code 2 on missing config.

``python evaluate.py`` (live) builds the real Pinecone/OpenAI clients through
``rag.pinecone_client``/``rag.embeddings`` factories, constructs one
``RAGSystem`` per mode (``vector``, ``bm25``, ``hybrid``), scores the Golden Set
at ``k=5``, writes ``artifacts/last_eval.json`` in the shape pinned by
``tests/test_evaluation.py::test_build_report_payload_matches_the_committed_artifact_shape``,
and prints a table. Retrieval errors (e.g. ``RetrievalEmptyError``) surface:
the spec has no continue-on-empty behavior and the error taxonomy reserves exit
code 2 for configuration problems only.

``python evaluate.py --calibration`` regenerates the committed artifact from
the frozen worked example in ``data/calibration_set.json`` — deterministically
and with no credentials — through the same ``score_retrieval`` +
``build_report_payload`` code a live run uses.

``python evaluate.py --offline`` reprints the committed artifact with no keys
or clients constructed, stating its provenance (calibration demonstration vs
live run).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import evaluation
from rag.config import load_settings, require_credentials
from rag.errors import MissingConfigError

MODES = ("vector", "bm25", "hybrid")
EVALUATION_K = evaluation.DEFAULT_K
CALIBRATION_SET_PATH = Path("data/calibration_set.json")
CALIBRATION_INDEX_NAME = "rag-pytest"
# Frozen so the committed artifact is byte-for-byte reproducible.
CALIBRATION_GENERATED_AT = "2026-10-22T23:00:00Z"


class LiveClients:
    """The three live objects every RAGSystem mode shares (test-injectable).

    A plain class, not a dataclass: the CLI tests load this module through
    ``importlib.util`` without registering it in ``sys.modules``, which the
    dataclass machinery requires.
    """

    def __init__(self, *, index, embeddings, vector_store):
        self.index = index
        self.embeddings = embeddings
        self.vector_store = vector_store


def main(argv=None, env=None, *, clients=None, system_factory=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--offline" in argv:
        return _offline_main()
    if "--calibration" in argv:
        return _calibration_main(env=env)

    try:
        settings = load_settings(env=env)
        require_credentials(settings)
    except MissingConfigError as exc:
        print(f"Missing configuration: {exc}")
        print("Copy .env.example to .env and set the listed values.")
        print("Credential-free checks: uv run pytest -q")
        return 2

    return _live_main(settings, clients=clients, system_factory=system_factory)


def _build_live_clients(settings) -> LiveClients:
    """Wire the real Pinecone/OpenAI objects via the existing factories."""
    from rag import embeddings as embeddings_module
    from rag import pinecone_client

    client = pinecone_client.build_client(settings)
    index = pinecone_client.build_index(client, settings)
    embedding_model = embeddings_module.build_embeddings(settings)
    vector_store = pinecone_client.build_vector_store(index, embedding_model, settings)
    return LiveClients(index=index, embeddings=embedding_model, vector_store=vector_store)


def _live_main(settings, *, clients=None, system_factory=None) -> int:
    from rag.chunking import load_corpus

    corpus_ids = {document.metadata["document_id"] for document in load_corpus(Path(settings.corpus_dir))}
    golden = evaluation.load_golden_set(
        Path(settings.golden_set_path),
        corpus_ids=corpus_ids,
        min_questions=10,
    )

    if clients is None:
        clients = _build_live_clients(settings)
    if system_factory is None:
        from rag.retrieval import RAGSystem

        system_factory = RAGSystem

    ablation_reports = {}
    for mode in MODES:
        system = system_factory(
            settings=settings,
            index=clients.index,
            embeddings=clients.embeddings,
            vector_store=clients.vector_store,
            mode=mode,
        )
        retrieved_lists = [
            [document.document_id for document in system.retrieve(question.question, k=EVALUATION_K)]
            for question in golden
        ]
        ablation_reports[mode] = evaluation.score_retrieval(golden, retrieved_lists, k=EVALUATION_K)

    payload = evaluation.build_report_payload(
        settings=settings,
        golden=golden,
        ablation=ablation_reports,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    payload_path = _write_payload(settings, payload)

    _print_report(
        k=payload["k"],
        n_questions=payload["n_questions"],
        primary=payload["primary"],
        ablations=payload["ablations"],
        source=f"{payload_path} ({_provenance(payload)})",
    )
    return 0


def _calibration_main(env=None) -> int:
    """Regenerate ``artifacts/last_eval.json`` from the frozen calibration set.

    Credential-free and deterministic: the same retrieved id lists are scored
    for every mode, so the artifact is internally consistent by construction.
    """
    from rag.chunking import load_corpus

    overrides = dict(env) if env is not None else {}
    settings = load_settings(
        env={
            "INDEX_NAME": CALIBRATION_INDEX_NAME,
            "GOLDEN_SET_PATH": CALIBRATION_SET_PATH.as_posix(),
            **overrides,
        }
    )

    calibration_path = Path(settings.golden_set_path)
    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path
    corpus_ids = {document.metadata["document_id"] for document in load_corpus(Path(settings.corpus_dir))}
    questions, retrieved = evaluation.load_calibration_set(calibration_path, corpus_ids=corpus_ids)

    ablation_reports = {
        mode: evaluation.score_retrieval(questions, retrieved, k=EVALUATION_K) for mode in MODES
    }
    payload = evaluation.build_report_payload(
        settings=settings,
        golden=questions,
        ablation=ablation_reports,
        generated_at=CALIBRATION_GENERATED_AT,
    )
    payload_path = _write_payload(settings, payload)

    _print_report(
        k=payload["k"],
        n_questions=payload["n_questions"],
        primary=payload["primary"],
        ablations=payload["ablations"],
        source=f"{payload_path} ({_provenance(payload)})",
    )
    return 0


def _offline_main() -> int:
    payload_path = ROOT / "artifacts" / "last_eval.json"
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"Missing offline artifact: {payload_path}")
        print("Regenerate it with `python evaluate.py --calibration` (no credentials)")
        print("or `python evaluate.py` (live run).")
        raise SystemExit(1) from exc

    _print_report(
        k=payload.get("k", 5),
        n_questions=payload.get("n_questions"),
        primary=payload.get("primary", {}),
        ablations=payload.get("ablations", {}),
        source=f"artifacts/last_eval.json ({_provenance(payload)})",
    )
    return 0


def _provenance(payload) -> str:
    """Honest provenance derived from the payload — never a stored extra key."""
    if str(payload.get("golden_set_path", "")).endswith("calibration_set.json"):
        return (
            "calibration demonstration: frozen worked example from "
            "specs/evaluation-metrics, not a live Pinecone/OpenAI run"
        )
    return "live run against Pinecone/OpenAI"


def _write_payload(settings, payload) -> Path:
    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload_path = artifacts_dir / "last_eval.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload_path


def _print_report(k, n_questions, primary, ablations, source):
    print(f"Retrieval evaluation (k={k}, n={n_questions}) — {source}")
    print()
    print("Mode     Precision@{k}    Recall@{k}       Ceiling    Normalized")
    print("-------  -------------  -------------  ---------  ----------")
    for mode in ("vector", "bm25", "hybrid"):
        report = ablations.get(mode) or {}
        p = report.get("precision", 0.0)
        r = report.get("recall", 0.0)
        c = report.get("ceiling", 0.0)
        n = report.get("normalized_precision", 0.0)
        print(f"{mode:<8}  {p:>13.6f}  {r:>13.6f}  {c:>9.6f}  {n:>10.6f}")
    print()
    print("Primary (hybrid):")
    print(f"  Precision@{k}        {primary.get('precision', 0.0):.6f}")
    print(f"  Recall@{k}           {primary.get('recall', 0.0):.6f}")
    print(f"  Ceiling              {primary.get('ceiling', 0.0):.6f}")
    print(f"  Normalized precision {primary.get('normalized_precision', 0.0):.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
