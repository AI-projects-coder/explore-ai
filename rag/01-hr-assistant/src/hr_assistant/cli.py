import argparse
import logging
from pathlib import Path

from hr_assistant.core.config import get_settings
from hr_assistant.db.models import RunStatus, RunTrigger
from hr_assistant.db.session import get_session_factory
from hr_assistant.embeddings.vertex import VertexEmbedder
from hr_assistant.ingestion.docling_parser import DoclingParser
from hr_assistant.ingestion.loader import LocalManifestLoader
from hr_assistant.ingestion.pipeline import run_ingestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hr-assistant")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index the documents in a folder (manifest.yaml required)")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--trigger", choices=[t.value for t in RunTrigger], default=RunTrigger.MANUAL.value)
    ingest.add_argument("--allow-mass-delete", action="store_true", help="skip the mass-deletion safety check")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    run = run_ingestion(
        loader=LocalManifestLoader(args.path),
        parser=DoclingParser(),
        embedder=VertexEmbedder.from_settings(settings),
        session_factory=get_session_factory(),
        settings=settings,
        trigger=args.trigger,
        allow_mass_delete=args.allow_mass_delete,
    )
    d = run.details
    print(
        f"\nrun {run.id}: {run.status}\n"
        f"  new={run.docs_new} changed={run.docs_changed} unchanged={run.docs_unchanged} "
        f"access_updated={d['access_updated']} deleted={run.docs_deleted} failed={run.docs_failed}\n"
        f"  chunks embedded={d['chunks_embedded']} reused={d['chunks_reused']}  embed_tokens={run.embed_tokens}"
    )
    for problem in d["problems"]:
        print(f"  problem: {problem}")
    for uri, err in d["failures"].items():
        print(f"  FAILED {uri}: {err}")
    if run.error:
        print(f"  error: {run.error}")
    return 0 if run.status == RunStatus.SUCCEEDED else 1  # non-zero exit marks a Cloud Run job as failed
