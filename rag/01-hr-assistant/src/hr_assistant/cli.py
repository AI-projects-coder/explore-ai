import argparse
from pathlib import Path

from hr_assistant.core.config import get_settings
from hr_assistant.core.logs import configure_logging
from hr_assistant.db.models import RunStatus, RunTrigger
from hr_assistant.db.retention import count_expired, purge_expired
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

    cleanup = commands.add_parser("cleanup", help="delete data older than the retention periods")
    cleanup.add_argument("--dry-run", action="store_true", help="only count what would be deleted")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(json_logs=settings.json_logs, level=settings.log_level)
    if args.command == "cleanup":
        if args.dry_run:
            print(f"would delete: {count_expired(get_session_factory(), settings)}")
        else:
            print(f"deleted: {purge_expired(get_session_factory(), settings)}")
        return 0

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
