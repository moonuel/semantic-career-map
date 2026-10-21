"""Port the curated LinkedIn markdown corpus into the raw layer.

LinkedIn has no key-free scraping API — its allowlisted ``linkedin`` source
is fed instead by this port of the hand-collected corpus in
``data/selected-job-postings/``. Each markdown file becomes one
content-addressed payload object holding only what the source gave (the
verbatim markdown and its format); employer/title resolution is a derived
concern for the normalization pipeline, not raw source content.

Idempotent like every raw-layer write: already-ported files are
hash-deduped (``duplicate`` observations), so the script can be re-run
whenever the hand-collected corpus grows. One ``lake/manifests/ingestion/``
manifest records the batch under its own ``ingestion_id`` (evidence of the
port; real posting provenance is the corpus curation itself).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from backend.ingestion import raw
from backend.ingestion.manifest import (
    BatchObservation,
    build_ingestion_manifest,
    pipeline_version,
    write_ingestion_manifest,
)
from backend.ingestion.raw import Sink
from backend.storage import ids
from backend.storage.store import ObjectStore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIR = _REPO_ROOT / "data" / "selected-job-postings"
# The corpus has no ATS board; manifest rows need a non-empty board.
_BOARD = "curated-md"


def run(*, source_dir: Path, sink: Sink) -> int:
    """Port every ``.md`` posting under ``source_dir`` into the raw layer.

    Args:
        source_dir: Directory of curated markdown postings.
        sink: The ingestion sink (the object store).

    Returns:
        Process exit code (0 on success).
    """
    run_started_at = datetime.now(UTC)
    existing_hashes: set[str] = set()
    observations = []
    for path in sorted(source_dir.glob("*.md")):
        payload = raw.canonical_json(
            {"format": "markdown", "content": path.read_text(encoding="utf-8")}
        )
        event = raw.ingest_payload(
            "linkedin",
            payload,
            sink=sink,
            existing_hashes=existing_hashes,
            board=_BOARD,
            source_id=path.stem,
        )
        observations.append(
            BatchObservation(
                source=event.source,
                action=event.event,
                content_sha256=event.content_sha256,
                source_id=event.source_id or None,
                board=event.board,
                employer=None,
                observed_at=datetime.now(UTC),
            )
        )
    ingestion_id = ids.new_id()
    manifest = build_ingestion_manifest(
        observations,
        ingestion_id=ingestion_id,
        ingested_on=run_started_at.date(),
        pipeline_version=pipeline_version(),
        configuration_hash=_corpus_configuration_hash(source_dir, sink),
        run_started_at=run_started_at,
        completed_at=datetime.now(UTC),
        status="ok",
        boards_attempted=1,
        boards_failed=0,
        board_status={_BOARD: "ok"},
    )
    manifest_key = write_ingestion_manifest(
        sink, manifest, ingested_on=run_started_at.date(), ingestion_id=ingestion_id
    )
    written = sum(o.action == "stored" for o in observations)
    duplicates = sum(o.action == "duplicate" for o in observations)
    print(
        f"linkedin: written={written} duplicates={duplicates} of "
        f"{len(observations)} corpus files  manifest={manifest_key}"
    )
    return 0


def _corpus_configuration_hash(source_dir: Path, sink: Sink) -> str:
    """Configuration hash for the port run: the corpus listing and sink.

    The port's only "configuration" is which files it ported and where; the
    hash covers both, so a run against a grown corpus is traceable to its
    exact inputs.
    """
    files = sorted(p.name for p in source_dir.glob("*.md"))
    return ids.content_sha256(
        raw.canonical_json(
            {"files": files, "sink": type(sink).__name__, "source_dir": str(source_dir)}
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_SOURCE_DIR,
        help=f"curated markdown corpus (default: {_SOURCE_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        load_dotenv(_REPO_ROOT / ".env")
        sink: Sink = ObjectStore.from_env()
        return run(source_dir=args.source_dir, sink=sink)
    except (BotoCoreError, ClientError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
