"""Seed the lakehouse raw layer from each source site.

One-off acquisition that populates the raw layer of the S3-compatible store
(the bucket whose ``lake/raw`` branch is defined in ``docs/architecture/data.md``;
``--target store``, default — the local Garage container) or its local file
mirror under ``data/`` (``--target files``), with a small set of postings from
every allowlisted source, so the storage I/O layer has data to be built and
tested against.

For the ATS sites — Ashby, Greenhouse, and Lever — a fixed list of public
boards is fetched through :mod:`backend.ingestion.scrapers`, and the first
``--per-board`` postings per board are normalized into the canonical
:class:`~backend.ingestion.scrapers.ScrapedJob` payload. For LinkedIn, which
has no key-free API, ``--linkedin-limit`` selected postings are ported from
the existing curated corpus in ``data/selected-job-postings/`` as their
verbatim markdown.

Each posting is stored as one content-addressed raw object:

    lake/raw/source=<source>/sha256=<hash>.json

The payload is the posting record serialized to canonical JSON (sorted keys,
no ASCII escaping); the SHA-256 of those exact bytes is the identity minted
through :func:`backend.storage.keys.raw_payload`. Ingestion metadata such as
fetch time deliberately stays out of the payload so identical content always
hashes to the same object (data.md §4), and the key carries no time partition
so each payload is stored once: re-running this script is a no-op beyond
already-seeded objects. Provenance dates live in batch manifests, not in keys.

This is production ingestion, not an experiment artifact; it deliberately
lives in an unnumbered file under ``scripts/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import requests
from dotenv import load_dotenv

from backend.ingestion.scrapers import (
    ScrapedJob,
    fetch_ashby_jobs,
    fetch_greenhouse_jobs,
    fetch_lever_jobs,
)
from backend.storage import ids, keys
from backend.storage.store import ObjectStore

# Seed corpus: (source, board, employer override or None). Two boards per ATS
# give employer diversity within each site; all are public, key-free boards.
_BOARDS: list[tuple[str, str]] = [
    ("ashby", "supabase"),
    ("ashby", "elevenlabs"),
    ("greenhouse", "anthropic"),
    ("greenhouse", "vercel"),
    ("lever", "palantir"),
    ("lever", "moonpay"),
]

# LinkedIn has no public API; its source corpus is the curated set
# already collected by hand in data/selected-job-postings/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_LINKEDIN_DIR = _REPO_ROOT / "data" / "selected-job-postings"
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"

_FETCHERS = {
    "ashby": fetch_ashby_jobs,
    "greenhouse": fetch_greenhouse_jobs,
    "lever": fetch_lever_jobs,
}


def _canonical_json(payload: dict) -> bytes:
    """Serialize a payload to the canonical raw-object bytes."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _site_payloads(
    board: str, source: str, limit: int
) -> Iterator[tuple[bytes, ScrapedJob]]:
    """Fetch one ATS board and yield ``limit`` usable postings.

    Postings are taken in stable ``source_id`` order and must carry a title
    and non-empty description, so the seeded corpus contains real content
    regardless of board ordering.
    """
    jobs = _FETCHERS[source](board)
    usable = sorted(
        (job for job in jobs if job.title and job.description_html),
        key=lambda job: job.source_id or "",
    )
    for job in usable[:limit]:
        yield _canonical_json(dataclasses.asdict(job)), job


def _linkedin_payloads(path: Path) -> tuple[bytes, ScrapedJob | None]:
    """Wrap one curated LinkedIn markdown posting as the raw payload.

    The payload holds only what the source gave: the verbatim markdown and
    its format. Employer/title resolution from the filename is a derived
    concern, not raw source content.
    """
    payload = {"format": "markdown", "content": path.read_text(encoding="utf-8")}
    return _canonical_json(payload), None


def _existing_lake_hashes(store: ObjectStore) -> set[str]:
    """Return the set of ``sha256=`` content hashes already under ``lake/raw/``."""
    seen: set[str] = set()
    for key in store.list_keys("lake/raw/"):
        leaf = key.rpartition("/")[2]
        if leaf.startswith("sha256=") and leaf.endswith(".json"):
            seen.add(leaf.removeprefix("sha256=").removesuffix(".json"))
    return seen


def run(
    *,
    target: str,
    store: ObjectStore | None = None,
    data_dir: Path | None = None,
    per_board: int,
    linkedin_limit: int,
) -> int:
    """Seed raw objects from every source site into the chosen target.

    Args:
        target: Where to write, ``store`` (S3-compatible Garage store) or
            ``files`` (local cache mirror under ``data_dir``).
        store: The object store, required for ``target="store"``.
        data_dir: Local lake cache root, required for ``target="files"``;
            each key is written as ``data_dir/<key>``.
        per_board: Postings to seed from each ATS board.
        linkedin_limit: Postings to port from the curated LinkedIn corpus.

    Returns:
        Process exit code (0 on success, 1 if any site failed entirely).
    """
    written_by_source: dict[str, int] = {}
    skipped_by_source: dict[str, int] = {}
    failures: list[str] = []

    if target == "store":
        assert store is not None
        existing_hashes = _existing_lake_hashes(store)

        def ingest(source: str, asset: bytes, note: str) -> None:
            content_hash = ids.content_sha256(asset)
            key = keys.raw_payload(source, content_hash)
            keys.validate_key(key)
            if content_hash in existing_hashes:
                skipped_by_source[source] += 1
                print(f"skip  {key}")
                return
            store.put(key, asset)
            existing_hashes.add(content_hash)
            written_by_source[source] += 1
            print(f"write {key}  <- {note}")

    else:
        assert data_dir is not None

        def ingest(source: str, asset: bytes, note: str) -> None:
            content_hash = ids.content_sha256(asset)
            key = keys.raw_payload(source, content_hash)
            keys.validate_key(key)
            path = data_dir / key
            if path.exists():
                skipped_by_source[source] += 1
                print(f"skip  {key}")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(asset)
            written_by_source[source] += 1
            print(f"write {key}  <- {note}")

    for source, board in _BOARDS:
        skipped_by_source.setdefault(source, 0)
        written_by_source.setdefault(source, 0)
        try:
            payloads = list(_site_payloads(board, source, per_board))
        except requests.RequestException as exc:
            failures.append(f"{source}/{board}: {exc}")
            print(f"error {source}/{board}: {exc}", file=sys.stderr)
            continue
        for payload, job in payloads:
            ingest(source, payload, note=job.source_id or job.url)

    for path in sorted(_LINKEDIN_DIR.glob("*.md"))[:linkedin_limit]:
        skipped_by_source.setdefault("linkedin", 0)
        written_by_source.setdefault("linkedin", 0)
        payload, _ = _linkedin_payloads(path)
        ingest("linkedin", payload, note=path.name)

    total_written = sum(written_by_source.values())
    print()
    for source in sorted(skipped_by_source):
        print(
            f"{source}: written={written_by_source[source]} "
            f"skipped={skipped_by_source[source]}"
        )
    if failures:
        print(f"site failures: {', '.join(failures)}", file=sys.stderr)
    print(f"total written={total_written}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("store", "files"),
        default="store",
        help="write to the S3-compatible store or the local file mirror "
        "(default: store)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help=f"local lake cache root for --target files (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--per-board",
        type=int,
        default=2,
        help="postings to seed from each ATS board (default: 2)",
    )
    parser.add_argument(
        "--linkedin-limit",
        type=int,
        default=4,
        help="postings to port from the curated LinkedIn corpus (default: 4)",
    )
    args = parser.parse_args(argv)

    store = None
    if args.target == "store":
        load_dotenv(_REPO_ROOT / ".env")
        store = ObjectStore.from_env()
    return run(
        target=args.target,
        store=store,
        data_dir=args.data_dir,
        per_board=args.per_board,
        linkedin_limit=args.linkedin_limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
