"""Ingest curated job postings into the Garage object store.

One-off migration that moves the hand-picked markdown postings in
``data/selected-job-postings/`` into the ``live/raw/`` branch defined by
``docs/architecture/data-layout.md``. Each posting becomes two objects:

    live/raw/<id>/source.md       verbatim file bytes
    live/raw/<id>/metadata.json   provenance sidecar (PostingMetadata)

The posting id is a UUIDv7 minted at ingestion by ``ids.new_id()`` (see
``backend/storage/ids.py``). Re-running is safe: a posting whose
``content_sha256`` is already present under ``live/raw/`` is skipped, so the
corpus is not duplicated.

This is production ingestion, not an experiment artifact; it deliberately lives
in an unnumbered directory under ``scripts/``. Store access now lives in
:mod:`backend.storage.store` (extracted when the second consumer,
``scripts/seed_lake_raw``, appeared); the metadata schema is still kept inline
here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from backend.storage import ids, keys
from backend.storage.store import ObjectStore

# --- Config ------------------------------------------------------------------

# Employers whose slug contains hyphens, so a mechanical split cannot recover
# them. Reviewed against all 37 filenames in the corpus. Small and explicit by
# design; expected to be replaced when ingestion scales.
_MULTIWORD_EMPLOYERS: frozenset[str] = frozenset(
    {
        "canada-life",
        "fgf-brands",
        "industrial-and-commercial-bank-of-china",
        "motion-recruitment",
        "ontario-teachers-pension-plan",
        "promise-robotics",
        "venterra-realty",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOURCE = _REPO_ROOT / "data" / "selected-job-postings"


# --- Metadata schema ---------------------------------------------------------


def split_slug(stem: str) -> tuple[str, str]:
    """Split a posting filename stem into ``(employer, title)``.

    The first token is the employer unless the stem starts with a known
    multi-word employer, in which case that whole prefix is used. The
    remainder is the title.

    Args:
        stem: filename without extension, e.g. ``canada-life-data-analyst``.

    Returns:
        A ``(employer, title)`` pair, e.g. ``("canada-life", "data-analyst")``.

    Raises:
        ValueError: if the stem has no recognisable employer/title split.
    """
    for employer in _MULTIWORD_EMPLOYERS:
        prefix = employer + "-"
        if stem.startswith(prefix):
            title = stem[len(prefix) :]
            if title:
                return employer, title
    head, sep, tail = stem.partition("-")
    if not sep or not head or not tail:
        raise ValueError(f"cannot split filename stem into employer/title: {stem!r}")
    return head, tail


@dataclass(frozen=True)
class PostingMetadata:
    """Provenance sidecar for a raw posting (``metadata.json``).

    Minimal fields only: identity, source, coarse labels, ingestion time, and
    the content hash. No parsed fields — those belong to later, derived
    artifacts, not to the raw sidecar.

    Attributes:
        id: Canonical UUIDv7 posting id, as minted by ``ids.new_id()``.
        source: Origin URL, or ``None`` for local files with no URL.
        employer: Employer derived from the filename slug.
        title: Role title derived from the filename slug.
        ingested_at: UTC ingestion timestamp, ISO-8601 with ``Z`` suffix.
        content_sha256: Lowercase hex SHA-256 of the verbatim source bytes.
    """

    id: str
    source: str | None
    employer: str
    title: str
    ingested_at: str
    content_sha256: str

    @classmethod
    def from_source(
        cls,
        *,
        source_bytes: bytes,
        stem: str,
        source: str | None = None,
        ingested_at: str | None = None,
    ) -> PostingMetadata:
        """Build metadata for a posting from its bytes and filename stem.

        Mints a fresh id and computes the content hash, so callers never
        supply them independently.

        Args:
            source_bytes: Verbatim posting bytes.
            stem: Filename without extension.
            source: Optional origin URL.
            ingested_at: Optional timestamp; defaults to now (UTC).

        Returns:
            A validated ``PostingMetadata``.
        """
        employer, title = split_slug(stem)
        meta = cls(
            id=ids.new_id(),
            source=source,
            employer=employer,
            title=title,
            ingested_at=ingested_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            content_sha256=ids.content_sha256(source_bytes),
        )
        meta.validate()
        return meta

    def to_json(self) -> str:
        """Serialize to canonical JSON (sorted keys, no ASCII escaping)."""
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> PostingMetadata:
        """Parse and validate a ``PostingMetadata`` from JSON text."""
        obj: dict[str, Any] = json.loads(raw)
        meta = cls(**obj)
        meta.validate()
        return meta

    def validate(self) -> None:
        """Raise ``ValueError`` if any field violates the schema invariants."""
        if not ids.is_canonical_uuid7(self.id):
            raise ValueError(f"id {self.id!r} is not a canonical UUIDv7")
        if self.source is not None and not self.source:
            raise ValueError("source must be a non-empty string or None")
        if not self.employer:
            raise ValueError("employer must be non-empty")
        if not self.title:
            raise ValueError("title must be non-empty")
        if not ids.is_sha256_hex(self.content_sha256):
            raise ValueError(
                f"content_sha256 {self.content_sha256!r} must be 64-char hex"
            )


# --- Store (inline round-trip utilities) -------------------------------------


# --- Migration ---------------------------------------------------------------


def _existing_hashes(store: ObjectStore) -> set[str]:
    """Return the set of ``content_sha256`` values already under ``live/raw/``."""
    seen: set[str] = set()
    for key in store.list_keys("live/raw/"):
        if not key.endswith("/metadata.json"):
            continue
        try:
            raw = store.get(key).decode("utf-8")
            seen.add(json.loads(raw)["content_sha256"])
        except FileNotFoundError, KeyError, ValueError, UnicodeDecodeError:
            continue
    return seen


def _planned(postings: Path) -> list[tuple[Path, bytes, PostingMetadata]]:
    """Build planned writes for every ``.md`` file under ``postings``."""
    items: list[tuple[Path, bytes, PostingMetadata]] = []
    for path in sorted(postings.glob("*.md")):
        data = path.read_bytes()
        meta = PostingMetadata.from_source(source_bytes=data, stem=path.stem)
        items.append((path, data, meta))
    return items


def run(*, source_dir: Path, dry_run: bool) -> int:
    """Migrate postings from ``source_dir`` into the store.

    Args:
        source_dir: Directory of ``.md`` postings.
        dry_run: If True, print the plan and write nothing.

    Returns:
        Process exit code (0 on success).
    """
    store = ObjectStore.from_env()
    existing = set() if dry_run else _existing_hashes(store)
    planned = _planned(source_dir)

    written = skipped = 0
    for path, data, meta in planned:
        keys_ok = keys.raw_posting(meta.id)
        if meta.content_sha256 in existing:
            skipped += 1
            print(f"skip  {path.name} (hash already ingested)")
            continue
        if dry_run:
            print(f"plan  {path.name} -> {keys_ok}")
            continue
        store.put(keys.raw_posting(meta.id), data)
        store.put(keys.raw_metadata(meta.id), meta.to_json().encode("utf-8"))
        written += 1
        print(f"write {path.name} -> {keys_ok}")

    verb = "planned" if dry_run else "written"
    print(f"\n{verb}={written if not dry_run else len(planned)} skipped={skipped}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"directory of postings (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without writing anything",
    )
    args = parser.parse_args(argv)

    load_dotenv(_REPO_ROOT / ".env")
    try:
        return run(source_dir=args.source, dry_run=args.dry_run)
    except (BotoCoreError, ClientError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
