#!/usr/bin/env python3
"""Migrate the legacy golden set to the authored eval dataset (task 4.1).

One-off migration: ``data/golden_cleaned.json`` (39 hand-cleaned references,
annotator ``manual`` — history: 5 legacy postings from exp 003, 34 ported by
008's workpad flow) becomes
``lake/eval/datasets/type=cleaning_golden/variant=author/version=v1/``
Parquet rows — ``source``, ``source_hash``, ``source_id``, ``golden_text``,
``annotator``, ``created_at``. The legacy JSON stays untouched as artifact
history (this script reads it, never rewrites it).

The migration's real work is identity resolution: the lake has no
``posting_id`` (design D9/D11-rejected) — the golden rows' join handle is
``source_hash``, so each reference is matched to its raw payload by word-set
coverage (golden text ⊂ raw text by construction of hand-cleaning). Matches
are reported one per posting; a reference that matches nothing (below
:data:`MIN_COVERAGE`) or matches several payloads within
:data:`UNIQUE_MARGIN` of each other aborts the migration — an aborted run
writes nothing. The human-readable ``posting_id`` appears in the report and
in ``source_id`` only when the payload actually carries one.

Run with --dry (default) to inspect the match report; --write mints a run id
and puts the Parquet part.

Usage:
    python scripts/migrate_golden_cleaned.py               # dry
    python scripts/migrate_golden_cleaned.py --write       # write part
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from backend.pipeline.cleaning.normalization import (
    SHAPE_TEXT_FIELD,
    normalize_html,
)
from backend.pipeline.eval import metrics as m
from backend.pipeline.eval import run as eval_run
from backend.storage import ids, keys
from backend.storage.raw import RawPayload, fetch_all
from backend.storage.store import ObjectStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("source_hash", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("golden_text", pa.string()),
        pa.field("annotator", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)

# A golden reference counts as a match when at least this fraction of its
# word set appears in the candidate payload's text (hand-cleaning keeps
# almost every word, so scores sit far above realistic false positives).
MIN_COVERAGE = 0.85

# A match is unique when the runner-up's coverage trails the best by at
# least this margin; anything closer is reported AMBIGUOUS instead.
UNIQUE_MARGIN = 0.05

# Scraper-overlap tie-break: the same posting is often ingested from two
# sources (linkedin + dataroles), so several payloads can cover the golden
# words completely. Among those, the tightest superset (fewest extra words)
# is the reference's origin — but only when it beats the runner-up by at
# least this many extra words; closer than that stays AMBIGUOUS.
UNIQUE_EXTRA_WORDS = 3


def candidates(payloads: list[RawPayload]) -> list[tuple[set, RawPayload]]:
    """Precompute each payload's word set once (normalization is the cost).

    Args:
        payloads: Every raw payload in the lake.

    Returns:
        ``(word_set, payload)`` pairs, payload order preserved — the
        matcher's ready-to-score candidate list; empty payloads (no shape
        text) are dropped.
    """
    out = []
    for payload in payloads:
        words = m.word_set(payload_text(payload))
        if words:
            out.append((words, payload))
    return out


def payload_text(payload: RawPayload) -> str:
    """Return the payload's shape text field, normalized like cleaning sees.

    Args:
        payload: A parsed raw payload.

    Returns:
        The normalized text (or "" when the shape field is absent).
    """
    field = SHAPE_TEXT_FIELD[payload.source]
    return normalize_html(payload.data.get(field, ""))


def match_golden(
    golden_text: str,
    candidates: list[tuple[set, RawPayload]],
) -> tuple[RawPayload | None, float, int]:
    """Resolve one golden reference to its unique raw payload.

    Args:
        golden_text: The golden reference text.
        candidates: Precomputed ``(word_set, payload)`` candidates
            (:func:`candidates`).

    Returns:
        ``(payload, coverage, runners_up)`` — payload is None when either
        no candidate reaches :data:`MIN_COVERAGE`, the best candidates sit
        within :data:`UNIQUE_MARGIN` (incomplete coverage), or full-coverage
        scraper-overlap ties can't be broken by the tightest superset
        (:data:`UNIQUE_EXTRA_WORDS` margin). ``runners_up`` counts the
        near-miss candidates.
    """
    golden_words = m.word_set(golden_text)
    scored = sorted(
        ((len(golden_words & cw) / len(golden_words), cw, p) for cw, p in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best, _, best_payload = scored[0]
    if best < MIN_COVERAGE:
        return None, best, 0
    if best < 1.0:
        # Incomplete coverage: uniqueness rides on the coverage margin alone
        runners = sum(1 for cov, _, _ in scored[1:] if cov > best - UNIQUE_MARGIN)
        if runners:
            return None, best, runners
        return best_payload, best, 0
    # Full coverage: scraper-overlap ties are broken by the tightest
    # superset (fewest extra words) — the golden's origin is the payload
    # whose rendering the hand-cleaned text was drawn from
    full = [
        (len(cw - golden_words), idx, p)
        for idx, (cov, cw, p) in enumerate(scored)
        if cov >= 1.0
    ]
    full.sort(key=lambda item: item[0])
    best_extra, _, winner = full[0]
    if len(full) > 1 and full[1][0] - best_extra < UNIQUE_EXTRA_WORDS:
        return None, best, len(full) - 1
    return winner, best, 0


def load_legacy_golden(path: Path) -> tuple[list[dict], str]:
    """Read the legacy golden JSON and its content hash.

    Args:
        path: The legacy golden file (default ``data/golden_cleaned.json``).

    Returns:
        The reference list (``posting_id`` / ``source`` / ``text``) and the
        file's sha256 hex (printed again after the run — the verify that the
        legacy artifact was left untouched).
    """
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Migration flow: match every reference, verify, optionally write.

    Args:
        argv: CLI arguments (default ``sys.argv[1:]``).

    Returns:
        0 on success (dry or write); 2 when any reference failed to match,
        the verify failed, or --write targets a version already carrying
        parts — no part is written in that case.
    """
    load_dotenv()  # STORE_S3_* live in .env (003 convention)
    parser = argparse.ArgumentParser(
        prog="scripts/migrate_golden_cleaned.py",
        description="Migrate data/golden_cleaned.json to the authored "
        "cleaning_golden eval dataset (match by word-set coverage).",
    )
    parser.add_argument(
        "--golden-path",
        default=str(PROJECT_ROOT / "data" / "golden_cleaned.json"),
        help="Legacy golden JSON path (read-only).",
    )
    parser.add_argument(
        "--golden-variant",
        default="author",
        help="Authoring variant of the target dataset.",
    )
    parser.add_argument(
        "--golden-version",
        default="v1",
        help="Version token of the target dataset (vN).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent raw-GET bound per source.",
    )
    parser.add_argument("--write", action="store_true", help="Write the part.")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Skip part writing (a previous --write landed it); write the "
        "dataset manifest only.",
    )
    parser.add_argument(
        "--created-at",
        default=None,
        help="Override the ISO-8601 row timestamp (defaults to now).",
    )
    args = parser.parse_args(argv)

    golden_path = Path(args.golden_path)
    refs, sha_before = load_legacy_golden(golden_path)
    print(
        f"legacy golden: {golden_path} ({len(refs)} references, sha {sha_before[:12]})"
    )

    store: object = ObjectStore.from_env()
    payloads: list[RawPayload] = []
    for source in sorted(keys.RAW_SOURCES):
        fetched = fetch_all(store, source, workers=args.workers)
        payloads.extend(p for p in fetched if p is not None)
        print(f"  source={source}: {len(fetched)} raw objects")

    created_at = args.created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return migrate(
        store,
        refs,
        candidates(payloads),
        sha_before=sha_before,
        write=args.write or args.manifest_only,
        part=not args.manifest_only,
        variant=args.golden_variant,
        version=args.golden_version,
        created_at=created_at,
        golden_path=golden_path,
    )


def migrate(
    store: object,
    refs: list[dict],
    candidates: list[tuple[set, RawPayload]],
    *,
    sha_before: str,
    write: bool,
    variant: str,
    version: str,
    created_at: str,
    golden_path: Path,
    part: bool = True,
) -> int:
    """Match, verify and (optionally) write the golden dataset part.

    Args:
        store: Object store (or testing double).
        refs: Legacy golden references (``posting_id`` / ``source`` / ``text``).
        candidates: Precomputed candidate pairs
            (:func:`candidates` over every raw payload).
        sha_before: The legacy file's sha256 at load (verify it stays).
        write: Write the Parquet part when True; dry otherwise.
        variant: Authoring variant of the target dataset.
        version: Version token of the target dataset (``vN``).
        created_at: ISO-8601 row timestamp for the migration event.
        golden_path: The legacy golden file (read-only; re-hashed after).

    Returns:
        0 on success (dry or write); 2 when any reference failed to match
        or the verify failed — no part is written in that case.
    """
    rows: list[dict] = []
    failures = 0
    print(f"\n{'posting_id':<48} {'coverage':>8}  finding")
    for ref in refs:
        posting_id = ref["posting_id"]
        payload, coverage, runners = match_golden(ref["text"], candidates)
        if payload is None:
            if runners:
                reason = f"AMBIGUOUS ({runners} within margin, best {coverage:.3f})"
            else:
                reason = f"NO MATCH (best coverage {coverage:.3f})"
            print(f"{posting_id:<48} {'—':>8}  {reason}")
            failures += 1
            continue
        print(
            f"{posting_id:<48} {coverage:>8.3f}  {payload.source} {payload.source_hash[:12]}"
        )
        rows.append(
            {
                "source": payload.source,
                "source_hash": payload.source_hash,
                "source_id": payload.data.get("source_id", ""),
                "golden_text": ref["text"],
                "annotator": ref.get("source", "manual"),
                "created_at": created_at,
            }
        )

    print(f"\nmatched {len(rows)}/{len(refs)} references")
    if failures:
        print(
            "ABORT: unmatched/ambiguous references must be resolved by hand "
            "(update the legacy golden or the matcher thresholds); "
            "nothing was written."
        )
        return 2

    golden_sha_after = hashlib.sha256(golden_path.read_bytes()).hexdigest()
    if golden_sha_after != sha_before:
        print("ABORT: the legacy golden file changed during the run; nothing written.")
        return 2

    if not write:
        print("dry run: no dataset part written (--write mints one)")
        return 0

    dataset_prefix = eval_run.golden_prefix(variant, version)
    run_id = ids.new_id()
    if part:
        # Idempotency guard: the dataset version is append-only and
        # downstream loaders list every part under the prefix, so a second
        # --write would silently double the ground truth.
        existing = [
            k for k in store.list_keys(dataset_prefix) if k.endswith(".parquet")
        ]
        if existing:
            print(
                "ABORT: the dataset version already carries "
                f"{len(existing)} part(s) under {dataset_prefix}; rerun with "
                "--manifest-only to reuse the existing part, or bump "
                "--golden-version to target a new version."
            )
            return 2
        part_key = keys.eval_dataset("cleaning_golden", variant, version, run_id, 1)
        table = pa.Table.from_pylist(rows, schema=GOLDEN_SCHEMA)
        sink = io.BytesIO()
        pq.write_table(table, sink)
        store.put(part_key, sink.getvalue())
        print(f"wrote {part_key} ({len(rows)} rows)")
    else:
        # --manifest-only: the part already landed in a previous --write
        # run; re-putting it would duplicate rows. Reuse the latest part.
        existing = sorted(
            k for k in store.list_keys(dataset_prefix) if k.endswith(".parquet")
        )
        if not existing:
            print("ABORT: --manifest-only with no part on the store")
            return 2
        part_key = existing[-1]
        run_id = PurePosixPath(part_key).name.removeprefix("part-").rsplit("-", 1)[0]
    back = pq.read_table(io.BytesIO(store.get(part_key)))
    print(f"round-trip {back.num_rows} rows")
    if back.num_rows != len(refs):
        print(f"ABORT: round-trip row count {back.num_rows} != {len(refs)}")
        return 2
    dataset_id = f"cleaning_golden-{variant}-{version}"
    manifest_key = keys.manifest_dataset(dataset_id)
    manifest = {
        "dataset_id": dataset_id,
        "kind": "authored_eval_dataset",
        "input_dataset": str(golden_path),
        "input_sha256": sha_before,
        "annotator": "manual",
        "record_count": back.num_rows,
        "pipeline_version": "n/a (one-off migration script)",
        "created_at": created_at,
        "run_id": run_id,
    }
    store.put(
        manifest_key,
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    )
    print(f"wrote {manifest_key}")
    print(f"legacy file untouched: {golden_sha_after[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
