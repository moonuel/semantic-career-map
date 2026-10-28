"""Backfill the embedding version-claim registry for prefixes already in use.

Why this exists: ``version=v1`` was written before claims existed, and it
already holds rows from two configurations (issue #22). Enforcing claims
without a backfill would make the next production run fail on day one — the
registry would be empty and the prefix in use.

The backfill records the *intended* owner, not the observed population. It
names the configuration the operator says owns the prefix; rows in that prefix
belonging to any other configuration are left physically in place and become a
measurable discrepancy instead of an invisible one. Healing those rows (or
moving them to their own prefix) is a separate, deliberate operation.

Usage::

    python scripts/claim_embedding_version.py --configuration-hash a29a35e3…   # dry run
    python scripts/claim_embedding_version.py --configuration-hash a29a35e3… --apply

Idempotent: re-running against a prefix it already claimed is a no-op, and a
prefix claimed by a different configuration is refused rather than overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline.embedding.version_claim import ClaimConflict, claim_version
from backend.storage import keys
from backend.storage.store import ObjectStore


def main() -> int:
    """Claim one embedding version prefix for a configuration hash."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all-minilm-l6-v2")
    parser.add_argument("--variant", default="minilm")
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--configuration-hash",
        required=True,
        help="the 64-char hex configuration_hash the operator declares the owner",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id to record (default: mint one for the backfill event)",
    )
    parser.add_argument(
        "--note",
        default="backfilled",
        help="free-text note stored with the claim",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the claim (default: print only)"
    )
    args = parser.parse_args()

    key = keys.embedding_version_claim(args.model, args.variant, args.version)
    store = ObjectStore.from_env()

    if not args.apply:
        _print_plan(key, store, args)
        return 0

    from backend.storage import ids

    run_id = args.run_id or ids.new_id()
    try:
        record = claim_version(
            store,
            _Proxy(args),
            args.configuration_hash,
            run_id=run_id,
            claimed_at=datetime.now(UTC),
            note=args.note,
        )
    except ClaimConflict as exc:
        print(f"REFUSED: {exc}")
        return 1

    print(f"claim -> {key}")
    print(json.dumps(record, indent=2, sort_keys=True))
    if record.get("claimed_by_run_id") != run_id:
        print("(already claimed by this configuration; nothing written)")
    return 0


def _print_plan(key: str, store, args: argparse.Namespace) -> None:
    """Print the claim that would be written, and the current state."""
    print(f"target: {key}")
    try:
        existing = json.loads(store.get(key))
    except FileNotFoundError:
        print("state: unclaimed")
    else:
        same = existing.get("configuration_hash") == args.configuration_hash
        print(f"state: claimed by {existing.get('configuration_hash')}")
        print(f"       this configuration matches: {same}")
    print(
        f"would claim: {args.configuration_hash}  (kind={keys.version_kind(args.version)})"
    )
    print("dry run — pass --apply to write")


class _Proxy:
    """Minimal config surface ``claim_version`` needs (model/variant/version).

    The backfill has no run configuration: it knows only the prefix and the
    declared owner, so it supplies exactly the three fields the claim key and
    record read.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.model_segment = args.model
        self.variant = args.variant
        self.version = args.version


if __name__ == "__main__":
    raise SystemExit(main())
