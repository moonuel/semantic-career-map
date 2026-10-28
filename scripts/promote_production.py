"""Publish a new production pointer: the operator write path.

The CLI over :mod:`backend.storage.pointer` — the only way anyone
permissions-promotes a production generation. The default is dry-run
(``--show`` semantics: print the current pointer, the resolution, and the
plan); nothing writes without ``--apply``. Mirrors
``scripts/claim_embedding_version.py`` conventions: dry by default, the
refusal prefix, and a refusal the operator must make land.

Build-then-flip (settled): a pointer may name only *complete* generations —
an embeddings entry whose claim or ``configs/<hash>.json`` object is
missing, or any tier whose prefix has no parts on disk, is refused. A
generation that was never built cannot be promoted into production by a
typo; the flag ``--allow-incomplete`` is the explicit, loudly-logged
override for the deliberate case (e.g. promoting the target before the
parts land, then letting the run fill the prefix).

Usage::

    python scripts/promote_production.py                          # show current + resolution
    python scripts/promote_production.py --bootstrap --apply      # first bootstrap (creates)
    python scripts/promote_production.py --from-file doc.json --apply

Idempotence: a create over an existing pointer is refused (bootstrap only
ever targets a fresh environment); a replace is content-addressed — a race
loser re-reads the pointer and re-plans.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline.production import ProductionSet, resolve_production
from backend.storage import keys
from backend.storage.pointer import (
    PointerConflict,
    content_hash,
    create,
    parse_document,
    read_pointer,
    replace,
)
from backend.storage.store import ObjectStore


def build_bootstrap_doc(args: argparse.Namespace) -> dict:
    """Build the hardcoded-reality bootstrap document from CLI flags.

    Args:
        args: The parsed CLI namespace (``updated_by``, ``updated_at``-
            independent; the embeddings tier flags).

    Returns:
        The bootstrap pointer document (validated by :func:`parse_document`
        before any write).

    Raises:
        ValueError: when a tier segment fails its key-builder re-mint
            (e.g. an ephemeral ``x-...`` version passed on the CLI).
    """
    doc = {
        "pointer_version": 1,
        "updated_by": args.updated_by,
        "updated_at": datetime.now(UTC).isoformat(),
        "reason": "bootstrap: describes reality",
        "previous": None,
        "tiers": {
            "cleaning": {
                "artifact_type": "cleaned_text",
                "variant": "llm",
                "version": "v1",
            },
            "embeddings": {
                "entries": [
                    {
                        "model_segment": args.model_segment,
                        "variant": args.variant,
                        "version": args.version,
                    }
                ]
            },
        },
    }
    return doc


def load_target_doc(args: argparse.Namespace) -> dict:
    """Build or load the candidate replacement document.

    Args:
        args: The parsed CLI namespace (``bootstrap``, ``from_file``,
            ``updated_by``, ``reason``).

    Returns:
        The candidate document (unvalidated; the caller parses).

    Raises:
        ValueError: from :func:`build_bootstrap_doc` on a bad tier segment,
            or ``json.JSONDecodeError`` from an unreadable ``--from-file``.
    """
    if args.bootstrap:
        return build_bootstrap_doc(args)
    with open(args.from_file, encoding="utf-8") as handle:
        doc = json.load(handle)
    doc.setdefault("pointer_version", 1)
    doc.setdefault("updated_by", args.updated_by)
    doc.setdefault("updated_at", datetime.now(UTC).isoformat())
    doc.setdefault("reason", args.reason)
    return doc


def readiness_refusals(store: ObjectStore, doc: dict) -> list[str]:
    """Build-then-flip: the checks a promotion must pass before flipping.

    Every embeddings entry must own a claim with a live, loadable config
    (the resolution chain's own verification); every tier's prefix must
    have parts on disk. Refusals name the exact operator command.

    Args:
        store: The object store.
        doc: The candidate document (parse-validated).

    Returns:
        One refusal line per failure; empty when the flip may apply.
    """
    from backend.pipeline.production import verify_embedding_entry

    refusals: list[str] = []
    for entry in doc["tiers"]["embeddings"]["entries"]:
        try:
            verify_embedding_entry(store, entry)
        except RuntimeError as exc:
            refusals.append(str(exc))
            continue
        model_segment, variant, version = (
            entry["model_segment"],
            entry["variant"],
            entry["version"],
        )
        prefix = (
            f"lake/embeddings/model={model_segment}/"
            f"variant={variant}/version={version}/"
        )
        if not any(key.endswith(".parquet") for key in store.list_keys(prefix)):
            refusals.append(
                f"embedding stack {model_segment}/{variant}/{version} is "
                "partless — build the generation first, then flip "
                "(override: --allow-incomplete)"
            )
    cleaning = doc["tiers"]["cleaning"]
    artifact_prefix = (
        f"lake/artifacts/type={cleaning['artifact_type']}/"
        f"variant={cleaning['variant']}/version={cleaning['version']}/"
    )
    if not any(key.endswith(".parquet") for key in store.list_keys(artifact_prefix)):
        refusals.append(
            f"cleaning stack {cleaning['artifact_type']}/{cleaning['variant']}/"
            f"{cleaning['version']} is partless — run the cleaning tier first, "
            "then flip (override: --allow-incomplete)"
        )
    return refusals


def show(store: ObjectStore) -> None:
    """The default dry view: current pointer content, hash, and resolution."""
    try:
        current = read_pointer(store)
    except FileNotFoundError:
        print("state: no production pointer")
        print("fix: python scripts/promote_production.py --bootstrap --apply")
        return
    except ValueError as exc:
        print(f"state: UNREADABLE — {exc}")
        return
    current_hash = content_hash(current)
    print(f"current: {keys.state_pointer()}")
    print(f"content hash: {current_hash}")
    print(json.dumps(current, indent=2, sort_keys=True))
    try:
        resolved: ProductionSet | None = resolve_production(store)
    except (RuntimeError, ValueError, PointerConflict) as exc:
        print(f"resolution: BROKEN — {exc}")
        return
    for i, emb in enumerate(resolved.embeddings):
        print(
            f"embeddings[{i}]: {emb.stack} owner {emb.configuration_hash[:12]}… "
            f"config {emb.config.model_name} @ {emb.config.model_revision}"
        )


def main() -> int:
    """CLI flow: show (default) or plan-and-apply a promotion."""
    load_dotenv()  # local convention; containers rely on task-def env
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="build-then-flip: a pointer may name only complete generations "
        "(claims + configs + parts live before the flip).",
    )
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--from-file", default=None)
    parser.add_argument(
        "--model-segment", default="all-minilm-l6-v2", help="bootstrap embeddings entry"
    )
    parser.add_argument(
        "--variant", default="minilm", help="bootstrap embeddings entry"
    )
    parser.add_argument(
        "--version", default="v1", help="bootstrap embeddings entry (stable vN)"
    )
    parser.add_argument(
        "--reason", default="manual promotion", help="recorded in the document"
    )
    parser.add_argument(
        "--updated-by",
        default=None,
        help="operator name recorded in the document (default: $USER)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="the visible escape hatch for promoting a not-yet-built generation",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write (default: print only)"
    )
    args = parser.parse_args()
    args.updated_by = args.updated_by or getpass.getuser()

    store = ObjectStore.from_env()

    if not args.apply:
        show(store)
        if args.bootstrap or args.from_file:
            target = load_target_doc(args)
            parse_document(target)  # refuse an unusable plan at planning time
            print("would write (dry run — pass --apply):")
            print(json.dumps(target, indent=2, sort_keys=True))
        return 0

    target = load_target_doc(args)
    try:
        parse_document(target)
    except ValueError as exc:
        print(f"REFUSED: {exc}")
        return 1

    try:
        current = read_pointer(store)
    except FileNotFoundError:
        current = None
    except ValueError as exc:
        print(f"REFUSED: pointer unreadable — {exc}")
        return 1

    if args.bootstrap and current is not None:
        print(
            "REFUSED: pointer already exists — bootstrap only targets a fresh environment"
        )
        return 1
    if current is None and not args.bootstrap:
        print("REFUSED: pointer absent; replace never creates — bootstrap first")
        return 1

    if current is None:
        created = create(store, target)
        if not created:
            print("REFUSED: lost the create race — read the winner and re-plan")
            return 1
        print(f"created -> {keys.state_pointer()}")
    else:
        refusals = readiness_refusals(store, target)
        if refusals and not args.allow_incomplete:
            print("REFUSED: build-then-flip — the target generation is incomplete:")
            for refusal in refusals:
                print(f"  - {refusal}")
            return 1
        if refusals:
            print(
                f"OVERRIDE --allow-incomplete: proceeding past {len(refusals)} refusal(s):"
            )
            for refusal in refusals:
                print(f"  - {refusal}")
        target["previous"] = content_hash(current)
        try:
            applied = replace(store, target, expected_hash=content_hash(current))
        except PointerConflict as exc:
            print(f"REFUSED: {exc}")
            return 1
        print(f"replaced -> {keys.state_pointer()}")
        print(f"previous: {target['previous']}")
        print(f"new content hash: {content_hash(applied)}")
    print(f"updated_by: {target['updated_by']}  reason: {target['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
