"""Layer 1 spike: prove the production-pointer resolution chain end to end.

Production-pointer plan (state/ branch), Layer 1 — the thinnest read/write
proof of the design settled in issue #22 requirement 1: one mutable object at
``state/production.json`` names the active production generation for every
scheduled tier as a map of partition segments; scheduled jobs resolve their
configuration downward from it (pointer → claim → ``configs/<hash>.json`` →
rows). Pointers name; they never search.

This pass does exactly one write: the idempotent bootstrap create of the
pointer document describing current reality (``put_if_absent`` — a re-run
prints the already-created no-op). It then resolves the pointer live,
verifies the embeddings claim owner and its pinned configuration, and
counts production rows (claim-hash filtered) against the unfiltered
partition-stack count through the shared Athena runner. The claim-filtered
count grows with every scheduled delta run, so the probe pins the
structural fact instead of a snapshot: the gap between the unfiltered
prefix and the claim-filtered count is exactly the 400 legacy foreign
rows in ``version=v1`` (a second configuration's population, issue #22).
At spike time the live pair was 9,570 / 9,970; the plan's 9,354 / 9,754
predates a 216-row delta embed.

Why the count pair: the claim-hash read rule is the feature's core —
production rows are rows under the pointer's pair stack whose
``configuration_hash`` equals the claim's hash. The 400-row gap is the
foreign population in ``version=v1`` that the rule isolates from every
production read.

Spike, not production: hardcoded pointer document, an inline CAS primitive
(``_cas_replace`` + ``_PointerConflict``) exercised against a plain dict
(no live write), no modules. The write half graduates to
``scripts/promote_production.py`` and the read chain to
``backend/pipeline/production.py`` in Layer 3; this probe is then deleted.

Usage:
    python scripts/production_set_probe.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.pipeline.athena import (
    aws_region,
    iter_query_results,
    poll_query_completion,
    results_location,
    submit_query,
)
from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash
from backend.storage import keys
from backend.storage.store import ObjectStore

POINTER_VERSION = 1
DATABASE = "semantic_career_map"
WORKGROUP = "semantic_career_map"
# The legacy deploy-sample population a second configuration wrote into
# production ``version=v1`` before claims existed (issue #22). The claim-hash
# read rule must exclude exactly these rows — no more, no fewer.
EXPECTED_FOREIGN_ROWS = 400
MAX_SCAN_BYTES = 256 * 1024 * 1024

# The live claim owner this bootstrap must describe (issue #22 backfill).
EXPECTED_CLAIM_HASH = "a29a35e37a0d7025a961b5241ff278904500ed43edf022ac7304dacdb97e0c98"


def _pointer_document(updated_by: str) -> dict:
    """Build the hardcoded pointer document describing current reality."""
    return {
        "pointer_version": POINTER_VERSION,
        "updated_by": updated_by,
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
                        "model_segment": "all-minilm-l6-v2",
                        "variant": "minilm",
                        "version": "v1",
                    }
                ]
            },
        },
    }


def _canonical_bytes(doc: dict) -> bytes:
    """Serialize a pointer document to canonical JSON bytes (sorted, compact)."""
    return json.dumps(
        doc, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _content_hash(doc: dict) -> str:
    """Return the 64-char hex content hash of a pointer document."""
    return hashlib.sha256(_canonical_bytes(doc)).hexdigest()


def _fail(message: str) -> None:
    """Print a contract violation and exit non-zero."""
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _bootstrap(store: ObjectStore, doc: dict) -> None:
    """Create the pointer if absent; print the idempotent no-op otherwise."""
    key = keys.state_pointer()
    created = store.put_if_absent(key, _canonical_bytes(doc))
    state = "created" if created else "already present (no-op)"
    print(f"bootstrap: {state} -> {key}")


def _resolve(store: ObjectStore) -> dict:
    """Resolve pointer → claim → config; assert identities along the chain."""
    key = keys.state_pointer()
    pointer = json.loads(store.get(key))
    if pointer.get("pointer_version") != POINTER_VERSION:
        _fail(f"stored pointer has pointer_version {pointer.get('pointer_version')!r}")

    clean_c = pointer["tiers"]["cleaning"]
    print(
        "cleaning stack: "
        f"{clean_c['artifact_type']}/{clean_c['variant']}/{clean_c['version']}"
    )

    entries = pointer["tiers"]["embeddings"]["entries"]
    resolved: list[dict] = []
    for entry in entries:
        stack = f"{entry['model_segment']}/{entry['variant']}/{entry['version']}"
        claim_key = keys.embedding_version_claim(
            entry["model_segment"], entry["variant"], entry["version"]
        )
        try:
            claim = json.loads(store.get(claim_key))
        except FileNotFoundError:
            _fail(f"embeddings stack {stack} has no claim at {claim_key}")
        owner = claim["configuration_hash"]
        print(f"embeddings stack: {stack}")
        print(f"claim: {claim_key}")
        print(f"claim owner: {owner}")
        if owner != EXPECTED_CLAIM_HASH:
            _fail(f"claim owner {owner} does not match known live owner a29a35e3…")

        config_key = keys.config(owner)
        try:
            config_doc = json.loads(store.get(config_key))
        except FileNotFoundError:
            _fail(f"orphan claim: {config_key} is absent for claim owner {owner}")
        config = EmbeddingConfig(**config_doc["config"])
        if configuration_hash(config) != owner:
            _fail(
                f"config at {config_key} hashes to "
                f"{configuration_hash(config)[:12]}…, not the claim owner {owner[:12]}…"
            )
        print(f"config: {config_key}")
        print(f"model: {config.model_name} @ {config.model_revision}")
        print(f"input: {config.input_artifact} {config.input_version}")
        resolved.append(
            {
                "entry": entry,
                "claim_key": claim_key,
                "claim_owner": owner,
                "config": config,
            }
        )
    return {"pointer": pointer, "embeddings": resolved}


def _athena():
    """An Athena client pinned to the lake bucket's region (repo rule)."""
    return boto3.Session().client(
        "athena",
        region_name=aws_region(),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def _count(athena, resolved: dict) -> tuple[int, int, int]:
    """One Athena scan: claim-filtered vs unfiltered counts for the stack."""
    entry = resolved["embeddings"][0]["entry"]
    claim_owner = resolved["embeddings"][0]["claim_owner"]
    sql = f"""
SELECT COUNT(*) AS prefix_rows,
       SUM(CASE WHEN configuration_hash = '{claim_owner}' THEN 1 ELSE 0 END) AS claim_rows
FROM {DATABASE}.postings_embeddings
WHERE model_segment = '{entry["model_segment"]}'
  AND variant = '{entry["variant"]}'
  AND version = '{entry["version"]}'
"""
    execution_id = submit_query(
        athena,
        sql,
        database=DATABASE,
        workgroup=WORKGROUP,
        output_location=results_location(),
    )
    poll_query_completion(athena, execution_id)
    rows = list(iter_query_results(athena, execution_id))
    if len(rows) != 2:
        _fail(f"count query returned {len(rows)} rows, expected header + 1")
    header, data = rows
    if header != ["prefix_rows", "claim_rows"]:
        _fail(f"count query header {header!r} does not match the projection")
    stats = athena.get_query_execution(QueryExecutionId=execution_id)["QueryExecution"][
        "Statistics"
    ]
    scanned = int(stats.get("DataScannedInBytes", 0))
    prefix_rows, claim_rows = (int(cell) for cell in data)
    return prefix_rows, claim_rows, scanned


def _check_counts(prefix_rows: int, claim_rows: int, scanned: int) -> None:
    """Assert the claim-filtered read rule against known live facts."""
    if scanned > MAX_SCAN_BYTES:
        _fail(f"count query scanned {scanned / 2**20:.1f} MB, over the cap")
    foreign = prefix_rows - claim_rows
    if foreign != EXPECTED_FOREIGN_ROWS:
        _fail(
            f"claim-hash filter excludes {foreign} prefix rows, expected exactly "
            f"{EXPECTED_FOREIGN_ROWS} foreign rows; the claim-hash read rule "
            "is wrong or a new foreign population landed"
        )
    print(f"production rows (claim-hash filtered): {claim_rows}")
    print(f"unfiltered prefix rows: {prefix_rows}")
    print(f"foreign rows excluded: {foreign}")
    print(f"scanned_mb={scanned / 2**20:.1f}")


class _PointerConflict(RuntimeError):
    """Raised when a CAS replace carries a stale expected content hash."""


def _cas_replace(holder: dict, expected_hash: str, new_doc: dict) -> None:
    """Replace ``holder``'s contents in place; refuse a stale expectation.

    Compare-then-swap keyed by the document's content hash — the mutation
    primitive the spike's dict demo proves and Layer 3 graduates into
    ``pointer.replace`` (raising ``PointerConflict`` there).
    """
    if _content_hash(holder) != expected_hash:
        raise _PointerConflict(
            f"expected {expected_hash}, found {_content_hash(holder)}"
        )
    holder.update(new_doc)


def _demo_cas(base: dict) -> None:
    """In-memory CAS demo: promote cleaning v1→v2; refuse a stale replace."""
    print("CAS demo (in-memory plain dict, no live write)")
    pointer = copy.deepcopy(base)
    current = _content_hash(pointer)
    print(f"pointer content hash: {current}")

    promoted = copy.deepcopy(pointer)
    promoted["tiers"]["cleaning"]["version"] = "v2"
    promoted["reason"] = "demo: promote cleaning v1 -> v2"
    promoted["previous"] = current
    _cas_replace(pointer, current, promoted)
    applied_hash = _content_hash(pointer)
    print(f"applied replace: cleaning v1 -> v2 (previous={current})")
    print(f"new pointer content hash: {applied_hash}")

    # A second promoter slept through the flip: its expectation is stale,
    # and the primitive — not this narration — must refuse it.
    competing = copy.deepcopy(pointer)
    competing["tiers"]["cleaning"]["version"] = "v3"
    competing["reason"] = "demo: promote cleaning v1 -> v3"
    competing["previous"] = current
    try:
        _cas_replace(pointer, current, competing)
    except _PointerConflict as exc:
        print(f"stale replace with expected={current}: REFUSED ({exc})")
    else:
        _fail("stale replace against the flipped pointer was not refused")


def main(argv: list[str] | None = None) -> int:
    """Run the bootstrap write, live resolution, count probe, CAS demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--updated-by", default="olimoon", help="operator name recorded at bootstrap"
    )
    args = parser.parse_args(argv)

    load_dotenv()
    doc = _pointer_document(args.updated_by)
    store = ObjectStore.from_env()

    _bootstrap(store, doc)

    resolved = _resolve(store)
    _demo_cas(resolved["pointer"])

    athena = _athena()
    prefix_rows, claim_rows, scanned = _count(athena, resolved)
    _check_counts(prefix_rows, claim_rows, scanned)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
