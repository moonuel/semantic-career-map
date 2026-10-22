"""Selection adapter: SQL-layer results become cleaning-run manifests (D12).

The middle piece of the D12 loop: the SQL layer selects (the
``select_uncleaned`` anti-join over the cataloged raw × artifacts tables),
this module renders the selection into the runner's pinned manifest format —
the exact bytes shape :mod:`backend.pipeline.sampling` produces — and
applies the failure ledger. The cleaning runner (task 3.7) consumes the
manifest by path and never learns whether a sampler or this adapter built
it.

Determinism contract (same as the sampler's): manifest bytes are canonical
JSON (sorted keys, no timestamps); rows are validated and sorted by
``(source, source_hash)`` before serialization; the only caller inputs are
the query-result rows, the run's ``configuration_hash``, and the store (read
only, for the failure ledger). Two renders of the same query result over the
same ledger are byte-identical.

Failure-ledger source: ``lake/manifests/runs/run_id=*/run.json`` documents
carrying a ``failed_source_hashes`` list (the schema contract task 3.6
writes; read defensively here — absent run records mean an empty ledger).
"""

from __future__ import annotations

import csv
import io
import json
import re
from hashlib import sha256
from typing import Any

from backend.pipeline import sampling
from backend.storage import keys
from backend.storage.raw import RawStore

# The selection query shipped beside this module; both live in the change
# whose runner consumes their product.
SELECT_SQL_PATH = "select_uncleaned.sql"

# Hex SHA-256 shape check for incoming query rows (defense vs a drifted
# table column: fail loudly at render time, not at run time).
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def select_uncleaned_sql() -> str:
    """Return the versioned selection query text.

    Returns:
        The SQL text of ``select_uncleaned.sql`` (parameterized on
        ``:config``).
    """
    from pathlib import Path

    return (Path(__file__).parent / "sql" / SELECT_SQL_PATH).read_text(encoding="utf-8")


def configuration_hash_for(model: str, system_prompt: str, user_template: str) -> str:
    """SHA-256 over the cleaning configuration (the ``:config`` parameter).

    Mirrors task 3.3's definition — the selection query is keyed on the same
    configuration digest the artifact rows carry — as a local helper so the
    selection flow can be exercised before 3.3 lands. 3.3's implementation is
    authoritative; this helper exists for the selection contract only.

    Args:
        model: Model identifier.
        system_prompt: System prompt text.
        user_template: User message template.

    Returns:
        64-char lowercase hex SHA-256 over ``model\\nsystem_prompt\\nuser_template``.
    """
    return sha256(f"{model}\n{system_prompt}\n{user_template}".encode()).hexdigest()


def parse_query_results(csv_text: str) -> list[dict[str, str]]:
    """Parse Athena's CSV output into validated selection rows.

    Expected header: ``source,source_hash,source_id`` (the query's SELECT
    list, in order). Rows failing validation raise — a drifted table schema
    must surface here, not downstream.

    Args:
        csv_text: The query-result CSV (header + rows).

    Returns:
        Row dicts keyed ``source`` / ``source_hash`` / ``source_id``.

    Raises:
        ValueError: on a missing header field, an unknown source, or a
            non-hex ``source_hash``.
    """
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("query result is empty (no header row)") from exc
    if [h.strip() for h in header[:3]] != ["source", "source_hash", "source_id"]:
        raise ValueError(
            f"unexpected query-result header {header!r}; expected "
            f"[source, source_hash, source_id]"
        )
    rows: list[dict[str, str]] = []
    for record in reader:
        if not record:
            continue
        source, source_hash = record[0].strip(), record[1].strip()
        if source not in keys.RAW_SOURCES:
            raise ValueError(f"query result row has unknown source {source!r}")
        if not _HASH_RE.fullmatch(source_hash):
            raise ValueError(
                f"query result row has bad source_hash {source_hash!r} "
                f"(expected 64-char hex)"
            )
        rows.append(
            {
                "source": source,
                "source_hash": source_hash,
                "source_id": record[2].strip() if len(record) > 2 else "",
            }
        )
    return rows


def failed_source_hashes(store: RawStore) -> set[str]:
    """Collect postings recorded failed by prior runs (the failure ledger).

    Reads every ``lake/manifests/runs/run_id=*/run.json`` defensively: run
    records are written by task 3.6 — absent, malformed, or ledger-less
    documents contribute nothing rather than raising, because the ledger
    filter must never be the thing that breaks a rerun.

    Args:
        store: The object store to read run records from.

    Returns:
        The set of failed postings' ``source_hash`` values.
    """
    failed: set[str] = set()
    prefix = "lake/manifests/runs/"
    for key in sorted(store.list_keys(prefix)):
        if not key.endswith("run.json"):
            continue
        try:
            record = json.loads(store.get(key).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        values = record.get("failed_source_hashes")
        if not isinstance(values, list):
            continue
        failed.update(v for v in values if isinstance(v, str) and _HASH_RE.fullmatch(v))
    return failed


def build_selection_manifest(
    rows: list[dict[str, str]],
    *,
    configuration_hash: str,
    store: RawStore | None = None,
    query_name: str = SELECT_SQL_PATH,
) -> bytes:
    """Render selection rows into the runner's canonical manifest bytes.

    Args:
        rows: Validated selection rows (see :func:`parse_query_results`).
        configuration_hash: The run's configuration digest; recorded in the
            manifest params (the selection is per-configuration).
        store: Read-only store for the failure ledger; ``None`` skips the
            ledger filter (fixture/CLI dry runs).
        query_name: Recorded lineage for the producing query.

    Returns:
        Canonical manifest bytes — :func:`sampling.serialize_manifest`
        format, so the runner consumes either producer's output.

    Raises:
        ValueError: if ``configuration_hash`` is not 64-char hex or a row is
            invalid.
    """
    if not _HASH_RE.fullmatch(configuration_hash):
        raise ValueError(
            f"configuration_hash {configuration_hash!r} must be 64-char hex"
        )
    ledger = failed_source_hashes(store) if store is not None else set()
    postings: list[dict[str, Any]] = []
    for row in rows:
        if row["source_hash"] in ledger:
            continue
        postings.append(
            {
                "source": row["source"],
                "source_hash": row["source_hash"],
                "source_id": row["source_id"],
                "key": keys.raw_payload(row["source"], row["source_hash"]),
            }
        )
    postings.sort(key=lambda p: (p["source"], p["source_hash"]))
    manifest = {
        "params": {
            "query": query_name,
            "configuration_hash": configuration_hash,
        },
        "postings": postings,
    }
    return sampling.serialize_manifest(manifest)
