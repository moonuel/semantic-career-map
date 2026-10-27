"""Selection adapter for the embedding tier: SQL text + row shaping.

The embedding tier's selection mirrors the cleaning tier's: a shipped
anti-join SQL (this module beside its file in ``backend/pipeline/sql/``)
renders into driver-shape rows. ``select_unembedded`` selects cleaned
artifact rows that have no embedding row under the run's configuration —
the continuously-ingesting lake embeds deltas, never the corpus. A
zero-row render is the healthy no-op (the day-2 idempotence artifact).

Row shaping applies the text-variant fallback per row: the configured
variant column first, the fallback column when the primary is
missing/empty; a row with no usable text in either is counted empty, never
silently dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

SELECT_SQL_PATH = "select_unembedded.sql"

# The registered embeddings table the anti-join targets. The single source
# of truth — scripts.provision_query_layer re-exports this so a rename here
# cannot drift the query layer's registration.
EMBEDDINGS_TABLE = "postings_embeddings"

# Deterministic row cap for the delta selection (deployment bound; the
# mirror of the cleaning tier's cap — CPU re-encodes are cheap, but the
# scheduled embedding tier still processes a bounded first-N on retries).
DEFAULT_MAX_POSTINGS = 5000

# Hex SHA-256 shape check for incoming query rows (fail loudly at render
# time, not at run time).
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def select_unembedded_sql(*, max_postings: int | None = None) -> str:
    """Return the versioned delta-selection SQL, deployment-bounded.

    Args:
        max_postings: Deterministic row cap; ``None`` uses
            :data:`DEFAULT_MAX_POSTINGS`, ``0`` disables the bound.

    Returns:
        The bounded SQL text (placeholders ``:config``, ``:model``,
        ``:variant``, ``:version`` unbound).
    """
    max_postings = DEFAULT_MAX_POSTINGS if max_postings is None else max_postings
    inner = (
        (Path(__file__).parent.parent / "sql" / SELECT_SQL_PATH)
        .read_text(encoding="utf-8")
        .strip()
        .replace("{embeddings_table}", EMBEDDINGS_TABLE)
    )
    lines = [f"SELECT s.source_hash, s.variant_text\nFROM ({inner}) s"]
    if max_postings > 0:
        lines.append(f"ORDER BY s.source_hash\nLIMIT {max_postings}")
    else:
        lines.append("ORDER BY s.source_hash")
    return "\n".join(lines)


def bind_sql_params(sql: str, params: dict[str, str]) -> str:
    """Bind ``:name`` placeholders to quoted literals (the gate's shape).

    Args:
        sql: The parameterized SQL text.
        params: Placeholder names (no colon) to quoted string values.

    Returns:
        The fully bound SQL text.

    Raises:
        ValueError: if a placeholder remains unbound (a half-bound SQL must
            never reach Athena).
    """
    bound = sql
    for name, value in params.items():
        if "'" in value or "\\" in value or "--" in value:
            raise ValueError(
                f"refusing to bind placeholder :{name} to a value that could "
                "break out of a SQL literal"
            )
        bound = bound.replace(f":{name}", f"'{value}'")
    leftovers = re.findall(r"(?<!'):[A-Za-z_][A-Za-z_0-9]*", bound)
    if leftovers:
        raise ValueError(f"SQL still carries unbound placeholder(s): {leftovers}")
    return bound


def shape_rows(
    rows: list[list[str]],
    *,
    text_variant: str,
    fallback_text_variant: str | None = None,
) -> tuple[list[tuple[str, str]], int]:
    """Shape Athena result rows into ``(source_hash, text)`` pairs.

    The streamed result carries a header row (column labels) which is
    skipped defensively by label; it must therefore remain the first row.
    Row values are validated: ``source_hash`` must be 64-char hex. The
    fallback applies per row when the primary variant column is
    missing/empty.

    Args:
        rows: Result rows as cell-value lists, in order (the header row
            first).
        text_variant: Primary text column's header label.
        fallback_text_variant: Fallback column's header label, or ``None``.

    Returns:
        ``(usable_rows, empty_count)``: usable ``(source_hash, text)``
        pairs and the count of rows with no usable text in either column.

    Raises:
        ValueError: on a drifted header, a short row, or a non-hex
            ``source_hash``.
    """
    expected = ["source_hash", text_variant] + (
        [fallback_text_variant] if fallback_text_variant else []
    )
    usable: list[tuple[str, str]] = []
    empty = 0
    for index, row in enumerate(rows):
        if index == 0 and [cell.strip() for cell in row[: len(expected)]] == expected:
            continue
        if len(row) < 2:
            raise ValueError(f"query result row {row!r} has fewer cells than expected")
        source_hash, primary = row[0].strip(), row[1]
        if not _HASH_RE.fullmatch(source_hash):
            raise ValueError(f"query result row has bad source_hash {source_hash!r}")
        text = primary if primary and primary.strip() else None
        if text is None and fallback_text_variant is not None and len(row) > 2:
            fallback = row[2]
            if fallback and fallback.strip():
                text = fallback
        if text is None:
            empty += 1
            continue
        usable.append((source_hash, text))
    return usable, empty
