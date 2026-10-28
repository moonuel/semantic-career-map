"""Pure key builders and validators for the object store.

Keys implement the layout in ``docs/architecture/data.md``. These functions are
the single source of truth for well-formed keys: they are pure (no I/O) so they
can be unit-tested without a store, and every other module derives keys through
them rather than by string interpolation.

This module implements the legacy ``live`` and ``configs`` branches plus the
lakehouse ``lake`` branch. The lakehouse is being migrated branch-by-branch,
key builders first; the ``raw``, ``artifacts``, ``eval``, ``embeddings``, and
all ``manifests`` builders have now landed.

Lakehouse ``raw`` objects are **content-addressed**: identity is the SHA-256 of
the verbatim payload, not a posting id. ``raw_payload`` emits
``lake/raw/source=<source>/sha256=<hash>.json``, pruned by the low-cardinality
``source`` partition. ``source`` is validated against an allowlist
(:data:`RAW_SOURCES`) to keep partitioning governed. There is deliberately no
time partition: identical re-ingested payloads map to the same key and
overwrite idempotently, so each payload is stored once. Ingestion time is not
encoded in the key — objects are never rewritten, so S3 ``Last-Modified`` is
the first-ingestion date, and provenance dates for date-range queries belong
in batch manifests, not here.

Legacy posting ids are time-ordered **UUIDv7** values (RFC 9562 §5.7), rendered
in the canonical 36-char lowercase hyphenated form. Identity minting and
content hashing live in :mod:`backend.storage.ids`; this module only validates
and places id/hash tokens in keys. Because ids are ``[0-9a-f-]``, **every**
segment in a key is lowercase; there is no uppercase exception.

    Public builders:
    - ``raw_payload(source, content_hash)`` -> ``lake/raw/source=<source>/sha256=<hash>.json``
    - ``manifest_ingestion(ingested_on, ingestion_id)`` ->
      ``lake/manifests/ingestion/ingested_on=<date>/ingestion_id=<uuid7>.parquet``
    - ``artifact_part(type, variant, version, run_id, seq)`` ->
      ``lake/artifacts/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``
    - ``eval_dataset(type, variant, version, run_id, seq)`` ->
      ``lake/eval/datasets/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``
    - ``eval_result(type, variant, version, run_id, seq)`` ->
      ``lake/eval/results/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``
    - ``embedding_part(model, variant, version, run_id, seq)`` ->
      ``lake/embeddings/model=<model>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``
    - ``manifest_dataset(dataset_id)``     -> ``lake/manifests/datasets/<dataset-id>.json``
    - ``manifest_run(run_id)``             -> ``lake/manifests/runs/run_id=<run-id>/run.json``
    - ``manifest_run_checkpoint(run_id, seq)`` ->
      ``lake/manifests/runs/run_id=<run-id>/flush=<seq>.json``
    - ``raw_posting(id)``        -> ``live/raw/<id>/source.md``
    - ``raw_metadata(id)``       -> ``live/raw/<id>/metadata.json``
    - ``derived(id, variant, v)``-> ``live/derived/<id>/<variant>/v<v>.json``
    - ``embedding(model, variant, v, id)`` -> ``live/embeddings/<model>/<variant>/v<v>/<id>.npy``
    - ``config(hash)``           -> ``configs/<hash>.json``

    Validators:
    - ``validate_key(key)`` raises ``ValueError`` on a malformed key.
"""

from __future__ import annotations

import datetime
import re

from backend.storage.ids import is_canonical_uuid7, is_sha256_hex

# --- Constants ---------------------------------------------------------------

# The single bucket's top-level branches. ``lake`` holds the lakehouse raw
# layer (content-addressed, immutable); ``live`` and ``configs`` are the legacy
# object-per-posting store still in use until each branch migrates.
_BUCKETS: frozenset[str] = frozenset({"eval", "live", "configs", "lake"})

# Sources admitted into ``lake/raw/source=<source>/``. Kept as an explicit
# allowlist so partitioning stays governed — a source is a partition column,
# and an unbounded source set is the partition-explosion / ungoverned-accumulation
# anti-pattern. Sources are lowercase and must satisfy _SEGMENT_RE.
#
# ``hiringcafe`` is the query-driven MCP source
# (``docs/specs/hiringcafe-mcp-query-integration.md``): provenance-poor by the
# protocol's design (no ATS/board join keys — the MCP result strips employer
# URLs and ATS identifiers), so its partition is standalone; its postings join
# to nothing outside ``source=hiringcafe``.
RAW_SOURCES: frozenset[str] = frozenset(
    {"linkedin", "lever", "greenhouse", "ashby", "dataroles", "hiringcafe"}
)

# Variants that live under live/derived/<id>/<variant>/.
# Kept open: new variants are added here and are additive (layout §5).
_DERIVED_VARIANTS: frozenset[str] = frozenset(
    {"sections", "clean-text", "llm-clean-text", "llm-role-context", "labels"}
)

# Artifact types admitted into ``lake/artifacts/type=<type>/``. Same reasoning
# as RAW_SOURCES: ``type`` is a partition column and an unbounded set is the
# partition-explosion / ungoverned-accumulation anti-pattern. Additive.
ARTIFACT_TYPES: frozenset[str] = frozenset({"cleaned_text"})

# Eval dataset types admitted into ``lake/eval/datasets|results/type=<type>/``.
# The ``eval`` branch keeps authored ground truth (``datasets``) separate from
# machine-written evaluation output (``results``) — the former eval/ vs live/
# split — but both share one governed type partition. Additive.
EVAL_TYPES: frozenset[str] = frozenset({"cleaning_golden", "cleaning_quality"})

# Regex for a well-formed non-id path segment: lowercase, digits, hyphens, `_`
# (for `config-hash`), and `=` for `key=value` partition pairs
# (e.g. `day=2026-09-23`, Report 005 §3.2). No leading/trailing hyphen.
_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*$")

# Regex for a well-formed terminal (filename) segment: a bare name or a
# `name.ext`, where the name follows the segment rules and the extension is
# lowercase alphanumeric. Dots are only allowed here, at the leaf.
_FILENAME_RE = re.compile(r"^[a-z0-9]+(?:[-_=][a-z0-9]+)*(?:\.[a-z0-9]+)?$")

# Regex for a plain name segment: segment rules minus `=` — no Hive partition
# pair. Used where a k=v value would nest a second `=` into an object key
# (e.g. a dataset id that is itself the addressed leaf).
_PLAIN_SEGMENT_RE = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")

# Version token: `v` followed by a positive integer. A stable token names a
# claimable generation: one configuration owns it, and it is the address a
# consumer means when it says "this dataset".
_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")

# Ephemeral version token: `x-` then a lowercase name, dots allowed
# (`x-v1.1`, `x-scratch-run3`). An ephemeral prefix is a throwaway, claimed
# only against concurrent writers — it never names production data and is
# never projected into the query layer's partition enum.
_EPHEMERAL_VERSION_RE = re.compile(r"^x-[a-z0-9]+(?:[._-][a-z0-9]+)*$")

# A `version=` partition segment, dots included. Structural only: each branch's
# shape pinner still decides which version vocabulary it accepts.
_VERSION_PARTITION_RE = re.compile(r"^version=[a-z0-9]+(?:[-_.][a-z0-9]+)*$")


# --- Helpers ----------------------------------------------------------------


def _check(segment: str, *, name: str) -> str:
    """Return ``segment`` if it is a well-formed non-id key segment, else raise."""
    if not segment:
        raise ValueError(f"{name} must be non-empty")
    if _SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(
            f"{name} {segment!r} is not a valid key segment "
            f"(lowercase letters, digits, '=', hyphens, underscores only)"
        )
    return segment


def _check_id(id: str, *, name: str = "posting id") -> str:
    """Return ``id`` if it is a well-formed posting id, else raise.

    A posting id is a canonical UUIDv7 in 36-char lowercase hyphenated form
    (``8-4-4-4-12``). Unlike every other segment, hyphens sit at fixed
    positions, so this dedicated shape check is used rather than
    :data:`_SEGMENT_RE`.
    """
    if not is_canonical_uuid7(id):
        raise ValueError(
            f"{name} {id!r} must be a canonical UUIDv7 (8-4-4-4-12), "
            f"e.g. '0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70'"
        )
    return id


def _check_version(
    version: str, *, name: str = "version", allow_ephemeral: bool = False
) -> str:
    """Return ``version`` if a stable token, or an allowed ephemeral token.

    Args:
        version: The version token to check.
        name: Label used in the error message.
        allow_ephemeral: Accept ``x-...`` throwaway tokens beside ``vN``.
            Default is stable-only, so widening the vocabulary for embeddings
            cannot leak into the artifact or eval branches.

    Raises:
        ValueError: if ``version`` is neither form.
    """
    if _VERSION_RE.fullmatch(version) is not None:
        return version
    if allow_ephemeral and _EPHEMERAL_VERSION_RE.fullmatch(version) is not None:
        return version
    expected = (
        "'vN' (v1, v2, ...) or an ephemeral 'x-...' token"
        if allow_ephemeral
        else "'vN' (v1, v2, ...)"
    )
    raise ValueError(f"{name} must be {expected}, got {version!r}")


def version_kind(version: str) -> str:
    """Classify a version token as ``"stable"`` or ``"ephemeral"``.

    Stable tokens (``vN``) are the claimable, production-addressable
    generations. Ephemeral tokens (``x-...``) are throwaway prefixes: still
    claimed, so two runs cannot interleave parts, but never projected into the
    query layer and never a production address.

    Args:
        version: The version token to classify.

    Returns:
        ``"stable"`` or ``"ephemeral"``.

    Raises:
        ValueError: if the token is neither form.
    """
    if _VERSION_RE.fullmatch(version) is not None:
        return "stable"
    if _EPHEMERAL_VERSION_RE.fullmatch(version) is not None:
        return "ephemeral"
    raise ValueError(
        f"version must be 'vN' (v1, v2, ...) or an ephemeral 'x-...' token, "
        f"got {version!r}"
    )


def _check_hash(hash_: str, *, name: str = "config hash") -> str:
    if not is_sha256_hex(hash_):
        raise ValueError(f"{name} must be a 64-char hex string, got {hash_!r}")
    return hash_


def _check_plain(segment: str, *, name: str) -> str:
    """Return ``segment`` if it is a well-formed plain (non-partition) segment.

    A plain segment is a :func:`_check` segment without ``=``: it must never
    read as a Hive ``k=v`` pair when placed at a key position that is not a
    partition.
    """
    if not segment:
        raise ValueError(f"{name} must be non-empty")
    if _PLAIN_SEGMENT_RE.fullmatch(segment) is None:
        raise ValueError(
            f"{name} {segment!r} is not a valid plain key segment "
            f"(lowercase letters, digits, hyphens, underscores only)"
        )
    return segment


def _check_seq(seq: int, *, name: str = "sequence") -> int:
    """Return ``seq`` if a positive 1-based sequence number, else raise."""
    if seq < 1:
        raise ValueError(f"{name} must be a 1-based sequence number, got {seq!r}")
    return seq


def _part_leaf(run_id: str, seq: int) -> str:
    """Render the run-minted Parquet part leaf: ``part-<run-id>-<seq>.parquet``.

    Every Parquet part in a dataset version names its producing run (its
    ``run_id``) and a 1-based flush sequence, zero-padded to 5 digits so
    lexicographic and numeric order agree within a run (sibling of the
    ``part-00000`` example in data.md §1). The flushing single writer commits
    one part per flush, so parts belong to a run, not to nothing.
    """
    return f"part-{run_id}-{seq:05d}.parquet"


def _flush_leaf(seq: int) -> str:
    """Render the flush-checkpoint leaf: ``flush=<seq>.json`` (5-wide-padded)."""
    return f"flush={seq:05d}.json"


def _check_date(value: str, *, name: str) -> str:
    """Return ``value`` if a canonical ``YYYY-MM-DD`` calendar date, else raise.

    The string form must be zero-padded and lexically sortable (a Hive
    partition value doubles as a range-query filter), and must name a real
    calendar day. ``datetime.date.fromisoformat`` would accept non-canonical
    forms, so the strict shape is pinned by regex first.
    """
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError(f"{name} {value!r} must be a canonical YYYY-MM-DD date")
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} {value!r} is not a calendar date") from None
    return value


# --- Builders ---------------------------------------------------------------


def raw_payload(source: str, content_hash: str) -> str:
    """Key for a verbatim, content-addressed raw payload.

    ``lake/raw/source=<source>/sha256=<content-hash>.json``

    The raw object is addressed by the SHA-256 of the received payload, not by
    a posting id (layout §2.1). ``source`` is a low-cardinality Hive partition
    pruned by Athena. The key carries no time partition: identical re-ingested
    payloads produce the same key, so re-runs overwrite idempotently and each
    payload is stored once. Ingestion time is observable as the object's S3
    ``Last-Modified`` (objects are never rewritten) and lands in batch
    manifests for Athena range queries; it is not encoded here.

    Args:
        source: Origin of the payload; must be in :data:`RAW_SOURCES`.
        content_hash: 64-char lowercase hex SHA-256 of the verbatim bytes — the
            content address.

    Returns:
        The well-formed raw object key.

    Raises:
        ValueError: if ``source`` is not allowlisted or ``content_hash`` is
            not 64-char hex.
    """
    # Allowlist check first so an unknown source gets the useful "expected one
    # of ..." message, not the generic segment error.
    if source not in RAW_SOURCES:
        raise ValueError(
            f"unknown raw source {source!r}; expected one of {sorted(RAW_SOURCES)}"
        )
    source = _check(source, name="source")
    return (
        f"lake/raw/source={source}/"
        f"sha256={_check_hash(content_hash, name='content hash')}.json"
    )


def manifest_ingestion(ingested_on: str, ingestion_id: str) -> str:
    """Key for a batch-scrape ingestion manifest (one Parquet file per run).

    ``lake/manifests/ingestion/ingested_on=<ingested-on>/ingestion_id=<id>.parquet``

    The ingestion event is an append-only Parquet row batch keyed by the run's
    ``ingestion_id`` (layout §2.1, §3): one object per batched scrape — not one
    per event — keeps the atomic-file count at scale down while still recording
    exactly what each run observed and stored. The ``ingested_on`` day partition
    lives *only* here (raw keys deliberately carry no date, §2.1); manifests are
    where batch provenance time is pruned for Athena range queries. Manifests
    are never rewritten, so a re-deduped re-run under the same id is undefined —
    the id is minted once per run.

    Args:
        ingested_on: Canonical ``YYYY-MM-DD`` run date.
        ingestion_id: Canonical UUIDv7 minted once for the batch run.

    Returns:
        The well-formed ingestion-manifest object key.

    Raises:
        ValueError: if the date is not canonical ``YYYY-MM-DD`` or the id is
            not a canonical UUIDv7.
    """
    return (
        f"lake/manifests/ingestion/ingested_on={_check_date(ingested_on, name='ingested_on')}/"
        f"ingestion_id={_check_id(ingestion_id, name='ingestion id')}.parquet"
    )


def artifact_part(
    artifact_type: str, variant: str, version: str, run_id: str, seq: int
) -> str:
    """Key for one Parquet part of a derived artifact dataset version.

    ``lake/artifacts/type=<artifact-type>/variant=<variant>/version=<version>/part-<run-id>-<seq>.parquet``

    A dataset version is the set of Parquet parts under its ``type`` /
    ``variant`` / ``version`` partitions (layout §1). ``type`` is a governed,
    additive partition column (:data:`ARTIFACT_TYPES`); ``variant`` names the
    producing method and stays open — method/prompt iterations add variants,
    while semantic revisions add a new ``version`` (§2.4).

    Args:
        artifact_type: Artifact family; must be in :data:`ARTIFACT_TYPES`
            (e.g. ``cleaned_text``).
        variant: Producing method (e.g. ``llm``).
        version: Version token ``vN``.
        run_id: Canonical UUIDv7 of the pipeline run that wrote this part.
        seq: 1-based flush sequence within the run.

    Returns:
        The well-formed artifact part key.

    Raises:
        ValueError: if any segment is malformed or ``artifact_type`` is not
            allowlisted.
    """
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(
            f"unknown artifact type {artifact_type!r}; "
            f"expected one of {sorted(ARTIFACT_TYPES)}"
        )
    artifact_type = _check(artifact_type, name="artifact type")
    return (
        f"lake/artifacts/type={artifact_type}/"
        f"variant={_check(variant, name='variant')}/"
        f"version={_check_version(version, name='artifact version')}/"
        f"{_part_leaf(_check_id(run_id, name='run id'), _check_seq(seq))}"
    )


def eval_dataset(
    eval_type: str, variant: str, version: str, run_id: str, seq: int
) -> str:
    """Key for one Parquet part of an authored eval ground-truth dataset.

    ``lake/eval/datasets/type=<type>/variant=<variant>/version=<version>/part-<run-id>-<seq>.parquet``

    Eval datasets are authored (annotated) datasets; eval *results* live under
    :func:`eval_result` — authored ground truth never shares a prefix with
    generated data (layout §1). ``eval_type`` is governed
    (:data:`EVAL_TYPES`); ``variant`` names the authoring method and stays open.

    Args:
        eval_type: Eval subject type; must be in :data:`EVAL_TYPES`
            (e.g. ``cleaning_golden``).
        variant: Authoring method (e.g. ``author``).
        version: Version token ``vN``.
        run_id: Canonical UUIDv7 of the run that wrote this part.
        seq: 1-based flush sequence within the run.

    Returns:
        The well-formed eval dataset part key.

    Raises:
        ValueError: if any segment is malformed or ``eval_type`` is not
            allowlisted.
    """
    if eval_type not in EVAL_TYPES:
        raise ValueError(
            f"unknown eval type {eval_type!r}; expected one of {sorted(EVAL_TYPES)}"
        )
    return (
        f"lake/eval/datasets/type={_check(eval_type, name='eval type')}/"
        f"variant={_check(variant, name='variant')}/"
        f"version={_check_version(version, name='eval version')}/"
        f"{_part_leaf(_check_id(run_id, name='run id'), _check_seq(seq))}"
    )


def eval_result(
    eval_type: str, variant: str, version: str, run_id: str, seq: int
) -> str:
    """Key for one Parquet part of an eval-results dataset over a dataset.

    ``lake/eval/results/type=<type>/variant=<variant>/version=<version>/part-<run-id>-<seq>.parquet``

    Machine output over an eval dataset, regenerated per evaluation
    configuration; versions add, they never overwrite (layout §1, §8).

    Args:
        eval_type: Eval subject type; must be in :data:`EVAL_TYPES`
            (e.g. ``cleaning_quality``).
        variant: Evaluation variant (e.g. the metric rubric name).
        version: Version token ``vN``.
        run_id: Canonical UUIDv7 of the evaluation run that wrote this part.
        seq: 1-based flush sequence within the run.

    Returns:
        The well-formed eval-result part key.

    Raises:
        ValueError: if any segment is malformed or ``eval_type`` is not
            allowlisted.
    """
    if eval_type not in EVAL_TYPES:
        raise ValueError(
            f"unknown eval type {eval_type!r}; expected one of {sorted(EVAL_TYPES)}"
        )
    return (
        f"lake/eval/results/type={_check(eval_type, name='eval type')}/"
        f"variant={_check(variant, name='variant')}/"
        f"version={_check_version(version, name='eval version')}/"
        f"{_part_leaf(_check_id(run_id, name='run id'), _check_seq(seq))}"
    )


def embedding_part(
    model: str, variant: str, version: str, run_id: str, seq: int
) -> str:
    """Key for one Parquet part of an embedding dataset version.

    ``lake/embeddings/model=<model>/variant=<variant>/version=<version>/part-<run-id>-<seq>.parquet``

    Embeddings are grouped model-first (data.md §7): a vector is a row in a
    model's corpus space, so the ``model`` partition leads. ``model`` names the
    model family segment (lowercased, slash-free; the full model id rides in
    each row's ``model`` column) — ``variant`` names the producing method and
    stays open. Like every dataset version, versions add and never overwrite:
    a configuration change re-embeds into a new ``version`` prefix rather than
    rewriting vector space. The run-minted part leaf names the producing run
    and its 1-based flush sequence (see :func:`_part_leaf`).

    The three levels are declared as Hive partition keys on the registered
    table (``provision_query_layer.EMBEDDINGS_PARTITION_KEYS``), so keeping
    this layout stable is what makes them queryable — a segment rename here
    is a catalog change, not just a path change. The partition key for the
    leading segment is named ``model_segment`` because the row already owns
    a ``model`` column.

    Args:
        model: Model family segment (e.g. ``all-minilm-l6-v2``).
        variant: Producing method segment (e.g. ``minilm``).
        version: Version token ``vN``, or an ephemeral ``x-...`` token.
        run_id: Canonical UUIDv7 of the pipeline run that wrote this part.
        seq: 1-based flush sequence within the run.

    Returns:
        The well-formed embedding part key.

    Raises:
        ValueError: if any segment is malformed.
    """
    return (
        f"lake/embeddings/model={_check(model, name='model')}/"
        f"variant={_check(variant, name='variant')}/"
        f"version={_check_version(version, name='embedding version', allow_ephemeral=True)}/"
        f"{_part_leaf(_check_id(run_id, name='run id'), _check_seq(seq))}"
    )


def embedding_version_claim(model: str, variant: str, version: str) -> str:
    """Key for the registry object recording who owns an embedding version.

    ``lake/manifests/versions/embeddings/<model>/<variant>/<version>.json``

    One JSON object per version prefix, written once with a conditional put
    (``ObjectStore.put_if_absent``) before a run writes any part. It answers
    the question the pipeline could not answer before: *which configuration
    owns this prefix?* Two different ``configuration_hash`` values can no
    longer mint parts into one prefix, which is the defect behind the 400
    foreign rows in ``version=v1`` (issue #22).

    The leaf is the version token with dots replaced by hyphens, keeping it a
    plain single-extension filename (``x-v1.1`` → ``x-v1-1.json``). The JSON
    body carries the token verbatim, so the lossy leaf stays unambiguous.

    Args:
        model: Model family segment (e.g. ``all-minilm-l6-v2``).
        variant: Producing method segment (e.g. ``minilm``).
        version: Version token ``vN`` or an ephemeral ``x-...`` token.

    Returns:
        The well-formed version-claim object key.

    Raises:
        ValueError: if any segment, or the token, is malformed.
    """
    token = _check_version(version, name="embedding version", allow_ephemeral=True)
    leaf = token.replace(".", "-")
    return (
        f"lake/manifests/versions/embeddings/{_check(model, name='model')}/"
        f"{_check(variant, name='variant')}/{_check_plain(leaf, name='version')}.json"
    )


def parse_part_leaf(leaf: str) -> tuple[str, int]:
    """Parse a run-minted part leaf back into ``(run_id, seq)``.

    The complement of :func:`_part_leaf`: reconciliation and healing code
    groups stored parts by producing run, and the run id lives in the leaf.
    One parser keeps the round-trip in lockstep with the builder (and with
    :func:`_check_part_leaf`'s own validation).

    Args:
        leaf: The ``part-<run-id>-<seq>.parquet`` leaf of a dataset part key.

    Returns:
        ``(run_id, seq)`` as minted.

    Raises:
        ValueError: if the leaf is not a run-minted part leaf.
    """
    if not leaf.endswith(".parquet"):
        raise ValueError(f"part leaf must end in '.parquet', got {leaf!r}")
    base = leaf.removesuffix(".parquet")
    if not base.startswith("part-"):
        raise ValueError(f"part leaf must start with 'part-', got {leaf!r}")
    run_id, sep, seq_digits = base.removeprefix("part-").rpartition("-")
    if not sep:
        raise ValueError(f"part leaf must carry '-<seq>', got {leaf!r}")
    _check_id(run_id, name="run id")
    if not seq_digits.isdigit() or int(seq_digits) < 1:
        raise ValueError(
            f"part leaf sequence must be a positive integer, got {seq_digits!r}"
        )
    return run_id, int(seq_digits)


def manifest_dataset(dataset_id: str) -> str:
    """Key for a dataset manifest (one JSON object per dataset generation).

    ``lake/manifests/datasets/<dataset-id>.json``

    The ``dataset_id`` is the leaf address itself          — high-cardinality, and data.md §2.3 bans partitioning on identifiers because an address identifies an
    object while a partition prunes a scan. Plain-name (no ``k=v``) dataset ids
    keep the leaf from reading as a Hive partition.

    Args:
        dataset_id: Plain dataset id without extension (e.g.
            ``cleaned_text-v1``).

    Returns:
        The well-formed dataset-manifest object key.

    Raises:
        ValueError: if ``dataset_id`` is empty or not a plain name.
    """
    return f"lake/manifests/datasets/{_check_plain(dataset_id, name='dataset id')}.json"


def manifest_run(run_id: str) -> str:
    """Key for an aggregated pipeline-run record.

    ``lake/manifests/runs/run_id=<run-id>/run.json``

    The run lives under its ``run_id`` grouping segment alongside the flush
    checkpoints it aggregates (:func:`manifest_run_checkpoint`), per the
    grouping layout of layout §1 ``manifests/runs/``.

    Args:
        run_id: Canonical UUIDv7 minted once for the run.

    Returns:
        The well-formed run-record object key.

    Raises:
        ValueError: if ``run_id`` is not a canonical UUIDv7.
    """
    return f"lake/manifests/runs/run_id={_check_id(run_id, name='run id')}/run.json"


def manifest_run_checkpoint(run_id: str, seq: int) -> str:
    """Key for a single flush checkpoint of a pipeline run.

    ``lake/manifests/runs/run_id=<run-id>/flush=<seq>.json``

    One checkpoint per committed Parquet part, written *after* its part
    (part → checkpoint commit order). ``seq`` is 1-based and 5-wide-padded so
    checkpoints enumerate in flush order by prefix listing.

    Args:
        run_id: Canonical UUIDv7 minted once for the run.
        seq: 1-based flush sequence within the run.

    Returns:
        The well-formed flush-checkpoint object key.

    Raises:
        ValueError: if ``run_id`` is not a canonical UUIDv7 or ``seq`` is
            not positive.
    """
    return (
        f"lake/manifests/runs/run_id={_check_id(run_id, name='run id')}/"
        f"{_flush_leaf(_check_seq(seq))}"
    )


def raw_posting(id: str) -> str:
    """Key for the verbatim source markdown of a posting.

    ``live/raw/<id>/source.md``
    """
    return f"live/raw/{_check_id(id)}/source.md"


def raw_metadata(id: str) -> str:
    """Key for the ingestion sidecar describing a posting.

    ``live/raw/<id>/metadata.json``
    """
    return f"live/raw/{_check_id(id)}/metadata.json"


def derived(id: str, variant: str, version: str = "v1") -> str:
    """Key for a derived artifact of a posting.

    ``live/derived/<id>/<variant>/v<version>.json``

    ``variant`` must be one of :data:`_DERIVED_VARIANTS`.
    """
    _check_id(id)
    if variant not in _DERIVED_VARIANTS:
        raise ValueError(
            f"unknown derived variant {variant!r}; expected one of "
            f"{sorted(_DERIVED_VARIANTS)}"
        )
    return (
        f"live/derived/{id}/"
        f"{_check(variant, name='variant')}/{_check_version(version)}.json"
    )


def embedding(model: str, variant: str, version: str, id: str) -> str:
    """Key for a posting's embedding vector.

    ``live/embeddings/<model>/<variant>/v<version>/<id>.npy``

    The model leads because embeddings are queried model-first: a vector is a
    row in a model's corpus space, so ``id`` sits at the leaf as a join handle
    back to the source (layout §2.1).
    """
    return (
        f"live/embeddings/{_check(model, name='model')}/"
        f"{_check(variant, name='variant')}/"
        f"{_check_version(version)}/{_check_id(id)}.npy"
    )


def config(hash_: str) -> str:
    """Key for a config manifest entry.

    ``configs/<hash>.json``

    ``hash_`` is the content hash of the config object; the manifest turns a
    version token into reproducible generation parameters (layout §2.3).
    """
    return f"configs/{_check_hash(hash_)}.json"


# --- Validators -------------------------------------------------------------


def _id_index(bucket: str, rest: list[str]) -> int | None:
    """Index in ``rest`` of the posting-id segment for this key shape, else None.

    The id is identified *positionally* by the branch, not by its characters:

    - ``live/raw/<id>/<file>``                 -> index 1
    - ``live/derived/<id>/<variant>/<version-file>`` -> index 1
    - ``live/embeddings/<model>/<variant>/<vN>/<id>.<ext>`` -> leaf (index ``-1``)
    - ``lake/raw/...`` (content-addressed)     -> no id segment
    - ``lake/artifacts``/``lake/eval``/``lake/manifests`` -> no id segment
      (run ids ride inside part/checkpoint leaves and the ``run_id=<id>/``
      grouping segment, validated by each branch's shape pinner)
    - ``meta/...`` and ``configs/...``         -> no id segment
    """
    if not rest:
        return None
    if bucket == "live" and rest[0] in {"raw", "derived"}:
        return 1 if len(rest) > 1 else None
    if bucket == "live" and rest[0] == "embeddings":
        return len(rest) - 1
    return None


def _validate_lake_manifest(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/manifests/ingestion`` key shape.

    ``lake/manifests/ingestion/ingested_on=<date>/ingestion_id=<uuid>.parquet``

    Pins the branch, the single ``ingested_on`` day partition, and the
    ``ingestion_id=<uuid7>.parquet`` leaf for the same reason
    :func:`_validate_lake_raw` pins the raw shape. The other ``manifests``
    sub-branches (``datasets``, ``runs``) are pinned in
    :func:`_validate_dataset_manifest` / :func:`_validate_runs_manifest`.
    """
    if len(rest) != 4 or rest[1] != "ingestion":
        raise ValueError(
            f"key {key!r} must be lake/manifests/ingestion/"
            f"ingested_on=<date>/ingestion_id=<id>.parquet"
        )
    partition = rest[2].partition("=")
    if partition[0] != "ingested_on":
        raise ValueError(
            f"key {key!r} must partition exactly on ingested_on, got {rest[2]!r}"
        )
    _check_date(partition[2], name="ingested_on")
    leaf_name, _, leaf_value = rest[-1].partition("=")
    if leaf_name != "ingestion_id" or not leaf_value.endswith(".parquet"):
        raise ValueError(
            f"key {key!r} must end in 'ingestion_id=<id>.parquet', got {rest[-1]!r}"
        )
    _check_id(leaf_value.removesuffix(".parquet"), name="ingestion id")


def _check_part_leaf(key: str, leaf: str) -> None:
    """Pin a run-minted Parquet part leaf: ``part-<run-id>-<seq>.parquet``.

    The complement of :func:`_part_leaf`: parses the leaf back, validating the
    run id and the 1-based sequence, so ``validate_key`` gates externally
    supplied part keys that a builder did not mint.
    """
    if not leaf.endswith(".parquet"):
        raise ValueError(
            f"key {key!r} must end in 'part-<run-id>-<seq>.parquet', got {leaf!r}"
        )
    base = leaf.removesuffix(".parquet")
    if not base.startswith("part-"):
        raise ValueError(
            f"key {key!r} part leaf must be named 'part-<run-id>-<seq>.parquet', "
            f"got {leaf!r}"
        )
    run_id, sep, seq_digits = base.removeprefix("part-").rpartition("-")
    if not sep:
        raise ValueError(
            f"key {key!r} must end in 'part-<run-id>-<seq>.parquet', got {leaf!r}"
        )
    _check_id(run_id, name="run id")
    if not seq_digits.isdigit() or int(seq_digits) < 1:
        raise ValueError(
            f"key {key!r} part leaf sequence must be a positive integer, got {leaf!r}"
        )


def _validated_partitions(
    key: str, segments: list[str], expected_names: list[str]
) -> list[str]:
    """Return partition values if ``segments`` are Hive ``k=v`` pairs in order.

    The partition names must match ``expected_names`` exactly — order and
    count included — so a dataset version's directory levels cannot flip or
    grow silently.
    """
    partitions = [segment.partition("=") for segment in segments]
    names = [name for name, _, _ in partitions]
    if names != expected_names:
        raise ValueError(
            f"key {key!r} must partition exactly on "
            f"{'/'.join(expected_names)}, got {names}"
        )
    return [value for _, _, value in partitions]


def _validate_lake_artifact(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/artifacts`` key shape.

    ``lake/artifacts/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``

    Pins the branch, the three Hive partitions in order, the allowlisted
    ``type``, the ``vN`` version, and the run-minted part leaf.
    """
    if len(rest) != 5 or rest[0] != "artifacts":
        raise ValueError(
            f"key {key!r} must be lake/artifacts/type=<type>/variant=<variant>"
            f"/version=<vN>/part-<run-id>-<seq>.parquet"
        )
    values = _validated_partitions(key, rest[1:4], ["type", "variant", "version"])
    if values[0] not in ARTIFACT_TYPES:
        raise ValueError(
            f"key {key!r} has unknown artifact type {values[0]!r}; "
            f"expected one of {sorted(ARTIFACT_TYPES)}"
        )
    _check_version(values[2], name="artifact version")
    _check_part_leaf(key, rest[-1])


def _validate_lake_eval(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/eval`` key shape.

    ``lake/eval/<datasets|results>/type=<type>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``

    Pins the branch, the ``datasets``/``results`` sub-branch (authored ground
    truth vs machine output), the three Hive partitions in order, the
    allowlisted ``type``, the ``vN`` version, and the part leaf.
    """
    if len(rest) != 6 or rest[0] != "eval" or rest[1] not in {"datasets", "results"}:
        raise ValueError(
            f"key {key!r} must be lake/eval/<datasets|results>/type=<type>"
            f"/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet"
        )
    values = _validated_partitions(key, rest[2:5], ["type", "variant", "version"])
    if values[0] not in EVAL_TYPES:
        raise ValueError(
            f"key {key!r} has unknown eval type {values[0]!r}; "
            f"expected one of {sorted(EVAL_TYPES)}"
        )
    _check_version(values[2], name="eval version")
    _check_part_leaf(key, rest[-1])


def _validate_lake_embeddings(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/embeddings`` key shape.

    ``lake/embeddings/model=<model>/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet``

    Pins the branch, the three Hive partitions in order (model-first per
    data.md §7), the version token, and the run-minted part leaf — the branch
    is no longer merely reserved; its positional layout is enforced like the
    other dataset branches.

    Embedding versions accept ephemeral ``x-...`` tokens beside ``vN``: an
    experimental generation writes into its own throwaway prefix instead of
    sharing the production one (issue #22). The claim registry
    (:func:`embedding_version_claim`) is what keeps two configurations from
    ever sharing a prefix.
    """
    if len(rest) != 5 or rest[0] != "embeddings":
        raise ValueError(
            f"key {key!r} must be lake/embeddings/model=<model>"
            f"/variant=<variant>/version=<vN>/part-<run-id>-<seq>.parquet"
        )
    values = _validated_partitions(key, rest[1:4], ["model", "variant", "version"])
    _check_version(values[2], name="embedding version", allow_ephemeral=True)
    _check_part_leaf(key, rest[-1])


def _validate_dataset_manifest(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/manifests/datasets`` key shape.

    ``lake/manifests/datasets/<dataset-id>.json``

    One addressed JSON object per dataset generation; no Hive partitions (the
    dataset id is an address, not a scan-pruning dimension).
    """
    if len(rest) != 3 or rest[1] != "datasets":
        raise ValueError(
            f"key {key!r} must be lake/manifests/datasets/<dataset-id>.json"
        )
    leaf = rest[-1]
    if not leaf.endswith(".json"):
        raise ValueError(f"key {key!r} must end in '<dataset-id>.json', got {leaf!r}")
    _check_plain(
        leaf.removesuffix(".json"),
        name="dataset id",
    )


def _validate_runs_manifest(key: str, rest: list[str]) -> None:
    """Semantic checks for the ``lake/manifests/runs`` key shape.

    ``lake/manifests/runs/run_id=<run-id>/run.json`` and
    ``lake/manifests/runs/run_id=<run-id>/flush=<seq>.json``

    Runs are grouped under their ``run_id`` segment: the aggregated run record
    (``run.json``) and its per-flush checkpoints (``flush=<seq>.json``) are
    siblings under the run they document. ``seq`` must parse as a positive
    integer.
    """
    if len(rest) != 4 or rest[1] != "runs":
        raise ValueError(
            f"key {key!r} must be lake/manifests/runs/run_id=<run-id>/"
            f"[run.json | flush=<seq>.json]"
        )
    run_partition = rest[2].partition("=")
    if run_partition[0] != "run_id":
        raise ValueError(f"key {key!r} must group exactly on run_id, got {rest[2]!r}")
    _check_id(run_partition[2], name="run id")
    leaf = rest[-1]
    if leaf == "run.json":
        return
    if leaf.startswith("flush=") and leaf.endswith(".json"):
        seq_value = leaf.removeprefix("flush=").removesuffix(".json")
        if seq_value.isdigit() and int(seq_value) >= 1:
            return
        raise ValueError(
            f"key {key!r} flush sequence must be a positive integer, got {leaf!r}"
        )
    raise ValueError(
        f"key {key!r} must end in 'run.json' or 'flush=<seq>.json', got {leaf!r}"
    )


def _validate_lake_raw(key: str, rest: list[str]) -> None:
    """Semantic checks for the content-addressed ``lake/raw`` key shape.

    ``lake/raw/source=<source>/sha256=<hash>.json``

    The safe-character loop in :func:`validate_key` only pins the segment
    alphabet; this pins the branch shape itself — the single ``source``
    partition and the hash leaf — because ``validate_key`` also gates
    externally supplied keys that :func:`raw_payload` did not mint. Other
    ``lake/`` branches get the same treatment when their builders land (the
    ingestion-manifest branch is pinned in :func:`_validate_lake_manifest`).
    """
    if len(rest) != 3:
        raise ValueError(
            f"key {key!r} must be lake/raw/source=<source>/sha256=<hash>.json"
        )
    leaf = rest[-1]
    leaf_name, _, leaf_value = leaf.partition("=")
    if leaf_name != "sha256" or not leaf_value.endswith(".json"):
        raise ValueError(
            f"key {key!r} must end in 'sha256=<content-hash>.json', got {leaf!r}"
        )
    _check_hash(leaf_value.removesuffix(".json"), name="content hash")
    partitions = [segment.partition("=") for segment in rest[1:-1]]
    names = [name for name, _, _ in partitions]
    if names != ["source"]:
        raise ValueError(f"key {key!r} must partition exactly on source, got {names}")
    for name, _, value in partitions:
        if value not in RAW_SOURCES:
            raise ValueError(
                f"key {key!r} has unknown raw source {value!r}; "
                f"expected one of {sorted(RAW_SOURCES)}"
            )


def validate_key(key: str) -> str:
    """Validate a key and return it unchanged, or raise ``ValueError``.

    Enforces the structural rules of ``data.md``:
      - first segment is one of the known branches (``eval``/``live``/``configs``/``lake``);
      - every segment matches the safe-character rule (no ``..``, no trailing
        slash, no reserved characters);
      - the posting-id segment — where the layout places one — is a canonical
        UUIDv7 (36-char lowercase hyphenated), while every other segment stays
        lowercase;
      - ``lake/raw`` keys additionally pin the whole content-addressed shape
        (allowlisted ``source`` partition, ``sha256=<hash>.json`` leaf);
      - ``lake/artifacts``, ``lake/eval/<datasets|results>``, and
        ``lake/embeddings`` pin their ``type|model``/``variant``/``version``
        partition stacks and the run-minted ``part-<run-id>-<seq>.parquet``
        leaf;
      - ``lake/manifests`` pins each sub-branch's shape: batch ingestion
        manifests (``ingested_on`` partition, ``ingestion_id`` leaf), dataset
        manifests (addressed ``<dataset-id>.json`` leaf), and runs (grouped
        ``run_id`` segment with ``run.json`` / ``flush=<seq>.json`` leaves);
      - no trailing slash; no leading slash.

    The once-reserved ``lake/embeddings`` branch is now pinned the same way:
    its builder (:func:`embedding_part`) and shape validator have landed with
    the embedding-pipeline change.
    """
    if not key or key.startswith("/"):
        raise ValueError(f"key must not be empty or start with '/': {key!r}")
    if key.endswith("/"):
        raise ValueError(f"key must not end with '/': {key!r}")
    if "//" in key:
        raise ValueError(f"key must not contain empty segments: {key!r}")

    segments = key.split("/")
    bucket, rest = segments[0], segments[1:]

    if bucket not in _BUCKETS:
        raise ValueError(
            f"key {key!r} starts with unknown bucket {bucket!r}; "
            f"expected one of {sorted(_BUCKETS)}"
        )

    for segment in rest:
        if segment in {".", ".."}:
            raise ValueError(
                f"key {key!r} contains a path-normalization segment {segment!r}"
            )

    id_index = _id_index(bucket, rest)
    for index, segment in enumerate(rest):
        if index == id_index:
            # Embeddings place the id at the leaf as ``<id>.npy``; the raw and
            # derived branches place it as a bare segment.
            id_segment = segment
            if rest and rest[0] == "embeddings":
                id_segment, matched, extension = segment.partition(".")
                if not matched or extension != "npy":
                    raise ValueError(
                        f"key {key!r} must end in '<id>.npy', got {segment!r}"
                    )
            _check_id(id_segment)
            continue
        is_last = index == len(rest) - 1
        if is_last:
            pattern = _FILENAME_RE
        elif segment.startswith("version="):
            # Ephemeral tokens carry a dot (`version=x-v1.1`); each branch's
            # shape pinner still decides whether that vocabulary is allowed.
            pattern = _VERSION_PARTITION_RE
        else:
            pattern = _SEGMENT_RE
        if pattern.fullmatch(segment) is None:
            raise ValueError(
                f"key {key!r} contains malformed segment {segment!r} "
                f"(lowercase letters, digits, '=', hyphens, underscores; dots "
                f"only in the last segment)"
            )

    if bucket == "lake" and rest[:1] == ["raw"]:
        _validate_lake_raw(key, rest)

    if bucket == "lake" and rest[:1] == ["artifacts"]:
        _validate_lake_artifact(key, rest)

    if bucket == "lake" and rest[:1] == ["eval"]:
        _validate_lake_eval(key, rest)

    if bucket == "lake" and rest[:1] == ["embeddings"]:
        _validate_lake_embeddings(key, rest)

    if bucket == "lake" and rest[:2] == ["manifests", "ingestion"]:
        _validate_lake_manifest(key, rest)

    if bucket == "lake" and rest[:2] == ["manifests", "datasets"]:
        _validate_dataset_manifest(key, rest)

    if bucket == "lake" and rest[:2] == ["manifests", "runs"]:
        _validate_runs_manifest(key, rest)

    return key
