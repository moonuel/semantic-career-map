"""Version-prefix claims: token vocabulary, claim key shape, and the
one-configuration-per-prefix write protocol."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backend.pipeline.embedding.config import EmbeddingConfig
from backend.pipeline.embedding.version_claim import (
    ClaimConflict,
    adopt_version_prefix,
    claim_bytes,
    claim_record,
    claim_version,
)
from backend.storage import keys

RUN_ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"
RUN_ID_B = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d71"
HASH_A = "a" * 64
HASH_B = "b" * 64
CLAIMED_AT = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "a" * 40,
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "text_variant": "variant_text",
        "version": "v1",
        "sql_text": "SELECT 1",
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


# --- Token vocabulary -------------------------------------------------------


@pytest.mark.parametrize("version", ["v1", "v2", "v10"])
def test_stable_tokens(version: str) -> None:
    assert keys.version_kind(version) == "stable"
    assert keys.embedding_part("m", "v", version, RUN_ID, 1).startswith(
        f"lake/embeddings/model=m/variant=v/version={version}/"
    )


@pytest.mark.parametrize("version", ["x-v1.1", "x-scratch", "x-run_3", "x-tmp.2"])
def test_ephemeral_tokens(version: str) -> None:
    assert keys.version_kind(version) == "ephemeral"
    built = keys.embedding_part("m", "v", version, RUN_ID, 1)
    assert keys.validate_key(built) == built


@pytest.mark.parametrize(
    "version", ["__v1", "v1.1", "V1", "v0", "x-", "x--v1", "x-v1."]
)
def test_bad_tokens_rejected(version: str) -> None:
    with pytest.raises(ValueError):
        keys.version_kind(version)
    with pytest.raises(ValueError):
        keys.embedding_part("m", "v", version, RUN_ID, 1)


def test_ephemeral_vocabulary_does_not_leak_to_artifacts() -> None:
    """Widening embeddings must not widen the artifact/eval branches."""
    with pytest.raises(ValueError, match="must be 'vN'"):
        keys.artifact_part("cleaned_text", "llm", "x-v1.1", RUN_ID, 1)


def test_ephemeral_part_key_validates() -> None:
    built = keys.embedding_part("all-minilm-l6-v2", "minilm", "x-v1.1", RUN_ID, 3)
    assert built == (
        "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/"
        f"version=x-v1.1/part-{RUN_ID}-00003.parquet"
    )
    assert keys.validate_key(built) == built


# --- Claim key --------------------------------------------------------------


def test_claim_key_shape() -> None:
    key = keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    assert key == (
        "lake/manifests/versions/embeddings/model=all-minilm-l6-v2/variant=minilm/"
        "version=v1/claim.json"
    )
    assert keys.validate_key(key) == key


def test_claim_keeps_token_verbatim() -> None:
    key = keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "x-v1.1")
    assert key.endswith("/variant=minilm/version=x-v1.1/claim.json")
    assert keys.validate_key(key) == key


def test_distinct_dotted_tokens_get_distinct_claims() -> None:
    """The claim segment must not fold `.` to `-`: the two tokens are
    distinct prefixes, so aliasing them onto one arbitration key would
    silently block a legitimate prefix."""
    dotted = keys.embedding_version_claim("m", "v", "x-a.b")
    literal = keys.embedding_version_claim("m", "v", "x-a-b")
    assert dotted != literal
    assert keys.validate_key(dotted) == dotted
    assert keys.validate_key(literal) == literal


# --- Claim protocol ---------------------------------------------------------


def test_first_claim_writes_the_record(memory_sink) -> None:
    record = claim_version(
        memory_sink,
        _config(),
        HASH_A,
        run_id=RUN_ID,
        claimed_at=CLAIMED_AT,
        pipeline_version="gitsha",
    )
    assert record["configuration_hash"] == HASH_A
    assert record["kind"] == "stable"
    assert record["version"] == "v1"
    key = keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    assert memory_sink.exists(key)


def test_same_configuration_reclaims_idempotently(memory_sink) -> None:
    first = claim_version(
        memory_sink, _config(), HASH_A, run_id=RUN_ID, claimed_at=CLAIMED_AT
    )
    stored = memory_sink.objects[
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    ]
    second = claim_version(
        memory_sink, _config(), HASH_A, run_id=RUN_ID_B, claimed_at=CLAIMED_AT
    )
    assert second == first
    assert (
        memory_sink.objects[
            keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
        ]
        == stored
    )


def test_different_configuration_is_refused(memory_sink) -> None:
    claim_version(memory_sink, _config(), HASH_A, run_id=RUN_ID, claimed_at=CLAIMED_AT)
    with pytest.raises(ClaimConflict) as excinfo:
        claim_version(
            memory_sink, _config(sql_text="SELECT 2"), HASH_B, run_id=RUN_ID_B
        )
    assert excinfo.value.owner_hash == HASH_A
    assert excinfo.value.incoming_hash == HASH_B
    assert "v1" in str(excinfo.value)
    assert HASH_A[:12] in str(excinfo.value)


def test_ephemeral_prefix_taken_by_another_configuration_is_refused(
    memory_sink,
) -> None:
    claim_version(memory_sink, _config(version="x-v1.1"), HASH_A, run_id=RUN_ID)
    with pytest.raises(ClaimConflict):
        claim_version(memory_sink, _config(version="x-v1.1"), HASH_B, run_id=RUN_ID_B)


def test_distinct_versions_do_not_contend(memory_sink) -> None:
    claim_version(memory_sink, _config(version="v1"), HASH_A, run_id=RUN_ID)
    record = claim_version(memory_sink, _config(version="v2"), HASH_B, run_id=RUN_ID_B)
    assert record["configuration_hash"] == HASH_B


def test_lost_race_is_judged_by_the_winner(memory_sink) -> None:
    """A claim that loses ``put_if_absent`` reads the winner and raises."""

    class _Racy(memory_sink.__class__):
        """Claims always lose the write race, then report the winner."""

        def put_if_absent(self, key: str, data: bytes) -> bool:
            if not super().exists(key):
                super().put(
                    key,
                    claim_bytes(
                        claim_record(
                            _config(),
                            HASH_A,
                            run_id=RUN_ID,
                            claimed_at=CLAIMED_AT,
                            pipeline_version="",
                        )
                    ),
                )
            return False

    store = _Racy()
    with pytest.raises(ClaimConflict):
        claim_version(store, _config(), HASH_B, run_id=RUN_ID_B)


def test_claim_precedes_reconcile_and_parts(journal) -> None:
    """The claim is the first durable write a run makes under its prefix."""
    claim_version(journal, _config(), HASH_A, run_id=RUN_ID, claimed_at=CLAIMED_AT)
    assert journal.journal == [
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    ]


# --- Legacy-prefix guard -----------------------------------------------------


def test_legacy_stable_prefix_refuses_unclaimed(memory_sink) -> None:
    """A stable prefix with parts but no claim is never silently adopted:
    first-writer-wins would mint an owner for production v1 (issue #22)."""
    part = keys.embedding_part("all-minilm-l6-v2", "minilm", "v1", RUN_ID, 1)
    memory_sink.put(part, b"")

    with pytest.raises(RuntimeError, match="claim_embedding_version"):
        claim_version(memory_sink, _config(), HASH_A, run_id=RUN_ID)


def test_legacy_ephemeral_prefix_still_claims(memory_sink) -> None:
    """Ephemeral prefixes have no legacy meaning — parts there were written
    post-registry, or the prefix is fresh; claiming proceeds either way."""
    part = keys.embedding_part("all-minilm-l6-v2", "minilm", "x-tmp.2", RUN_ID, 1)
    memory_sink.put(part, b"")

    record = claim_version(
        memory_sink, _config(version="x-tmp.2"), HASH_A, run_id=RUN_ID
    )
    assert record["configuration_hash"] == HASH_A


def test_operator_adopted_legacy_owner_reclaims_idempotently(memory_sink) -> None:
    """After the backfill adopts a legacy prefix, the owning configuration's
    pipeline runs proceed without the guard blocking them."""
    memory_sink.put(
        keys.embedding_part("all-minilm-l6-v2", "minilm", "v1", RUN_ID, 1), b""
    )

    adopted = adopt_version_prefix(
        memory_sink,
        _config(),
        HASH_A,
        run_id=RUN_ID,
    )
    assert adopted["configuration_hash"] == HASH_A

    repeat = claim_version(memory_sink, _config(), HASH_A, run_id=RUN_ID_B)
    assert repeat == adopted
    assert memory_sink.objects[
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    ] == claim_bytes(adopted)


# --- Gate drift check -------------------------------------------------------


def test_gate_reports_unclaimed_prefixes(memory_sink) -> None:
    from scripts.provision_query_layer.gate import check_embedding_version_claims

    part = keys.embedding_part("all-minilm-l6-v2", "minilm", "v1", RUN_ID, 1)
    memory_sink.put(part, b"")

    claimed, missing = check_embedding_version_claims(memory_sink)
    assert not claimed
    assert missing == ["all-minilm-l6-v2/minilm/version=v1"]

    memory_sink.put(
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1"), b"{}"
    )
    claimed, missing = check_embedding_version_claims(memory_sink)
    assert claimed and missing == []


def test_gate_maps_dotted_ephemeral_tokens(memory_sink) -> None:
    from scripts.provision_query_layer.gate import check_embedding_version_claims

    part = keys.embedding_part("all-minilm-l6-v2", "minilm", "x-v1.1", RUN_ID, 1)
    memory_sink.put(part, b"")
    memory_sink.put(
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "x-v1.1"), b"{}"
    )
    claimed, missing = check_embedding_version_claims(memory_sink)
    assert claimed and missing == []


def test_gate_reports_malformed_token_instead_of_crashing(memory_sink) -> None:
    """A bad version token on disk is drift to report, not a traceback."""
    from scripts.provision_query_layer.gate import (
        _backfill_command,
        check_embedding_version_claims,
    )

    memory_sink.objects[
        "lake/embeddings/model=all-minilm-l6-v2/variant=minilm/"
        f"version=v01/part-{RUN_ID}-00001.parquet"
    ] = b""

    claimed, missing = check_embedding_version_claims(memory_sink)
    assert not claimed
    assert missing == ["all-minilm-l6-v2/minilm/version=v01"]
    assert _backfill_command(missing[0]) is None


# --- Registry key shape -----------------------------------------------------


def test_registry_claim_key_is_positionally_pinned() -> None:
    """The claim branch has a shape pinner, like every other manifest branch."""
    good = keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
    assert keys.validate_key(good) == good

    for bad in (
        # wrong registry namespace
        good.replace("versions/embeddings", "versions/other"),
        # partition order flipped
        good.replace("model=all-minilm-l6-v2/variant=minilm", "variant=minilm/model=m"),
        # wrong leaf
        good.replace("claim.json", "owner.json"),
        # malformed token in the partition
        good.replace("version=v1", "version=1"),
        # missing a partition level
        good.replace("/variant=minilm", ""),
    ):
        with pytest.raises(ValueError):
            keys.validate_key(bad)


def test_claim_version_has_no_guard_bypass() -> None:
    """Adoption is a separate function, not a flag on the enforcing path."""
    import inspect

    from backend.pipeline.embedding import version_claim

    assert (
        "allow_preexisting_parts"
        not in inspect.signature(version_claim.claim_version).parameters
    )
    assert "adopt_version_prefix" in dir(version_claim)
