"""Contract tests for :mod:`backend.storage.keys` (the key builder spec).

This suite is the **source of truth** for the intended key-building contract.
It deliberately does NOT mirror the current ``keys.py`` implementation: where
the implementation disagrees with the intended behavior, the implementation is
wrong and is expected to change. See
``.kilo/plans/1790319348784-key-builder-testing-spec.md``; its id scheme is
superseded by ``.kilo/plans/1790352000000-uuidv7-posting-id-migration.md``.

Scope: key *shape and rules* only. Nothing here asserts anything about real
postings, their content, or data correctness — data evaluations live outside
pytest (deferred ``eval/`` machinery).
"""

from __future__ import annotations

import sys
import uuid

import pytest

from backend.storage import keys

# --- Fixtures ----------------------------------------------------------------
#
# ``ID`` is a fixed literal UUIDv7 (canonical 36-char lowercase hyphenated).
# Posting ids are assigned at ingestion by ``keys.new_id()``; they are opaque
# and time-ordered, not derived from content.
ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"
MODEL = "all-minilm-l6-v2"
VARIANT = "llm-clean-text"
HASH = "a" * 64

ALL_VARIANTS = sorted(keys._DERIVED_VARIANTS)
ALL_VERSIONS = ["v1", "v2", "v10"]

requires_py314 = pytest.mark.skipif(
    sys.version_info < (3, 14),
    reason="uuid.uuid7() is new in Python 3.14 (RFC 9562); new_id() needs it",
)


# --- 1. Exact shapes ---------------------------------------------------------


class TestExactShapes:
    """Each builder returns the documented key byte-for-byte."""

    def test_raw_posting(self) -> None:
        assert keys.raw_posting(ID) == f"live/raw/{ID}/source.md"

    def test_raw_metadata(self) -> None:
        assert keys.raw_metadata(ID) == f"live/raw/{ID}/metadata.json"

    def test_derived_default_version(self) -> None:
        assert keys.derived(ID, VARIANT) == f"live/derived/{ID}/{VARIANT}/v1.json"

    def test_derived_explicit_version(self) -> None:
        assert keys.derived(ID, "labels", "v2") == f"live/derived/{ID}/labels/v2.json"

    def test_embedding(self) -> None:
        assert (
            keys.embedding(MODEL, VARIANT, "v1", ID)
            == f"live/embeddings/{MODEL}/{VARIANT}/v1/{ID}.npy"
        )

    def test_config(self) -> None:
        assert keys.config(HASH) == f"configs/{HASH}.json"

    def test_derived_fixture_matches_minted_id(self) -> None:
        # Cross-check: a freshly minted id is accepted and round-trips into a
        # key unchanged. ``new_id()`` is the only minter; builders only validate.
        minted = keys.new_id()
        assert keys.raw_posting(minted) == f"live/raw/{minted}/source.md"


# --- 2. Concept placement ----------------------------------------------------


class TestConceptPlacement:
    """Positional invariants: where each concept sits in the key."""

    def test_id_is_third_segment_for_raw(self) -> None:
        assert keys.raw_posting(ID).split("/")[2] == ID

    def test_id_is_third_segment_for_derived(self) -> None:
        assert keys.derived(ID, VARIANT).split("/")[2] == ID

    def test_bucket_leads_raw_and_derived(self) -> None:
        assert keys.raw_posting(ID).split("/")[0] == "live"
        assert keys.derived(ID, VARIANT).split("/")[0] == "live"

    def test_bucket_leads_config(self) -> None:
        assert keys.config(HASH).split("/")[0] == "configs"

    def test_embedding_model_leads_after_bucket(self) -> None:
        # live/embeddings/<model>/... -> model at index 2
        assert keys.embedding(MODEL, VARIANT, "v1", ID).split("/")[2] == MODEL

    def test_embedding_id_is_leaf_not_leading(self) -> None:
        parts = keys.embedding(MODEL, VARIANT, "v1", ID).split("/")
        assert ID not in parts[:-1]
        assert parts[-1] == f"{ID}.npy"

    def test_derived_variant_and_version_positions(self) -> None:
        parts = keys.derived(ID, VARIANT, "v2").split("/")
        assert parts[3] == VARIANT
        assert parts[4] == "v2.json"


# --- 3. Round trip -----------------------------------------------------------


class TestRoundTrip:
    """Inputs are recoverable from a well-formed key."""

    def test_derived_recovers_id_variant_version(self) -> None:
        parts = keys.derived(ID, VARIANT, "v3").split("/")
        # live/derived/<id>/<variant>/v<N>.json
        assert (parts[2], parts[3], parts[4]) == (ID, VARIANT, "v3.json")

    def test_embedding_recovers_model_variant_version_id(self) -> None:
        parts = keys.embedding(MODEL, VARIANT, "v4", ID).split("/")
        # live/embeddings/<model>/<variant>/v<N>/<id>.npy
        assert parts[2] == MODEL
        assert parts[3] == VARIANT
        assert parts[4] == "v4"
        assert parts[5] == f"{ID}.npy"

    def test_raw_recovers_id(self) -> None:
        assert keys.raw_posting(ID).split("/")[2] == ID
        assert keys.raw_metadata(ID).split("/")[2] == ID


# --- 4. Config hash ----------------------------------------------------------


class TestConfigHash:
    """The config hash token: 64-char lowercase hex, deterministic."""

    def test_accepts_full_hex(self) -> None:
        assert keys.config(HASH) == f"configs/{HASH}.json"

    def test_is_deterministic(self) -> None:
        assert keys.config(HASH) == keys.config(HASH)

    @pytest.mark.parametrize(
        "bad",
        ["", "abc", "z" * 64, "A" * 64, "a" * 63, "a" * 65],
        ids=["empty", "short", "non-hex-z", "uppercase-hex", "63-chars", "65-chars"],
    )
    def test_rejects_bad_hash(self, bad: str) -> None:
        with pytest.raises(ValueError):
            keys.config(bad)


# --- 5. Malformed keys -------------------------------------------------------


class TestRejectMalformedKeys:
    """``validate_key`` rejects structurally bad keys."""

    @pytest.mark.parametrize(
        "key",
        [
            "",
            "/live/raw/x/source.md",
            "live/raw/x/source.md/",
            "live//raw/x/source.md",
            "live/raw/../source.md",
            "live/raw/./source.md",
            "other/raw/x/source.md",
            "live/raw/x/source.md/extra.json",
        ],
        ids=[
            "empty",
            "leading-slash",
            "trailing-slash",
            "empty-segment",
            "dotdot",
            "dot",
            "unknown-bucket",
            "dot-in-nonterminal",
        ],
    )
    def test_rejects(self, key: str) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(key)

    def test_uppercase_outside_id_segment_is_rejected(self) -> None:
        # Everything is lowercase now: ids are [0-9a-f-], so there is no
        # uppercase exception anywhere. This is the only case rule.
        for key in (
            f"live/raw/{ID}/Source.md",
            f"live/derived/{ID}/LLM-clean-text/v1.json",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    @pytest.mark.parametrize(
        "leaf",
        [
            f"{ID}.txt",
            ID,
            f"{ID}.npy.npy",
        ],
        ids=["wrong-extension", "no-extension", "double-extension"],
    )
    def test_embedding_leaf_must_be_id_npy(self, leaf: str) -> None:
        prefix = "live/embeddings/all-minilm-l6-v2/llm-clean-text/v1"
        with pytest.raises(ValueError):
            keys.validate_key(f"{prefix}/{leaf}")


# --- 6. Bad builder inputs ---------------------------------------------------


class TestRejectBadBuilderInputs:
    """Builders reject bad input at construction time."""

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "0190f3a27b417c9e8a3d5f6e1b2c4d70",  # missing hyphens
            "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d7",  # too short (35)
            "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d700",  # too long (37)
            "0190f3a2-7b41-4c9e-8a3d-5f6e1b2c4d70",  # version nibble 4
            "0190f3a2-7b41-7c9e-1a3d-5f6e1b2c4d70",  # wrong variant
            "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d7g",  # non-hex char
            "0190F3A2-7B41-7C9E-8A3D-5F6E1B2C4D70",  # uppercase hex
            "0190f3a2-7b41-7c9e-8a3d5f6e1b2c4d70",  # hyphen positions
            "0190f3a2-7b417c9e-8a3d-5f6e1b2c4d70",  # hyphen positions
            "bad#char1234567890",  # reserved char
            "bad/char1234567890",  # slash
            "bad char1234567890",  # space
        ],
        ids=[
            "empty",
            "missing-hyphens",
            "too-short",
            "too-long",
            "wrong-version",
            "wrong-variant",
            "non-hex",
            "uppercase",
            "hyphen-after-4",
            "hyphen-after-6",
            "hash-char",
            "slash-char",
            "space-char",
        ],
    )
    def test_raw_posting_rejects_bad_id(self, bad_id: str) -> None:
        with pytest.raises(ValueError):
            keys.raw_posting(bad_id)

    @pytest.mark.parametrize(
        "bad_variant",
        ["", "NOT-A-VARIANT", "llm_clean_text"],
        ids=["empty", "uppercase", "underscore"],
    )
    def test_derived_rejects_unknown_variant(self, bad_variant: str) -> None:
        with pytest.raises(ValueError):
            keys.derived(ID, bad_variant)

    @pytest.mark.parametrize(
        "bad_version",
        ["", "v0", "v01", "2", "v", "V1"],
        ids=["empty", "zero", "leading-zero", "no-v", "bare-v", "uppercase-v"],
    )
    def test_derived_rejects_bad_version(self, bad_version: str) -> None:
        with pytest.raises(ValueError):
            keys.derived(ID, VARIANT, bad_version)

    @pytest.mark.parametrize(
        "bad_model",
        ["", "SomeModel", "model with space"],
        ids=["empty", "uppercase", "space"],
    )
    def test_embedding_rejects_bad_model(self, bad_model: str) -> None:
        with pytest.raises(ValueError):
            keys.embedding(bad_model, VARIANT, "v1", ID)


# --- 7. Determinism ----------------------------------------------------------


class TestDeterminism:
    """Builders are pure: same inputs -> identical output."""

    def test_all_builders_are_idempotent(self) -> None:
        assert keys.raw_posting(ID) == keys.raw_posting(ID)
        assert keys.raw_metadata(ID) == keys.raw_metadata(ID)
        assert keys.derived(ID, VARIANT, "v2") == keys.derived(ID, VARIANT, "v2")
        assert keys.embedding(MODEL, VARIANT, "v1", ID) == keys.embedding(
            MODEL, VARIANT, "v1", ID
        )
        assert keys.config(HASH) == keys.config(HASH)


# --- 8. Builder outputs are valid --------------------------------------------


class TestBuilderOutputsAreValid:
    """Cross-property: every builder output passes ``validate_key`` unchanged.

    This guards the two bugs found while drafting: dot-in-leaf and
    ``=``-in-partition. It also enforces the one-rule-set invariant — the
    builders' rules and the validator's rules must agree.
    """

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    def test_derived_output_is_valid(self, variant: str, version: str) -> None:
        key = keys.derived(ID, variant, version)
        assert keys.validate_key(key) == key

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    def test_embedding_output_is_valid(self, variant: str, version: str) -> None:
        key = keys.embedding(MODEL, variant, version, ID)
        assert keys.validate_key(key) == key

    def test_raw_outputs_are_valid(self) -> None:
        assert keys.validate_key(keys.raw_posting(ID)) == keys.raw_posting(ID)
        assert keys.validate_key(keys.raw_metadata(ID)) == keys.raw_metadata(ID)

    def test_config_output_is_valid(self) -> None:
        assert keys.validate_key(keys.config(HASH)) == keys.config(HASH)


# --- 9. new_id(): UUIDv7 minting ---------------------------------------------


@requires_py314
class TestNewId:
    """``new_id`` mints canonical, time-ordered UUIDv7 ids."""

    def test_is_parseable_uuid(self) -> None:
        parsed = uuid.UUID(keys.new_id())
        assert parsed.version == 7
        assert parsed.variant == uuid.RFC_4122

    def test_matches_id_regex(self) -> None:
        assert keys._ID_RE.fullmatch(keys.new_id()) is not None

    def test_is_canonical_lowercase_form(self) -> None:
        minted = keys.new_id()
        assert len(minted) == 36
        assert minted == minted.lower()
        assert minted == str(uuid.UUID(minted))

    def test_validate_key_round_trips_minted_id(self) -> None:
        minted = keys.new_id()
        key = keys.raw_posting(minted)
        assert keys.validate_key(key) == key

    def test_ids_are_monotonic_in_a_tight_loop(self) -> None:
        # UUIDv7 sorts lexicographically by generation time; the intra-ms
        # counter guarantees non-decreasing order within the same millisecond.
        ids = [keys.new_id() for _ in range(1000)]
        assert ids == sorted(ids)

    def test_ids_are_unique(self) -> None:
        ids = [keys.new_id() for _ in range(1000)]
        assert len(set(ids)) == 1000
