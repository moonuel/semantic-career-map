"""Contract tests for :mod:`backend.storage.keys` (the key builder spec).

This suite is the **source of truth** for the intended key-building contract.
It deliberately does NOT mirror the current ``keys.py`` implementation: where
the implementation disagrees with the intended behavior, the implementation is
wrong and is expected to change. See
``.kilo/plans/1790319348784-key-builder-testing-spec.md``.

Scope: key *shape and rules* only. Nothing here asserts anything about real
postings, their content, or data correctness — data evaluations live outside
pytest (deferred ``eval/`` machinery).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib

import pytest

from backend.storage import keys

# --- Fixtures ----------------------------------------------------------------
#
# ``ID`` is a synthetic 16-char base64url string, *deliberately mixed-case* to
# exercise case-sensitivity: base64url (RFC 4648 §5) is case-sensitive, and
# uppercase is legal in the id segment even though it is illegal elsewhere.
ID = "K3hI-o7w4Qx_9m2N"
MODEL = "all-minilm-l6-v2"
VARIANT = "llm-clean-text"
HASH = "a" * 64
DATE = dt.date(2026, 9, 23)

ALL_VARIANTS = sorted(keys._DERIVED_VARIANTS)
ALL_VERSIONS = ["v1", "v2", "v10"]


def _valid_id_from(source: bytes) -> str:
    """The intended id derivation: first 16 chars of unpadded base64url(sha256).

    Encoded here independently of ``keys.py`` so the test does not merely
    re-run the implementation. The builder layer never computes this (id
    computation is a separate concern), but the *shape* it must accept is this.
    """
    digest = hashlib.sha256(source).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")[:16]


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

    def test_manifest(self) -> None:
        assert keys.manifest(DATE) == "live/meta/datasets/day=2026-09-23/manifest.json"

    def test_derived_fixture_matches_recomputed_id(self) -> None:
        # Cross-check: an id derived by the intended algorithm is accepted and
        # round-trips into a key unchanged.
        derived_id = _valid_id_from(b"# Senior Engineer\nWe are hiring...")
        assert len(derived_id) == 16
        assert keys.raw_posting(derived_id) == f"live/raw/{derived_id}/source.md"


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

    def test_manifest_partition_and_leaf_positions(self) -> None:
        parts = keys.manifest(DATE).split("/")
        assert parts[-2] == "day=2026-09-23"
        assert parts[-1] == "manifest.json"


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

    def test_manifest_recovers_iso_date(self) -> None:
        partition = keys.manifest(DATE).split("/")[-2]
        assert partition == "day=2026-09-23"
        assert dt.date.fromisoformat(partition.removeprefix("day=")) == DATE


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

    def test_uppercase_id_segment_is_accepted(self) -> None:
        # Uppercase is legal in the id segment: base64url ids are mixed-case.
        assert keys.validate_key(f"live/raw/{ID}/source.md") == (
            f"live/raw/{ID}/source.md"
        )

    @pytest.mark.parametrize(
        "key",
        [
            f"live/raw/{ID}/Source.md",
            f"live/derived/{ID}/LLM-clean-text/v1.json",
        ],
        ids=["uppercase-filename", "uppercase-variant"],
    )
    def test_uppercase_outside_id_segment_is_rejected(self, key: str) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(key)

    @pytest.mark.parametrize(
        "leaf",
        ["K3hI-o7w4Qx_9m2N.txt", "K3hI-o7w4Qx_9m2N", "K3hI-o7w4Qx_9m2N.npy.npy"],
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
            "a..b",
            "shortID",  # 7 chars: wrong length
            "x" * 32,  # too long
            "bad#char1234567",
            "bad/char1234567",
            "bad=char1234567",
            "bad char1234567",
        ],
        ids=[
            "empty",
            "dots",
            "too-short",
            "too-long",
            "hash-char",
            "slash-char",
            "equals-char",
            "space-char",
        ],
    )
    def test_raw_posting_rejects_bad_id(self, bad_id: str) -> None:
        with pytest.raises(ValueError):
            keys.raw_posting(bad_id)

    def test_uppercase_is_not_why_an_id_is_rejected(self) -> None:
        # ``Affirm`` is valid base64url (6 chars) and would be *legal case-wise*,
        # but id length is fixed at 16, so it is rejected for length, not case.
        # This documents that we do NOT reject uppercase ids.
        with pytest.raises(ValueError):
            keys.raw_posting("Affirm")

    def test_16_char_uppercase_id_is_accepted(self) -> None:
        upper_id = "AffirmTechnologi"  # 16 chars, mixed case
        assert len(upper_id) == 16
        assert keys.raw_posting(upper_id) == f"live/raw/{upper_id}/source.md"

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
        assert keys.manifest(DATE) == keys.manifest(DATE)


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

    def test_manifest_output_is_valid(self) -> None:
        assert keys.validate_key(keys.manifest(DATE)) == keys.manifest(DATE)
