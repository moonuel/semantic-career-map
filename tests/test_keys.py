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

import pytest

from backend.storage import ids, keys

# --- Fixtures ----------------------------------------------------------------
#
# ``ID`` is a fixed literal UUIDv7 (canonical 36-char lowercase hyphenated).
# Posting ids are assigned at ingestion by ``ids.new_id()``; they are opaque
# and time-ordered, not derived from content. Identity minting is tested in
# ``test_ids.py``.
ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"
MODEL = "all-minilm-l6-v2"
VARIANT = "llm-clean-text"
HASH = "a" * 64

ALL_VARIANTS = sorted(keys._DERIVED_VARIANTS)
ALL_VERSIONS = ["v1", "v2", "v10"]
ALL_RAW_SOURCES = sorted(keys._RAW_SOURCES)


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

    def test_manifest_ingestion(self) -> None:
        assert (
            keys.manifest_ingestion("2026-10-06", ID)
            == f"lake/manifests/ingestion/ingested_on=2026-10-06/ingestion_id={ID}.parquet"
        )

    def test_derived_fixture_matches_minted_id(self) -> None:
        # Cross-check: a freshly minted id is accepted and round-trips into a
        # key unchanged. ``new_id()`` is the only minter; builders only validate.
        minted = ids.new_id()
        assert keys.raw_posting(minted) == f"live/raw/{minted}/source.md"


# --- 1b. Raw lakehouse payload -----------------------------------------------


class TestRawPayload:
    """The content-addressed ``lake/raw`` payload builder."""

    def test_full_shape(self) -> None:
        assert (
            keys.raw_payload("linkedin", HASH)
            == f"lake/raw/source=linkedin/sha256={HASH}.json"
        )

    @pytest.mark.parametrize("source", ALL_RAW_SOURCES)
    def test_each_allowlisted_source(self, source: str) -> None:
        assert keys.raw_payload(source, HASH).startswith(f"lake/raw/source={source}/")

    def test_output_is_valid(self) -> None:
        key = keys.raw_payload("greenhouse", HASH)
        assert keys.validate_key(key) == key

    def test_is_deterministic(self) -> None:
        assert keys.raw_payload("linkedin", HASH) == keys.raw_payload("linkedin", HASH)

    @pytest.mark.parametrize(
        "source",
        ["", "linkedIn", "Linkedin", "stackoverflow", "a" * 40, "with space"],
        ids=["empty", "mixed-case", "uppercase", "unknown-source", "too-long", "space"],
    )
    def test_rejects_non_allowlisted_source(self, source: str) -> None:
        with pytest.raises(ValueError):
            keys.raw_payload(source, HASH)

    def test_rejects_bad_hash(self) -> None:
        for bad in ("", "abc", "a" * 63, "A" * 64):
            with pytest.raises(ValueError):
                keys.raw_payload("linkedin", bad)

    def test_validate_key_rejects_bad_raw_partition(self) -> None:
        # source is a partition: it must be a well-formed key segment.
        key = f"lake/raw/source=../sha256={HASH}.json"
        with pytest.raises(ValueError):
            keys.validate_key(key)


class TestValidateLakeRaw:
    """``validate_key`` pins the semantic shape of ``lake/raw``, not just the
    segment alphabet: the single ``source`` partition, the allowlisted source,
    and the ``sha256=<hex>.json`` leaf."""

    def _key(self, **overrides: str) -> str:
        parts = {
            "source": "linkedin",
            "hash": HASH,
        }
        parts.update(overrides)
        return f"lake/raw/source={parts['source']}/sha256={parts['hash']}.json"

    def test_valid_key_round_trips(self) -> None:
        assert keys.validate_key(self._key()) == self._key()

    def test_rejects_wrong_segment_count(self) -> None:
        for key in (
            "lake/raw/source=linkedin",
            f"lake/raw/source=linkedin/sha256={HASH}.json/extra",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_rejects_unknown_source_partition(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(self._key(source="stackoverflow"))

    def test_rejects_bad_hash_leaf(self) -> None:
        for wrong_leaf in (
            "sha256=abc.json",
            f"sha256={'a' * 63}.json",
            f"sha256={'A' * 64}.json",
            f"md5={HASH}.json",
            f"sha256={HASH}",  # missing .json extension
            f"sha256={HASH}.md",  # wrong extension
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/raw/source=linkedin/{wrong_leaf}")

    def test_rejects_wrong_partition_names_or_order(self) -> None:
        for wrong_tail in (
            f"sha256={HASH}.json",  # source partition missing
            f"linkedin/sha256={HASH}.json",  # bare segment, no k=v partition
            f"source=linkedin/day=2026-10-05/sha256={HASH}.json",  # extra
            f"source=linkedin/source=linkedin/sha256={HASH}.json",  # duplicated
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/raw/{wrong_tail}")


class TestValidateLakeManifest:
    """``validate_key`` pins the semantic shape of ``lake/manifests/ingestion``:
    the single ``ingested_on`` day partition and the
    ``ingestion_id=<uuid7>.parquet`` leaf."""

    def _key(self, **overrides: str) -> str:
        parts = {
            "date": "2026-10-06",
            "id": ID,
        }
        parts.update(overrides)
        return (
            f"lake/manifests/ingestion/ingested_on={parts['date']}"
            f"/ingestion_id={parts['id']}.parquet"
        )

    def test_valid_key_round_trips(self) -> None:
        assert keys.validate_key(self._key()) == self._key()

    def test_rejects_wrong_segment_count(self) -> None:
        for key in (
            "lake/manifests/ingestion",
            self._key() + "/extra",
            "lake/manifests/ingestion/ingested_on=2026-10-06",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_other_manifest_sub_branches_stay_unpinned(self) -> None:
        # ``datasets``/``runs`` are reserved and not yet committed to a shape;
        # only the ``ingestion`` sub-branch is pinned here.
        key = "lake/manifests/datasets/part-00000.parquet"
        assert keys.validate_key(key) == key

    def test_rejects_bad_dates(self) -> None:
        for bad in (
            "2026-10-6",  # not zero-padded
            "20261006",  # no dashes
            "26-10-06",  # short year
            "2026-13-01",  # no such month
            "2026-02-30",  # no such day
            "",  # empty
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/manifests/ingestion/ingested_on={bad}")
                keys.manifest_ingestion(bad, ID)

    def test_rejects_non_canonical_partition_name(self) -> None:
        for wrong in (
            f"event_on=2026-10-06/ingestion_id={ID}.parquet",
            f"date=2026-10-06/ingestion_id={ID}.parquet",
            f"ingested_on=2026-10-06/day=2026-10-06/ingestion_id={ID}.parquet",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/manifests/ingestion/{wrong}")

    def test_rejects_bad_leaf(self) -> None:
        for wrong_leaf in (
            "ingestion_id=abc.parquet",  # not a uuid
            f"ingestion_id={ID}",  # missing extension
            f"ingestion_id={ID}.json",  # wrong extension
            f"run_id={ID}.parquet",  # wrong name
        ):
            with pytest.raises(ValueError):
                keys.validate_key(
                    f"lake/manifests/ingestion/ingested_on=2026-10-06/{wrong_leaf}"
                )


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
        assert keys.raw_payload("linkedin", HASH) == keys.raw_payload("linkedin", HASH)
        assert keys.derived(ID, VARIANT, "v2") == keys.derived(ID, VARIANT, "v2")
        assert keys.embedding(MODEL, VARIANT, "v1", ID) == keys.embedding(
            MODEL, VARIANT, "v1", ID
        )
        assert keys.config(HASH) == keys.config(HASH)
        assert keys.manifest_ingestion("2026-10-06", ID) == keys.manifest_ingestion(
            "2026-10-06", ID
        )


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

    @pytest.mark.parametrize("source", ALL_RAW_SOURCES)
    def test_raw_payload_output_is_valid(self, source: str) -> None:
        assert keys.validate_key(keys.raw_payload(source, HASH)) == keys.raw_payload(
            source, HASH
        )

    def test_config_output_is_valid(self) -> None:
        assert keys.validate_key(keys.config(HASH)) == keys.config(HASH)

    def test_manifest_ingestion_output_is_valid(self) -> None:
        key = keys.manifest_ingestion("2026-10-06", ID)
        assert keys.validate_key(key) == key
