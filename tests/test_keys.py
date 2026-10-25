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

# ``run_id`` shares the posting-id identity: a canonical UUIDv7 minted once
# per pipeline run by ``ids.new_id()``; builders only validate it.
RUN_ID = ID
ARTIFACT_TYPE = "cleaned_text"
ARTIFACT_VARIANT = "llm"
EVAL_TYPE_GOLDEN = "cleaning_golden"
EVAL_TYPE_RESULT = "cleaning_quality"
EVAL_VARIANT = "author"
DATASET_ID = "cleaned_text-llm-v1"

ALL_VARIANTS = sorted(keys._DERIVED_VARIANTS)
ALL_VERSIONS = ["v1", "v2", "v10"]
ALL_RAW_SOURCES = sorted(keys.RAW_SOURCES)
ALL_ARTIFACT_TYPES = sorted(keys.ARTIFACT_TYPES)
ALL_EVAL_TYPES = sorted(keys.EVAL_TYPES)


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
        # ``datasets``/``runs`` are now pinned by their builders
        # (TestManifestDataset / TestManifestRuns); ``ingestion`` stays here.
        key = "lake/manifests/datasets/x.json"
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


# --- 1c. Artifact parts (lakehouse artifacts branch) -------------------------


class TestArtifactPart:
    """The ``lake/artifacts`` Parquet part builder: governed ``type`` /
    ``variant`` / ``version`` partitions and a run-minted part leaf."""

    def test_full_shape(self) -> None:
        assert keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, 1) == (
            f"lake/artifacts/type={ARTIFACT_TYPE}/variant={ARTIFACT_VARIANT}"
            f"/version=v1/part-{RUN_ID}-00001.parquet"
        )

    def test_seq_is_zero_padded_by_five(self) -> None:
        key = keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v2", RUN_ID, 12)
        assert key.endswith(f"part-{RUN_ID}-00012.parquet")

    @pytest.mark.parametrize("artifact_type", ALL_ARTIFACT_TYPES)
    def test_each_allowlisted_type(self, artifact_type: str) -> None:
        assert keys.artifact_part(
            artifact_type, ARTIFACT_VARIANT, "v1", RUN_ID, 1
        ).startswith(f"lake/artifacts/type={artifact_type}/")

    def test_output_is_valid(self) -> None:
        key = keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v3", RUN_ID, 1)
        assert keys.validate_key(key) == key

    def test_is_deterministic(self) -> None:
        assert keys.artifact_part(
            ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, 1
        ) == keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, 1)

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError):
            keys.artifact_part("knowbs", ARTIFACT_VARIANT, "v1", RUN_ID, 1)

    def test_rejects_bad_run_id(self) -> None:
        for bad in ("", "not-a-uuid", ID.replace("7", "4"), RUN_ID.upper()):
            with pytest.raises(ValueError):
                keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", bad, 1)

    def test_rejects_bad_seq(self) -> None:
        for seq in (0, -1):
            with pytest.raises(ValueError):
                keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, seq)


class TestValidateLakeArtifact:
    """``validate_key`` pins the semantic shape of ``lake/artifacts``: the
    ``type``/``variant``/``version`` partition stack in order, the allowlisted
    type, the ``vN`` version, and the ``part-<run-id>-<seq>.parquet`` leaf."""

    def _key(self, **overrides: str) -> str:
        parts = {
            "type": ARTIFACT_TYPE,
            "variant": ARTIFACT_VARIANT,
            "version": "v1",
            "run_id": RUN_ID,
            "seq": "00001",
        }
        parts.update(overrides)
        return (
            f"lake/artifacts/type={parts['type']}/variant={parts['variant']}"
            f"/version={parts['version']}/part-{parts['run_id']}-{parts['seq']}.parquet"
        )

    def test_valid_key_round_trips(self) -> None:
        assert keys.validate_key(self._key()) == self._key()

    def test_rejects_wrong_segment_count(self) -> None:
        for key in (
            "lake/artifacts",
            f"lake/artifacts/type={ARTIFACT_TYPE}",
            f"lake/artifacts/type={ARTIFACT_TYPE}/variant={ARTIFACT_VARIANT}",
            (
                f"lake/artifacts/type={ARTIFACT_TYPE}/variant={ARTIFACT_VARIANT}"
                "/version=v1"
            ),
            self._key() + "/extra",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_rejects_unknown_type_partition(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(self._key(type="knowbs"))

    def test_rejects_wrong_partition_names_or_order(self) -> None:
        long_run_part = f"part-{RUN_ID}-00001.parquet"
        reordered = (
            f"type={ARTIFACT_TYPE}/version=v1/variant={ARTIFACT_VARIANT}"
            f"/{long_run_part}"
        )
        extra_partition = (
            f"type={ARTIFACT_TYPE}/variant={ARTIFACT_VARIANT}/day=2026-10-05"
            f"/version=v1/{long_run_part}"
        )
        for wrong_tail in (
            f"{ARTIFACT_VARIANT}/version=v1/{long_run_part}",  # type missing
            reordered,
            extra_partition,
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/artifacts/{wrong_tail}")

    def test_rejects_non_version_partition_value(self) -> None:
        for bad_version in ("v0", "v01", "V1", "1"):
            with pytest.raises(ValueError):
                keys.validate_key(self._key(version=bad_version))

    def test_rejects_bad_part_leaf(self) -> None:
        for wrong_leaf in (
            f"part-{RUN_ID}-00000.parquet",  # zero is not 1-based
            f"part-{RUN_ID}-00001",  # missing extension
            f"part-{RUN_ID}-00001.json",  # wrong extension
            f"part-{RUN_ID.replace('7', '4')}-00001.parquet",  # not a uuid7
            f"part=chunk-{RUN_ID}-00001.parquet",  # leaf reads as a partition
            f"chunk-{RUN_ID}-00001.parquet",  # not a part leaf
            f"part-{RUN_ID}.parquet",  # missing sequence
        ):
            with pytest.raises(ValueError):
                keys.validate_key(
                    f"lake/artifacts/type={ARTIFACT_TYPE}"
                    f"/variant={ARTIFACT_VARIANT}/version=v1/{wrong_leaf}"
                )


# --- 1d. Eval datasets and results (lakehouse eval branch) -------------------


class TestEvalParts:
    """Authored ground truth (``datasets``) and machine output (``results``)
    share the partition stack but never a prefix."""

    @pytest.mark.parametrize(
        ("builder", "sub_branch", "eval_type"),
        [
            (keys.eval_dataset, "datasets", EVAL_TYPE_GOLDEN),
            (keys.eval_result, "results", EVAL_TYPE_RESULT),
        ],
        ids=["dataset", "result"],
    )
    def test_full_shape(self, builder, sub_branch: str, eval_type: str) -> None:
        assert (
            builder(eval_type, EVAL_VARIANT, "v1", RUN_ID, 1)
            == f"lake/eval/{sub_branch}/type={eval_type}/variant={EVAL_VARIANT}"
            f"/version=v1/part-{RUN_ID}-00001.parquet"
        )

    @pytest.mark.parametrize("eval_type", ALL_EVAL_TYPES)
    def test_each_allowlisted_type(self, eval_type: str) -> None:
        assert keys.eval_dataset(eval_type, EVAL_VARIANT, "v1", RUN_ID, 1).startswith(
            f"lake/eval/datasets/type={eval_type}/"
        )
        assert keys.eval_result(eval_type, EVAL_VARIANT, "v1", RUN_ID, 1).startswith(
            f"lake/eval/results/type={eval_type}/"
        )

    def test_outputs_are_valid(self) -> None:
        for builder in (keys.eval_dataset, keys.eval_result):
            key = builder(EVAL_TYPE_RESULT, EVAL_VARIANT, "v2", RUN_ID, 3)
            assert keys.validate_key(key) == key

    def test_rejects_unknown_type(self) -> None:
        for builder in (keys.eval_dataset, keys.eval_result):
            with pytest.raises(ValueError):
                builder("knowbs", EVAL_VARIANT, "v1", RUN_ID, 1)

    def test_rejects_bad_run_id_or_seq(self) -> None:
        for builder in (keys.eval_dataset, keys.eval_result):
            with pytest.raises(ValueError):
                builder(EVAL_TYPE_RESULT, EVAL_VARIANT, "v1", "not-a-uuid", 1)
            with pytest.raises(ValueError):
                builder(EVAL_TYPE_RESULT, EVAL_VARIANT, "v1", RUN_ID, 0)


class TestValidateLakeEval:
    """``validate_key`` pins the semantic shape of ``lake/eval/<datasets|results>``."""

    def _key(self, sub_branch: str = "results", **overrides: str) -> str:
        parts = {
            "type": EVAL_TYPE_RESULT,
            "variant": EVAL_VARIANT,
            "version": "v1",
            "run_id": RUN_ID,
            "seq": "00001",
        }
        parts.update(overrides)
        return (
            f"lake/eval/{sub_branch}/type={parts['type']}"
            f"/variant={parts['variant']}/version={parts['version']}"
            f"/part-{parts['run_id']}-{parts['seq']}.parquet"
        )

    def test_valid_keys_round_trip(self) -> None:
        assert keys.validate_key(self._key("datasets")) == self._key("datasets")
        assert keys.validate_key(self._key("results")) == self._key("results")

    def test_rejects_wrong_segment_count(self) -> None:
        for key in (
            "lake/eval",
            "lake/eval/datasets",
            "lake/eval/results/type=x",
            self._key() + "/extra",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_rejects_unknown_sub_branch(self) -> None:
        for sub_branch in ("data", "dashboards", "datasets/extra"):
            with pytest.raises(ValueError):
                keys.validate_key(self._key(sub_branch))

    def test_rejects_wrong_partition_names_or_order(self) -> None:
        long_run_part = f"part-{RUN_ID}-00001.parquet"
        reordered = (
            f"type={EVAL_TYPE_RESULT}/version=v1/variant={EVAL_VARIANT}/{long_run_part}"
        )
        no_version = f"type={EVAL_TYPE_RESULT}/variant={EVAL_VARIANT}/{long_run_part}"
        for wrong_tail in (reordered, no_version):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/eval/results/{wrong_tail}")

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(self._key(type="knowbs"))

    def test_rejects_bad_version(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(self._key(version="v0"))


# --- 1e. Dataset manifests (lakehouse manifests/datasets branch) --------------


class TestManifestDataset:
    """The addressed ``lake/manifests/datasets/<dataset-id>.json`` builder.

    The dataset id is the leaf address, never a partition (data.md §2.3), so
    plain names without ``=`` are enforced.
    """

    def test_full_shape(self) -> None:
        assert keys.manifest_dataset(DATASET_ID) == (
            f"lake/manifests/datasets/{DATASET_ID}.json"
        )

    def test_output_is_valid(self) -> None:
        key = keys.manifest_dataset(DATASET_ID)
        assert keys.validate_key(key) == key

    def test_is_deterministic(self) -> None:
        assert keys.manifest_dataset(DATASET_ID) == keys.manifest_dataset(DATASET_ID)

    @pytest.mark.parametrize(
        "bad",
        ["", "type=knowbs", "Cleaned Text", "cleaned/text", "a=b"],
        ids=["empty", "reads-as-partition", "uppercase", "slash", "equals"],
    )
    def test_rejects_non_plain_names(self, bad: str) -> None:
        with pytest.raises(ValueError):
            keys.manifest_dataset(bad)

    def test_validate_rejects_bad_leaf(self) -> None:
        for wrong_leaf in (
            f"{DATASET_ID}.parquet",  # wrong extension
            f"{DATASET_ID}.yaml",  # wrong extension
        ):
            with pytest.raises(ValueError):
                keys.validate_key(f"lake/manifests/datasets/{wrong_leaf}")

    def test_validate_rejects_nested_leaf(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(f"lake/manifests/datasets/{DATASET_ID}.json/extra")

    def test_validate_accepts_plain_json_leaf(self) -> None:
        key = "lake/manifests/datasets/dataset.json"
        assert keys.validate_key(key) == key


class TestManifestRuns:
    """The grouped ``lake/manifests/runs/run_id=<run-id>/`` builders: the
    aggregated run record (``run.json``) and per-flush checkpoints
    (``flush=<seq>.json``)."""

    def test_run_record_shape(self) -> None:
        assert keys.manifest_run(RUN_ID) == (
            f"lake/manifests/runs/run_id={RUN_ID}/run.json"
        )

    def test_checkpoint_shape(self) -> None:
        assert keys.manifest_run_checkpoint(RUN_ID, 1) == (
            f"lake/manifests/runs/run_id={RUN_ID}/flush=00001.json"
        )

    def test_checkpoint_seq_is_zero_padded_by_five(self) -> None:
        key = keys.manifest_run_checkpoint(RUN_ID, 42)
        assert key.endswith("flush=00042.json")

    def test_outputs_are_valid(self) -> None:
        for key in (keys.manifest_run(RUN_ID), keys.manifest_run_checkpoint(RUN_ID, 1)):
            assert keys.validate_key(key) == key

    def test_are_deterministic(self) -> None:
        assert keys.manifest_run(RUN_ID) == keys.manifest_run(RUN_ID)
        assert keys.manifest_run_checkpoint(RUN_ID, 2) == keys.manifest_run_checkpoint(
            RUN_ID, 2
        )

    def test_rejects_bad_run_id(self) -> None:
        for build in (
            keys.manifest_run,
            lambda rid: keys.manifest_run_checkpoint(rid, 1),
        ):
            with pytest.raises(ValueError):
                build("not-a-uuid")

    def test_rejects_bad_seq(self) -> None:
        for seq in (0, -3):
            with pytest.raises(ValueError):
                keys.manifest_run_checkpoint(RUN_ID, seq)


class TestValidateLakeManifestRuns:
    """``validate_key`` pins the semantic shape of ``lake/manifests/runs``."""

    def _run_key(self, **overrides: str) -> str:
        parts = {"run_id": RUN_ID, "leaf": "run.json"}
        parts.update(overrides)
        return f"lake/manifests/runs/run_id={parts['run_id']}/{parts['leaf']}"

    def test_valid_keys_round_trip(self) -> None:
        assert keys.validate_key(self._run_key()) == self._run_key()
        assert keys.validate_key(self._run_key(leaf="flush=00001.json")) == (
            self._run_key(leaf="flush=00001.json")
        )

    def test_rejects_wrong_segment_count(self) -> None:
        for key in (
            "lake/manifests/runs",
            f"lake/manifests/runs/run_id={RUN_ID}",
            f"lake/manifests/runs/run_id={RUN_ID}/run.json/extra",
        ):
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_rejects_wrong_grouping_name(self) -> None:
        for wrong in (
            f"id={RUN_ID}/run.json",
            f"listing_run={RUN_ID}/run.json",
            f"run-{RUN_ID}/run.json",
        ):
            key = f"lake/manifests/runs/{wrong}"
            with pytest.raises(ValueError):
                keys.validate_key(key)

    def test_rejects_bad_run_id_partition_value(self) -> None:
        with pytest.raises(ValueError):
            keys.validate_key(
                self._run_key(run_id="0190f3a2-7b41-4c9e-8a3d-5f6e1b2c4d70")
            )  # uuid7 version nibble

    def test_rejects_bad_leaf(self) -> None:
        for wrong_leaf in (
            "flush.json",
            "flush=abc.json",
            "flush=00000.json",  # not 1-based
            "flush=00001",  # missing extension
            "run.md",  # wrong extension
            "run",  # no extension
            "ray=00001.json",  # unknown leaf family
        ):
            with pytest.raises(ValueError):
                keys.validate_key(self._run_key(leaf=wrong_leaf))


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
        assert keys.artifact_part(
            ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, 1
        ) == keys.artifact_part(ARTIFACT_TYPE, ARTIFACT_VARIANT, "v1", RUN_ID, 1)
        assert keys.eval_dataset(
            EVAL_TYPE_GOLDEN, EVAL_VARIANT, "v1", RUN_ID, 1
        ) == keys.eval_dataset(EVAL_TYPE_GOLDEN, EVAL_VARIANT, "v1", RUN_ID, 1)
        assert keys.eval_result(
            EVAL_TYPE_RESULT, EVAL_VARIANT, "v1", RUN_ID, 1
        ) == keys.eval_result(EVAL_TYPE_RESULT, EVAL_VARIANT, "v1", RUN_ID, 1)
        assert keys.manifest_dataset(DATASET_ID) == keys.manifest_dataset(DATASET_ID)
        assert keys.manifest_run(RUN_ID) == keys.manifest_run(RUN_ID)
        assert keys.manifest_run_checkpoint(RUN_ID, 1) == keys.manifest_run_checkpoint(
            RUN_ID, 1
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

    @pytest.mark.parametrize("artifact_type", ALL_ARTIFACT_TYPES)
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    def test_artifact_part_output_is_valid(
        self, artifact_type: str, version: str
    ) -> None:
        key = keys.artifact_part(artifact_type, ARTIFACT_VARIANT, version, RUN_ID, 1)
        assert keys.validate_key(key) == key

    @pytest.mark.parametrize("eval_type", ALL_EVAL_TYPES)
    @pytest.mark.parametrize("version", ALL_VERSIONS)
    def test_eval_outputs_are_valid(self, eval_type: str, version: str) -> None:
        for builder in (keys.eval_dataset, keys.eval_result):
            key = builder(eval_type, EVAL_VARIANT, version, RUN_ID, 2)
            assert keys.validate_key(key) == key

    def test_manifest_dataset_output_is_valid(self) -> None:
        key = keys.manifest_dataset(DATASET_ID)
        assert keys.validate_key(key) == key

    def test_manifest_run_outputs_are_valid(self) -> None:
        for key in (keys.manifest_run(RUN_ID), keys.manifest_run_checkpoint(RUN_ID, 1)):
            assert keys.validate_key(key) == key
