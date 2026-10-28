"""Production-resolution unit tests (Layer 3 harvest).

The read chain attacker-model: a pointer document whose embeddings entries
resolve through claims to content-addressed configs; every gap is a loud
failure naming the exact operator command. All at the MemorySink double —
no live endpoints. Also the embed entry-path units (pointer-mode flag
arithmetic + pointer-resolved config selection).
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pytest

from backend.pipeline.embedding import config as config_module
from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash
from backend.pipeline.production import resolve_production
from backend.storage import keys, pointer
from tests.conftest import MemorySink

# --- Fixtures ----------------------------------------------------------------


def run_config() -> EmbeddingConfig:
    """The production generation's parameters (the live a29a35e3… shape)."""
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        model_segment="all-minilm-l6-v2",
        variant="minilm",
        text_variant="variant_text",
        batch_size=16,
        max_length=256,
        normalize=True,
        input_artifact=config_module.DEFAULT_INPUT_ARTIFACT,
        input_version="v1",
        version="v1",
        sql_text="SELECT source_hash FROM postings",
    )


def pointer_doc() -> dict:
    """A parse-valid pointer document naming one embeddings generation."""
    config = run_config()
    return {
        "pointer_version": 1,
        "updated_by": "olimoon",
        "updated_at": "2026-10-28T17:00:00+00:00",
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
                        "model_segment": config.model_segment,
                        "variant": config.variant,
                        "version": config.version,
                    }
                ]
            },
        },
    }


def seed_chain(sink: MemorySink, config: EmbeddingConfig) -> str:
    """Seed pointer + claim + config object; return the owner hash."""
    owner = configuration_hash(config)
    sink.objects[keys.state_pointer()] = pointer.canonical_bytes(pointer_doc())
    sink.objects[
        keys.embedding_version_claim(
            config.model_segment, config.variant, config.version
        )
    ] = json.dumps({"configuration_hash": owner}).encode()
    sink.objects[keys.config(owner)] = config_module.config_document(config)
    return owner


# --- 1. The happy chain ------------------------------------------------------


class TestResolveProduction:
    """The read chain end to end over the sink."""

    def test_resolves_stacks_claim_and_config(self, memory_sink: MemorySink) -> None:
        owner = seed_chain(memory_sink, run_config())
        production = resolve_production(memory_sink)
        assert production.cleaning.stack == "cleaned_text/llm/v1"
        (emb,) = production.embeddings
        assert emb.stack == "all-minilm-l6-v2/minilm/v1"
        assert emb.configuration_hash == owner
        assert emb.config == run_config()

    def test_pre_parsed_pointer_injection(self, memory_sink: MemorySink) -> None:
        """The ``pointer`` parameter bypasses the read (fixture documents)."""
        seed_chain(memory_sink, run_config())
        doc = json.loads(memory_sink.objects[keys.state_pointer()])
        del memory_sink.objects[keys.state_pointer()]
        production = resolve_production(memory_sink, pointer=doc)
        assert production.pointer == doc

    def test_identity_bind_is_inherent(self, memory_sink: MemorySink) -> None:
        """The returned hash re-hashes from the loaded config — the anti-join
        value a reader binds is computed, not trusted from the claim."""
        owner = seed_chain(memory_sink, run_config())
        production = resolve_production(memory_sink)
        assert configuration_hash(production.embeddings[0].config) == owner


# --- 2. Loud failures with operator commands ---------------------------------


class TestLoudFailures:
    """Every gap names its fix — never a silent fallback."""

    def test_absent_pointer_names_bootstrap(self, memory_sink: MemorySink) -> None:
        with pytest.raises(RuntimeError, match="--bootstrap --apply"):
            resolve_production(memory_sink)

    def test_missing_claim_names_backfill_command(
        self, memory_sink: MemorySink
    ) -> None:
        sink_objects = memory_sink.objects
        sink_objects[keys.state_pointer()] = pointer.canonical_bytes(pointer_doc())
        with pytest.raises(
            RuntimeError, match="claim_embedding_version.py.*--apply"
        ) as exc:
            resolve_production(memory_sink)
        assert "all-minilm-l6-v2/minilm/v1" in str(exc.value)

    def test_orphan_claim_names_the_missing_config(
        self, memory_sink: MemorySink
    ) -> None:
        config = run_config()
        owner = seed_chain(memory_sink, config)
        del memory_sink.objects[keys.config(owner)]
        with pytest.raises(RuntimeError, match="orphan claim"):
            resolve_production(memory_sink)

    def test_config_drift_from_address_is_loud(self, memory_sink: MemorySink) -> None:
        config = run_config()
        owner = seed_chain(memory_sink, config)
        # The stored config object drifted: different parameters at the
        # same address (tampered content).
        tampered = dataclasses.replace(config, sql_text="SELECT something_else")
        memory_sink.objects[keys.config(owner)] = config_module.config_document(
            tampered
        )
        with pytest.raises(RuntimeError, match="drifted from its address"):
            resolve_production(memory_sink)

    def test_unknown_tier_refused(self, memory_sink: MemorySink) -> None:
        doc = pointer_doc()
        doc["tiers"]["embedings"] = doc["tiers"].pop("embeddings")
        with pytest.raises(ValueError, match="must tier exactly"):
            resolve_production(memory_sink, pointer=doc)

    def test_ephemeral_version_in_pointer_refused(
        self, memory_sink: MemorySink
    ) -> None:
        """A pointer names production; ``x-...`` tokens never qualify."""
        doc = pointer_doc()
        doc["tiers"]["embeddings"]["entries"][0]["version"] = "x-scratch"
        with pytest.raises(ValueError, match="never production"):
            resolve_production(memory_sink, pointer=doc)


# --- 3. The `production.resolved` summary (Layer 5) ---------------------------


class TestResolvedSummary:
    """The canonical observability renderer: who named this, what, why."""

    def test_carries_provenance_stacks_and_config_identity(
        self, memory_sink: MemorySink
    ) -> None:
        from backend.pipeline.production import resolved_summary

        owner = seed_chain(memory_sink, run_config())
        summary = resolved_summary(resolve_production(memory_sink))
        assert summary["pointer"] == keys.state_pointer()
        assert summary["updated_by"] == "olimoon"
        assert summary["reason"] == "bootstrap: describes reality"
        assert summary["cleaning"] == "cleaned_text/llm/v1"
        (emb,) = summary["embeddings"]
        assert emb["stack"] == "all-minilm-l6-v2/minilm/v1"
        assert emb["claim_owner"] == owner
        assert emb["config_path"] == keys.config(owner)
        assert emb["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        # The rendered hash is the stored document's canonical hash.
        assert summary["pointer_hash"] == pointer.content_hash(pointer_doc())

    def test_json_serializable(self, memory_sink: MemorySink) -> None:
        import json

        from backend.pipeline.production import resolved_summary

        seed_chain(memory_sink, run_config())
        text = json.dumps(resolved_summary(resolve_production(memory_sink)))
        assert "all-minilm-l6-v2/minilm/v1" in text


# --- 4. The embed entry path --------------------------------------------------


def _args(**kw: object) -> argparse.Namespace:
    """A parsed-embed CLI namespace: every pointer flag None by default."""
    from scripts.embed import _POINTER_FLAGS

    fields: dict[str, object] = {name: None for name in _POINTER_FLAGS}
    fields.update({"database": "db", "workgroup": "wg", "write_config_to": None})
    fields.update(kw)
    return argparse.Namespace(**fields)


class TestEmbedEntryPath:
    """Flag-mode arithmetic + pointer-resolved config selection."""

    def test_all_unset_is_pointer_mode(self) -> None:
        from scripts.embed import pointer_mode

        assert pointer_mode(_args()) is True

    def test_any_config_flag_selects_build_path(self) -> None:
        from scripts.embed import pointer_mode

        assert pointer_mode(_args(variant="x")) is False
        assert pointer_mode(_args(config_file="c.json")) is False
        assert pointer_mode(_args(sql_file="s.sql")) is False

    def test_pointer_resolved_config_is_the_scheduled_one(
        self, memory_sink: MemorySink
    ) -> None:
        from scripts.embed import config_from_store

        owner = seed_chain(memory_sink, run_config())
        config = config_from_store(memory_sink)
        assert configuration_hash(config) == owner
        assert config == run_config()

    def test_pointer_mode_without_chain_fails_loudly(
        self, memory_sink: MemorySink
    ) -> None:
        from scripts.embed import config_from_store

        with pytest.raises(RuntimeError, match="--bootstrap --apply"):
            config_from_store(memory_sink)

    def test_build_path_defaults_match_prepointer_behavior(self) -> None:
        """Unset flags fall back to the build defaults the CLI shipped with."""
        from scripts.embed import build_config

        config = build_config(_args())
        assert config.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.variant == "minilm"
        assert config.batch_size == 16
        assert config.version == "v1"
        # The build path never resolves segments from the pointer: explicit
        # flags mint their own generation.
        assert config.model_segment == "all-minilm-l6-v2"
