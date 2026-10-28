"""Promote-CLI and pointer-gate unit tests (Layer 3 harvest).

Build-then-flip refusals (claims, configs, parts), the bootstrap/create
split, and the pointer gate's row semantics — ERROR rows carry the exact
operator command, malformed input is reported not raised, catalog drift
rows name the re-provision command. All MemorySink-based; the catalog
check runs against a FakeAthena.
"""

from __future__ import annotations

import json

import pytest

from backend.pipeline.embedding import config as config_module
from backend.pipeline.embedding.config import EmbeddingConfig, configuration_hash
from backend.storage import keys, pointer
from backend.storage.pointer import content_hash
from scripts.promote_production import (
    build_bootstrap_doc,
    load_target_doc,
    readiness_refusals,
)
from scripts.provision_query_layer.gate import check_production_pointer
from tests.conftest import MemorySink

RUN_ID = "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70"


@pytest.fixture
def bucket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store's single required env (run_query reads the bucket)."""
    monkeypatch.setenv("STORE_S3_BUCKET", "lake")


# --- Shared fixture helpers ---------------------------------------------------


def run_config(version: str = "v1") -> EmbeddingConfig:
    """One real-generation's parameters (identity verifiable)."""
    return EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        model_segment="all-minilm-l6-v2",
        variant="minilm",
        text_variant="variant_text",
        version=version,
        sql_text="SELECT source_hash FROM postings",
    )


def pointer_doc(version: str = "v1") -> dict:
    """A parse-valid pointer naming one embeddings entry + cleaning v1."""
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
                        "model_segment": "all-minilm-l6-v2",
                        "variant": "minilm",
                        "version": version,
                    }
                ]
            },
        },
    }


def seed_full_chain(
    sink: MemorySink, *, version: str = "v1", with_parts: bool = True
) -> None:
    """Pointer + claim + config + (optionally) one part in each prefix."""
    config = run_config(version)
    owner = configuration_hash(config)
    sink.objects[keys.state_pointer()] = pointer.canonical_bytes(pointer_doc(version))
    sink.objects[
        keys.embedding_version_claim("all-minilm-l6-v2", "minilm", version)
    ] = json.dumps({"configuration_hash": owner}).encode()
    sink.objects[keys.config(owner)] = config_module.config_document(config)
    if with_parts:
        sink.objects[keys.artifact_part("cleaned_text", "llm", "v1", RUN_ID, 1)] = b"P"
        sink.objects[
            keys.embedding_part("all-minilm-l6-v2", "minilm", version, RUN_ID, 1)
        ] = b"P"


def cli_args(**kw: object):
    """Promote CLI namespace with the shell's always-present fields."""
    import argparse

    fields: dict[str, object] = {
        "updated_by": "olimoon",
        "reason": "manual promotion",
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "version": "v1",
        "bootstrap": False,
        "from_file": None,
        "allow_incomplete": False,
    }
    fields.update(kw)
    return argparse.Namespace(**fields)


class FakeAthena:
    """run_query's Athena stand-in for the catalog-projection check.

    ``rows`` is the full result stream: header first, then data rows.
    """

    def __init__(self, rows: list[str]) -> None:
        self._rows = rows

    def start_query_execution(self, **kw) -> dict:
        return {"QueryExecutionId": "q-gate7"}

    def get_query_execution(self, *, QueryExecutionId: str) -> dict:
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, *, QueryExecutionId: str) -> dict:
        return {
            "ResultSet": {
                "Rows": [
                    {"Data": [{"VarCharValue": c} for c in row.split(",")]}
                    for row in self._rows
                ]
            }
        }


# --- 1. Bootstrap document builder --------------------------------------------


class TestBootstrapDoc:
    """The bootstrap minter: parse-valid, stable tokens only."""

    def test_defaults_parse_and_describe_reality(self) -> None:
        doc = build_bootstrap_doc(cli_args())
        assert pointer.parse_document(doc) is doc
        assert doc["tiers"]["embeddings"]["entries"][0]["version"] == "v1"

    def test_ephemeral_bootstrap_version_refused(self) -> None:
        doc = build_bootstrap_doc(cli_args(version="x-scratch"))
        with pytest.raises(ValueError, match="re-mint|never production"):
            pointer.parse_document(doc)


# --- 2. Build-then-flip -------------------------------------------------------


class TestReadinessRefusals:
    """A pointer may name only complete generations."""

    def test_complete_generation_passes(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink)
        assert readiness_refusals(memory_sink, pointer_doc()) == []

    def test_missing_claim_refuses_with_command(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink)
        del memory_sink.objects[
            keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
        ]
        refusals = readiness_refusals(memory_sink, pointer_doc())
        assert len(refusals) == 1
        assert "claim_embedding_version.py" in refusals[0]

    def test_missing_config_refuses_orphan(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink)
        del memory_sink.objects[keys.config(configuration_hash(run_config()))]
        refusals = readiness_refusals(memory_sink, pointer_doc())
        assert any("orphan claim" in r for r in refusals)

    def test_partless_embeddings_refuses_build_first(
        self, memory_sink: MemorySink
    ) -> None:
        seed_full_chain(memory_sink, with_parts=False)
        refusals = readiness_refusals(memory_sink, pointer_doc())
        assert any("partless" in r and "--allow-incomplete" in r for r in refusals)

    def test_partless_cleaning_refuses_run_first(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink)
        prefix = keys.artifact_part("cleaned_text", "llm", "v1", RUN_ID, 1)
        del memory_sink.objects[prefix]
        refusals = readiness_refusals(memory_sink, pointer_doc())
        assert any(r.startswith("cleaning stack") and "partless" in r for r in refusals)

    def test_from_file_without_bootstrap_parses_with_defaults(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--from-file tiers ride; metadata defaults fill from flags."""
        doc = pointer_doc()
        path = tmp_path / "promote.json"
        path.write_text(json.dumps(doc))
        monkeypatch.setattr("sys.argv", ["prog"])
        args = cli_args(from_file=str(path))
        loaded = load_target_doc(args)
        assert loaded["tiers"] == doc["tiers"]
        assert loaded["updated_by"] == "olimoon"


# --- 3. Pointer gate (gate 7) -------------------------------------------------


def promote_main(monkeypatch, argv: list[str]) -> int:
    """Run the promote CLI's main with argv patched (returns its exit code)."""
    from scripts import promote_production

    monkeypatch.setattr("sys.argv", ["promote_production.py", *argv])
    return promote_production.main()


class TestPromoteCli:
    """main() over an injected store: the write path's refusal walk.

    ``main`` builds its store via ``ObjectStore.from_env``; the tests
    monkeypatch that factory to a sink — the CLI's real store never moves
    in these tests.
    """

    def _patch(self, monkeypatch: pytest.MonkeyPatch, sink: MemorySink) -> None:
        from scripts import promote_production

        monkeypatch.setattr(
            promote_production.ObjectStore, "from_env", staticmethod(lambda: sink)
        )

    @staticmethod
    def _target_file(tmp_path, doc: dict) -> str:
        path = tmp_path / "promote.json"
        path.write_text(json.dumps(doc | {"updated_by": "olimoon"}))
        return str(path)

    @staticmethod
    def _v9_target_doc() -> dict:
        """A parse-valid target whose cleaning stack has no parts."""
        doc = pointer_doc()
        doc["tiers"]["cleaning"]["version"] = "v9"
        doc["reason"] = "promote: cleaning v9 (not yet built)"
        return doc

    def test_show_default_writes_nothing(
        self, monkeypatch, capsys, memory_sink: MemorySink
    ) -> None:
        self._patch(monkeypatch, memory_sink)
        seed_full_chain(memory_sink)
        before = memory_sink.objects[keys.state_pointer()]
        assert promote_main(monkeypatch, []) == 0
        assert memory_sink.objects[keys.state_pointer()] == before
        assert "content hash" in capsys.readouterr().out

    def test_show_reports_broken_resolution(
        self, monkeypatch, capsys, memory_sink: MemorySink
    ) -> None:
        self._patch(monkeypatch, memory_sink)
        seed_full_chain(memory_sink)
        del memory_sink.objects[
            keys.embedding_version_claim("all-minilm-l6-v2", "minilm", "v1")
        ]
        assert promote_main(monkeypatch, []) == 0
        assert "resolution: BROKEN" in capsys.readouterr().out

    def test_bootstrap_create_then_conflict(
        self, monkeypatch, capsys, memory_sink: MemorySink
    ) -> None:
        self._patch(monkeypatch, memory_sink)
        assert promote_main(monkeypatch, ["--bootstrap", "--apply"]) == 0
        assert keys.state_pointer() in memory_sink.objects
        assert promote_main(monkeypatch, ["--bootstrap", "--apply"]) == 1
        assert "REFUSED" in capsys.readouterr().out

    def test_bootstrap_and_from_file_are_exclusive(self, monkeypatch) -> None:
        self._patch(monkeypatch, MemorySink())
        with pytest.raises(SystemExit) as exc:
            promote_main(monkeypatch, ["--bootstrap", "--from-file", "x.json"])
        assert exc.value.code == 2

    def test_refusal_exits_1_and_leaves_pointer(
        self, monkeypatch, capsys, tmp_path, memory_sink: MemorySink
    ) -> None:
        self._patch(monkeypatch, memory_sink)
        seed_full_chain(memory_sink)
        target = self._target_file(tmp_path, self._v9_target_doc())
        assert promote_main(monkeypatch, ["--from-file", target, "--apply"]) == 1
        out = capsys.readouterr().out
        assert "REFUSED" in out
        assert memory_sink.objects[keys.state_pointer()] == pointer.canonical_bytes(
            pointer_doc()
        ), "a refused promotion must not write"

    def test_override_applies_and_logs_loudly(
        self, monkeypatch, capsys, tmp_path, memory_sink: MemorySink
    ) -> None:
        self._patch(monkeypatch, memory_sink)
        seed_full_chain(memory_sink)
        expected_previous = content_hash(pointer_doc())
        target = self._target_file(tmp_path, self._v9_target_doc())
        assert (
            promote_main(
                monkeypatch, ["--from-file", target, "--apply", "--allow-incomplete"]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "OVERRIDE" in out
        live = json.loads(memory_sink.objects[keys.state_pointer()])
        assert live["previous"] == expected_previous
        assert live["tiers"]["cleaning"]["version"] == "v9"


def part_keys_for(sink: MemorySink) -> list[str]:
    """The sink's embedding part keys (the gate's shared enumeration)."""
    return [
        key
        for key in sink.objects
        if key.startswith("lake/embeddings/") and key.endswith(".parquet")
    ]


class TestCheckProductionPointer:
    """The chain cross-check: rows, commands, drift."""

    def test_absent_pointer_reports_bootstrap(self, memory_sink: MemorySink) -> None:
        ok, rows = check_production_pointer(memory_sink, [], None)
        assert ok is False
        assert any("--bootstrap --apply" in r for r in rows)

    def test_agreement_is_ok_and_quiet(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink)
        ok, rows = check_production_pointer(
            memory_sink, part_keys_for(memory_sink), None
        )
        assert ok is True
        assert rows == []

    def test_live_unclaimed_prefix_is_informational(
        self, memory_sink: MemorySink
    ) -> None:
        seed_full_chain(memory_sink)
        part_keys = part_keys_for(memory_sink)
        part_keys.append(
            keys.embedding_part("all-minilm-l6-v2", "minilm", "x-scratch", RUN_ID, 1)
        )
        ok, rows = check_production_pointer(memory_sink, part_keys, None)
        assert ok is True
        assert any(row.startswith("INFO") for row in rows)

    def test_partless_pointer_stack_is_error(self, memory_sink: MemorySink) -> None:
        seed_full_chain(memory_sink, with_parts=False)
        ok, rows = check_production_pointer(memory_sink, [], None)
        assert ok is False
        assert any("partless" in r for r in rows)

    def test_cleaning_partless_is_error(self, memory_sink: MemorySink) -> None:
        # The files-only branch follows the same build-then-flip rule.
        seed_full_chain(memory_sink)
        del memory_sink.objects[
            keys.artifact_part("cleaned_text", "llm", "v1", RUN_ID, 1)
        ]
        ok, rows = check_production_pointer(
            memory_sink, part_keys_for(memory_sink), None
        )
        assert ok is False
        assert any(r.startswith("ERROR pointer cleaning stack") for r in rows)

    def test_catalog_drift_is_error_with_reprovision_command(
        self, bucket_env, memory_sink: MemorySink
    ) -> None:
        seed_full_chain(memory_sink)
        # The catalog (projection enums) still names an older reality: the
        # pointer's stack is invisible to Athena.
        athena = FakeAthena(["model_segment,variant,version"])
        ok, rows = check_production_pointer(
            memory_sink, part_keys_for(memory_sink), athena
        )
        assert ok is False
        assert any("re-provision" in r for r in rows)

    def test_catalog_agreement_passes(
        self, bucket_env, memory_sink: MemorySink
    ) -> None:
        seed_full_chain(memory_sink)
        athena = FakeAthena(
            [
                "model_segment,variant,version",
                "all-minilm-l6-v2,minilm,v1",
            ]
        )
        ok, rows = check_production_pointer(
            memory_sink, part_keys_for(memory_sink), athena
        )
        assert ok is True
        assert rows == []
