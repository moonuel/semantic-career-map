"""Recovery tests — delta-selection idempotence end to end and
concurrent-run reconciliation.

The selection double here is stronger than the roundtrip's static page: it
recomputes the ``select_unembedded`` anti-join from the store's own part
contents (the cache identity lives in the data), so a run observes whatever
the lake holds at selection time. ``freeze()`` pins a shared page — the
race's shared invisible state — reproducing the concurrent-run duplicate
harm that Layer 2 hit live (309 orphan parts) without real concurrency.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow.parquet as pq
import pytest

from backend.pipeline.embedding import driver, reconcile
from backend.pipeline.embedding.config import (
    EmbeddingConfig,
    configuration_hash,
    dataset_id,
)
from backend.pipeline.embedding.encoder import StubEncoder
from backend.pipeline.embedding.reconcile import (
    heal_orphan_checkpoints,
    reconcile_configuration,
)
from backend.pipeline.embedding.writer import build_flush_checkpoint
from backend.storage import keys
from tests.conftest import MemorySink


@pytest.fixture(autouse=True)
def lake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store-rooted env pieces results_location/athena helpers read."""
    monkeypatch.setenv("STORE_S3_BUCKET", "lake")
    monkeypatch.setenv("PIPELINE_VERSION", "emb-test-1")


def _h(stem: int) -> str:
    return f"{stem}" * 64


ANTI_SQL = (
    "SELECT a.source_hash, a.cleaned_text AS variant_text "
    "FROM artifacts_cleaned a LEFT JOIN postings_embeddings e "
    "ON e.source_hash = a.source_hash AND e.configuration_hash = :config "
    "WHERE e.source_hash IS NULL"
)

# Backdated wall clock: older than the reconcile in-flight guard.
PAST = datetime(2026, 1, 1, tzinfo=UTC)

# Fixed run ids for synthetic lake states (run-minted part leaf pairing).
RUN_YOUNG = "01020304-0000-7f7e-a000-0000000000aa"
RUN_A = "01020304-0000-7f7e-a000-0000000000a1"
RUN_B = "01020304-0000-7f7e-a000-0000000000b1"
RUN_C = "01020304-0000-7f7e-a000-0000000000c1"


def _config(**kw) -> EmbeddingConfig:
    defaults = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "rev-1",
        "model_segment": "all-minilm-l6-v2",
        "variant": "minilm",
        "text_variant": "variant_text",
        "batch_size": 2,
        "version": "v1",
        "sql_text": ANTI_SQL,
    }
    defaults.update(kw)
    return EmbeddingConfig(**defaults)


class AntiJoinAthena:
    """Selection double recomputing the anti-join from store contents."""

    def __init__(
        self, store: MemorySink, config: EmbeddingConfig, cleaned: list[tuple[str, str]]
    ) -> None:
        self._store = store
        self._digest = configuration_hash(config)
        self._prefix = reconcile.version_prefix(config)
        self.cleaned = list(cleaned)
        self._frozen_page: list[list[str]] | None = None
        self.executed: list[str] = []

    def start_query_execution(self, *, QueryString: str, **kw) -> dict:
        self.executed.append(QueryString)
        return {"QueryExecutionId": f"q-{len(self.executed):04d}"}

    def get_query_execution(self, *, QueryExecutionId: str) -> dict:
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, *, QueryExecutionId: str, NextToken=None) -> dict:
        page = self._page() if not self.frozen else self._frozen_page
        return {
            "ResultSet": {
                "Rows": [
                    {"Data": [{"VarCharValue": cell} for cell in row]} for row in page
                ]
            }
        }

    @property
    def frozen(self) -> bool:
        return self._frozen_page is not None

    def freeze(self) -> None:
        """Pin the current anti-join view (the race's shared selection)."""
        self._frozen_page = self._page()

    def thaw(self) -> None:
        """Return to recomputing the anti-join per selection."""
        self._frozen_page = None

    def _page(self) -> list[list[str]]:
        embedded: set[str] = set()
        for key in self._store.list_keys(self._prefix):
            if not key.endswith(".parquet"):
                raise AssertionError(f"unexpected object in prefix: {key!r}")
            table = pq.read_table(io.BytesIO(self._store.get(key)))
            for src, cfg in zip(
                table.column("source_hash").to_pylist(),
                table.column("configuration_hash").to_pylist(),
            ):
                if cfg == self._digest:
                    embedded.add(src)
        selected = [
            (source, text) for source, text in self.cleaned if source not in embedded
        ]
        selected.sort(key=lambda pair: pair[0])
        return [["source_hash", "variant_text"]] + [
            [source, text] for source, text in selected
        ]


def _part_keys(store: MemorySink, config: EmbeddingConfig) -> list[str]:
    prefix = reconcile.version_prefix(config)
    return [key for key in store.list_keys(prefix) if key.endswith(".parquet")]


def _hash_counts(store: MemorySink, config: EmbeddingConfig) -> Counter[str]:
    digest = configuration_hash(config)
    counts: Counter[str] = Counter()
    for key in _part_keys(store, config):
        table = pq.read_table(io.BytesIO(store.get(key)))
        for src, cfg in zip(
            table.column("source_hash").to_pylist(),
            table.column("configuration_hash").to_pylist(),
        ):
            if cfg == digest:
                counts[src] += 1
    return counts


def _row_run_ids(store: MemorySink, key: str) -> list[str]:
    table = pq.read_table(io.BytesIO(store.get(key)))
    return table.column("run_id").to_pylist()


def _race(
    store: MemorySink, at: datetime, stems: list[int]
) -> tuple[EmbeddingConfig, ...]:
    """Two frozen-selection runs of one configuration (the race's shape).

    Returns ``(config, first, second, second_parts)`` — the second run's
    part keys captured before any healing pass can delete them.
    """
    config = _config()
    athena = AntiJoinAthena(store, config, _cleaned(*stems))
    encoder = StubEncoder(dimension=8)
    athena.freeze()  # both runs stream the same pre-commit selection
    first = driver.run_embedding(store, athena, encoder, config, now=lambda: at)
    second = driver.run_embedding(store, athena, encoder, config, now=lambda: at)
    athena.thaw()
    second_parts = [key for key in _part_keys(store, config) if second.run_id in key]
    assert first.run_id != second.run_id
    assert len(second_parts) == 1  # buffered: one part per racing run
    return config, first, second, second_parts


def _cleaned(*stems: int) -> list[tuple[str, str]]:
    return [(_h(stem), f"text {stem}") for stem in stems]


def _vectors(n: int, fill: float) -> np.ndarray:
    return np.full((n, 8), fill, dtype=np.float32)


# --- 3.1 delta-selection idempotence, end to end -----------------------------


def test_delta_selection_embeds_only_new_postings(memory_sink: MemorySink) -> None:
    config = _config()
    athena = AntiJoinAthena(memory_sink, config, _cleaned(0, 1, 2, 3, 4))
    encoder = StubEncoder(dimension=8)
    first = driver.run_embedding(memory_sink, athena, encoder, config)
    assert (first.selected, first.embedded) == (5, 5)

    athena.cleaned.extend(_cleaned(5, 6))  # new cleanings arrive
    second = driver.run_embedding(memory_sink, athena, encoder, config)
    assert (second.selected, second.embedded) == (2, 2) and not second.zero_delta

    # only the new postings were encoded — the embedded corpus never re-encodes
    assert encoder.calls[-1] == ["text 5", "text 6"]
    encoded = Counter(text for call in encoder.calls for text in call)
    assert all(encoded[f"text {stem}"] == 1 for stem in range(7))

    counts = _hash_counts(memory_sink, config)
    assert len(counts) == 7 and all(count == 1 for count in counts.values())

    # current manifest semantics: one addressed object per dataset id — the
    # latest run's view overwrites the earlier run's (lossy until the
    # flush-checkpoint layer lands)
    manifest_key = keys.manifest_dataset(dataset_id(config))
    assert first.dataset_manifest_key == second.dataset_manifest_key == manifest_key
    manifest = json.loads(memory_sink.get(manifest_key))
    assert manifest["run_id"] == second.run_id
    assert manifest["record_count"] == 2
    assert manifest["parts"] == second.part_keys


def test_fully_embedded_delta_lands_noop(memory_sink: MemorySink) -> None:
    config = _config()
    athena = AntiJoinAthena(memory_sink, config, _cleaned(0, 1, 2))
    encoder = StubEncoder(dimension=8)
    driver.run_embedding(memory_sink, athena, encoder, config)

    before = set(memory_sink.objects)
    noop = driver.run_embedding(memory_sink, athena, encoder, config)
    assert noop.zero_delta and noop.selected == 0 and noop.embedded == 0
    assert noop.part_keys == [] and noop.dataset_manifest_key is None
    assert set(memory_sink.objects) == before  # zero writes, healthy no-op


# --- 3.2 concurrent-run idempotence ------------------------------------------


def test_concurrent_runs_duplicate_then_next_run_heals(memory_sink: MemorySink) -> None:
    stems = [0, 1, 2, 3, 4, 5]
    config, first, _second, second_parts = _race(memory_sink, PAST, stems)

    pre = _hash_counts(memory_sink, config)  # the incident's harm, reproduced
    assert len(pre) == len(stems) and all(count == 2 for count in pre.values())

    healed = driver.run_embedding(
        memory_sink,
        AntiJoinAthena(memory_sink, config, _cleaned(*stems)),
        StubEncoder(dimension=8),
        config,
    )
    assert healed.zero_delta and healed.selected == 0  # everything embedded

    report = healed.reconcile
    assert report is not None
    assert report.canonical_run_id == first.run_id  # UUIDv7: earliest mints
    assert report.duplicate_row_count == len(stems)
    assert report.deferred_runs == []
    assert report.rewritten_parts == []
    assert sorted(report.deleted_parts) == sorted(second_parts)
    assert all(first.run_id in key for key in _part_keys(memory_sink, config))
    for key in _part_keys(memory_sink, config):
        assert _row_run_ids(memory_sink, key) == [first.run_id] * len(stems)

    counts = _hash_counts(memory_sink, config)
    assert len(counts) == len(stems) and all(count == 1 for count in counts.values())

    # reconciliation is idempotent — a second pass sees nothing to heal
    again = reconcile_configuration(memory_sink, config)
    assert again.duplicate_row_count == 0 and again.deleted_parts == []
    assert _hash_counts(memory_sink, config) == counts


def test_young_run_deferred_by_reconcile(memory_sink: MemorySink) -> None:
    stems = [0, 1, 2, 3]
    config, first, second, _second_parts = _race(memory_sink, datetime.now(UTC), stems)

    before = _hash_counts(memory_sink, config)
    report = reconcile_configuration(memory_sink, config)
    assert report.deferred_runs == sorted([first.run_id, second.run_id])
    assert report.duplicate_row_count == 0  # nothing touched while in flight
    assert _hash_counts(memory_sink, config) == before  # duplicates persist

    # guard disabled: the same state heals once runs age out
    healed = reconcile_configuration(memory_sink, config, min_run_age=timedelta(0))
    assert healed.duplicate_row_count == len(stems)
    counts = _hash_counts(memory_sink, config)
    assert len(counts) == len(stems) and all(count == 1 for count in counts.values())


def test_reconcile_is_noop_on_clean_lake(memory_sink: MemorySink) -> None:
    config = _config()
    athena = AntiJoinAthena(memory_sink, config, _cleaned(0, 1, 2, 3, 4))
    encoder = StubEncoder(dimension=8)
    ran = driver.run_embedding(memory_sink, athena, encoder, config)
    assert ran.reconcile is not None
    assert ran.reconcile.scanned_parts == 0  # nothing in the lake yet

    before = set(memory_sink.objects)
    report = reconcile_configuration(memory_sink, config, min_run_age=timedelta(0))
    assert report.scanned_parts == 1 and report.duplicate_row_count == 0
    assert report.rewritten_parts == [] and report.deleted_parts == []
    assert set(memory_sink.objects) == before  # zero writes on a clean lake


# --- 3.3/3.4 flush checkpoints + orphan healing -------------------------------


def test_orphan_part_back_fills_on_next_run(memory_sink: MemorySink) -> None:
    config = _config()
    athena = AntiJoinAthena(memory_sink, config, _cleaned(0, 1, 2))
    encoder = StubEncoder(dimension=8)
    crashed = driver.run_embedding(
        memory_sink, athena, encoder, config, now=lambda: PAST
    )  # part landed; pretend the crash hit between the two puts
    orphan_book = keys.manifest_run_checkpoint(crashed.run_id, 1)
    assert orphan_book in set(memory_sink.objects)
    memory_sink.delete(orphan_book)

    healed_run = driver.run_embedding(memory_sink, athena, encoder, config)
    assert healed_run.zero_delta  # selection unaffected — rows were durable
    assert len(healed_run.recovered) == 1
    recovered = healed_run.recovered[0]
    assert recovered["run_id"] == crashed.run_id
    assert recovered["trigger"] == "recovered" and recovered["recovered"] is True
    assert recovered["part_key"] == crashed.part_keys[0]
    assert recovered["row_count"] == 3
    assert recovered["batch_range"] is None  # rows carry no batch telemetry
    back = json.loads(memory_sink.get(orphan_book))
    assert back == recovered  # the book is back on disk
    counts = _hash_counts(memory_sink, config)
    assert len(counts) == 3 and all(count == 1 for count in counts.values())


def test_heal_defers_young_orphan_parts(memory_sink: MemorySink) -> None:
    from backend.pipeline.embedding.writer import part_bytes

    config = _config()
    digest = configuration_hash(config)
    part_key = keys.embedding_part(
        config.model_segment, config.variant, config.version, RUN_YOUNG, 1
    )
    memory_sink.put(
        part_key,
        part_bytes(
            ["a" * 64],
            np.array([[1.0] * 8], dtype=np.float32),
            config,
            RUN_YOUNG,
            datetime.now(UTC),
            digest,
        ),
    )  # fresh stamp, no checkpoint — a live-writer-shaped orphan

    assert heal_orphan_checkpoints(memory_sink, config) == []  # deferred

    recovered = heal_orphan_checkpoints(memory_sink, config, min_run_age=timedelta(0))
    assert len(recovered) == 1 and recovered[0]["row_count"] == 1
    assert keys.manifest_run_checkpoint(RUN_YOUNG, 1) in set(memory_sink.objects)


def test_reconcile_keeps_books_consistent_with_parts(memory_sink: MemorySink) -> None:
    from backend.pipeline.embedding.writer import part_bytes

    config = _config()
    digest = configuration_hash(config)
    created = PAST

    def _seed(run_id: str, hashes: list[str]) -> None:
        part_key = keys.embedding_part(
            config.model_segment, config.variant, config.version, run_id, 1
        )
        memory_sink.put(
            part_key,
            part_bytes(
                hashes, _vectors(len(hashes), 1.0), config, run_id, created, digest
            ),
        )
        checkpoint = build_flush_checkpoint(
            run_id=run_id,
            seq=1,
            trigger="rows",
            part_key=part_key,
            source_hashes=hashes,
            configuration_hash=digest,
            batch_range=["encode-0001", "encode-0001"],
            written_at="x",
        )
        memory_sink.put(
            keys.manifest_run_checkpoint(run_id, 1),
            json.dumps(checkpoint).encode("utf-8"),
        )
        return part_key

    # run A: [h0, h1] canonical; run B: [h1, h2] → h1 dup, h2 unique;
    # run C: [h0, h1] fully duplicate
    _seed(RUN_A, ["0" * 64, "1" * 64])  # canonical run — untouched
    part_b = _seed(RUN_B, ["1" * 64, "2" * 64])
    part_c = _seed(RUN_C, ["0" * 64, "1" * 64])
    book_a = keys.manifest_run_checkpoint(RUN_A, 1)
    book_c = keys.manifest_run_checkpoint(RUN_C, 1)

    report = reconcile_configuration(memory_sink, config, min_run_age=timedelta(0))
    assert report.duplicate_row_count == 3  # h1 twice, h0+h1 once more
    assert report.rewritten_parts == [part_b] and report.deleted_parts == [part_c]

    # run C's part is gone AND its book with it — no dangling checkpoint
    assert book_c not in set(memory_sink.objects)
    assert part_c not in set(memory_sink.objects)
    # run B's rewritten part holds only its unique row; its book retruths it
    book_b = json.loads(memory_sink.get(keys.manifest_run_checkpoint(RUN_B, 1)))
    assert book_b["trigger"] == "reconciled" and book_b["reconciled"] is True
    assert book_b["row_count"] == 1 and book_b["source_hashes"] == ["2" * 64]
    assert book_b["batch_range"] is None
    assert part_b in set(memory_sink.objects)
    # run A untouched, book intact (content-addressed survivor)
    assert json.loads(memory_sink.get(book_a))["row_count"] == 2
    counts = _hash_counts(memory_sink, config)
    assert len(counts) == 3 and all(count == 1 for count in counts.values())
