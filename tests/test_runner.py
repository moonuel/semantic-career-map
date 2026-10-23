"""Integration tests for the cleaning runner (task 3.7, stubbed client).

Covers: clean+cache counts on a failing run; stubbed-call durations in
run records; committed-flush durability across interruption; rerun
performs zero client calls (cache resolution); no-HTML contract via the
captured user messages.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from typing import Any

import pyarrow.parquet as pq

from backend.pipeline import sampling
from backend.pipeline.cleaning import runner
from backend.pipeline.cleaning.llm_client import price_call
from backend.pipeline.cleaning.prompts import (
    BOILERPLATE_ONLY_SYSTEM_PROMPT,
    USER_MESSAGE_TEMPLATE,
)
from backend.storage import keys
from tests.conftest import JournalSink

TAGS = ("gh1", "ab2", "hc3", "lk4", "lv5")
MODEL = "google/gemini-2.5-flash-lite"


class StubClient:
    """Gateway stand-in: canned outputs, call log, scripted failures."""

    def __init__(self, fail_on: dict[str, Exception] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail_on = fail_on or {}

    def complete(
        self, system_prompt: str, user_message: str, *, model: str, **kw: Any
    ) -> tuple[str, dict[str, Any]]:
        time.sleep(0.001)  # nonzero, measurable duration
        self.calls.append({"user_message": user_message, "model": model})
        for needle, exc in self._fail_on.items():
            if needle in user_message:
                raise exc
        for tag in TAGS:
            if tag in user_message:
                return f"cleaned-{tag}", {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
        raise AssertionError("no canned output for " + user_message)


def _post(store: JournalSink, source: str, field: str, body: str) -> dict[str, Any]:
    """Place a raw payload; return its manifest posting row."""
    source_hash = hashlib.sha256(body.encode()).hexdigest()
    store.objects[keys.raw_payload(source, source_hash)] = json.dumps(
        {field: body, "source_id": f"{source}:x:1", "employer": "Acme"}
    ).encode()
    return {
        "source": source,
        "source_hash": source_hash,
        "source_id": f"{source}:x:1",
        "key": keys.raw_payload(source, source_hash),
    }


def _make_manifest(store: JournalSink) -> tuple[str, dict[str, Any]]:
    """Five postings across the three payload shapes; returns (path, dict)."""
    postings = [
        _post(store, "greenhouse", "description_html", "html gh1 <p>boiler</p>"),
        _post(store, "ashby", "description_html", "html ab2 <ul><li>duty</li></ul>"),
        _post(store, "hiringcafe", "description", "plain hc3 duties: python"),
        _post(store, "linkedin", "content", "md lk4 **skills** sql"),
        _post(store, "lever", "description_html", "html lv5 <p>utc-5</p>"),
    ]
    manifest = {"params": {}, "postings": postings}
    path = "/tmp/kilo/manifest-test-five.json"
    with open(path, "wb") as f:
        f.write(sampling.serialize_manifest(manifest))
    return path, manifest


def _run(
    store: JournalSink, path: str, client: StubClient, **kw: Any
) -> dict[str, Any]:
    return runner.run_cleaning(
        store,
        runner.load_manifest(path),
        input_dataset=path,
        model=MODEL,
        system_prompt=BOILERPLATE_ONLY_SYSTEM_PROMPT,
        user_template=USER_MESSAGE_TEMPLATE,
        client=client,
        pipeline_version="dev",
        workers=1,
        pacing_budget=0.0,
        **kw,
    )


def test_first_run_cleans_all_calls_recorded() -> None:
    store = JournalSink()
    path, _ = _make_manifest(store)
    client = StubClient()
    record = _run(store, path, client, flush_rows=2)

    assert record["status"] == "completed"
    assert record["counters"] == {"cleaned": 5, "cache_hits": 0, "failed": 0}
    assert record["latency"]["n_calls"] == 5
    assert all(d > 0 for d in record["latency"]["durations"])
    one_call = price_call(MODEL, {"prompt_tokens": 100, "completion_tokens": 50})
    assert record["cost_usd"] == round(one_call * 5, 8) > 0
    # no-HTML contract: no markup reached the LLM
    for call in client.calls:
        assert "<p>" not in call["user_message"]
        assert call["model"] == MODEL
    # 5 rows @ flush_rows=2 → parts of 2+2+1
    assert len(record["flush_refs"]) == 3
    dataset = json.loads(
        store.get(
            keys.manifest_dataset(f"cleaned_text-llm-v1-{record['run_id']}")
        ).decode()
    )
    assert dataset["record_count"] == 5
    assert dataset["model"] == MODEL


def test_second_run_performs_zero_client_calls() -> None:
    store = JournalSink()
    path, _ = _make_manifest(store)
    _run(store, path, StubClient(), flush_rows=2)
    second = StubClient()
    record = _run(store, path, second, flush_rows=2)
    assert second.calls == []  # ZERO API calls
    assert record["counters"] == {"cleaned": 0, "cache_hits": 5, "failed": 0}
    assert record["flush_refs"] == []


def test_fail_fast_counts_and_identifies_failing_posting() -> None:
    store = JournalSink()
    path, manifest = _make_manifest(store)
    bad_hash = manifest["postings"][2]["source_hash"]  # hiringcafe posting
    client = StubClient(fail_on={"hc3": RuntimeError("stub boom")})
    record = _run(store, path, client, flush_rows=2)
    assert record["status"] == "failed"
    # hc3 fails; lk4 (already dequeued on the single worker) may join and
    # commit before the record merges — scheduler race, 2 or 3 cleaned,
    # never more; the failed posting is identified, never counted
    assert record["counters"]["cleaned"] in {2, 3}
    assert record["counters"]["cache_hits"] == 0
    assert record["counters"]["failed"] == 1
    assert record["failed_source_hashes"] == [bad_hash]
    stored = json.loads(store.get(keys.manifest_run(record["run_id"])).decode())
    assert stored["failed_source_hashes"] == [bad_hash]
    assert stored["status"] == "failed"


def _part_hashes(store: JournalSink, part_key: str) -> set[str]:
    return set(
        pq.read_table(io.BytesIO(store.get(part_key))).column("source_hash").to_pylist()
    )


def test_interrupted_run_keeps_flushes_durable_and_rerun_skips() -> None:
    store = JournalSink()
    path, manifest = _make_manifest(store)
    # SIGINT arrives during the 3rd posting's call; the two remaining
    # postings interrupt too, so nothing can buffer after the abort
    client = StubClient(
        fail_on={
            "hc3": KeyboardInterrupt(),
            "lk4": KeyboardInterrupt(),
            "lv5": KeyboardInterrupt(),
        }
    )
    record = _run(store, path, client, flush_rows=2)
    assert record["status"] == "interrupted"
    # no posting failed → the ledger stays empty (the rerun re-cleans hc3)
    assert record["failed_source_hashes"] == []
    committed: set[str] = set()
    for part_key in record["flush_refs"]:
        committed |= _part_hashes(store, part_key)
    assert committed == {
        manifest["postings"][0]["source_hash"],
        manifest["postings"][1]["source_hash"],
    }

    # rerun: committed rows resolve as cache hits; only the rest re-clean
    second = StubClient()
    record2 = _run(store, path, second, flush_rows=2)
    assert record2["status"] == "completed"
    assert record2["counters"] == {"cleaned": 3, "cache_hits": 2, "failed": 0}
    assert len(second.calls) == 3
    for tag in ("hc3", "lk4", "lv5"):
        assert any(tag in c["user_message"] for c in second.calls)
    for tag in ("gh1", "ab2"):  # durable rows were not re-cleaned
        assert not any(tag in c["user_message"] for c in second.calls)


def test_rerun_after_failure_retries_failed_posting() -> None:
    store = JournalSink()
    path, _ = _make_manifest(store)
    first = _run(
        store, path, StubClient(fail_on={"hc3": RuntimeError("boom")}), flush_rows=2
    )
    assert first["status"] == "failed"

    # rerun: the fixed posting cleans; the failed run's committed rows
    # (2 or 3 depending on the join race) resolve as cache hits
    second = StubClient()
    record2 = _run(store, path, second, flush_rows=2)
    assert record2["status"] == "completed"
    prior = first["counters"]["cleaned"]
    assert record2["counters"] == {
        "cleaned": 3 if prior == 2 else 2,
        "cache_hits": prior,
        "failed": 0,
    }
    assert len(second.calls) == record2["counters"]["cleaned"]
