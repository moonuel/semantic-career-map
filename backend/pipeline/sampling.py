"""Seeded, source-stratified sampling for cleaning tuning runs (D12).

The tuning-time pin behind the ``subset-sampling`` capability: prompt
iterations must sweep a small, *pinned* subset of the raw layer instead of
the full corpus (D2's cost-control argument — a full sweep per configuration
at ~D8/D5 prices belongs to final configurations, not iteration loops).
Production cleaning-run selection is **not** this module: it is the
``select_uncleaned`` anti-join through the SQL layer, rendered by
:mod:`backend.pipeline.selection` (D12). Both producers emit the same
canonical manifest format, so the cleaning runner (task 3.7) consumes either
by path.

Determinism contract: two runs with the same parameters over the same raw
layer produce **byte-identical** manifests. That holds because candidate
lists are ordered by object key before any RNG touches them, the seeded
``random.Random`` draws are the only stochastic step, and the manifest is
serialized with sorted keys and no timestamps.

Selection spreads across distinct employers where available: within a length
band, a first greedy pass picks one posting per employer before a second
filling pass ignores employer identity. ``linkedin`` payloads carry neither
``source_id`` nor ``employer`` (design.md shape 3) — those fields are empty
strings there and the employer pass cannot distinguish them.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from dotenv import load_dotenv

from backend.storage import keys
from backend.storage.raw import RawPayload, RawStore, fetch_all
from backend.storage.store import ObjectStore

# Length-band thresholds separating short / medium / long postings, in
# characters of the posting's text field. Live-lake medians sit at 5–11k with
# p90 under 14k (exp 009), so these three bands are occupied everywhere;
# adjustable via the CLI.
DEFAULT_BANDS: tuple[int, ...] = (5000, 15000)

# Text fields of the three raw-payload shapes, probed in order (shape 1
# description_html, shape 2 description, shape 3 content).
TEXT_FIELDS: tuple[str, ...] = ("description_html", "description", "content")


def band_of(length: int, bands: Sequence[int]) -> int:
    """Return the band index a text length falls into.

    ``bands`` are ascending thresholds; band ``i`` is
    ``bands[i-1] <= length < bands[i]`` (band 0 starts at -inf).

    Args:
        length: Text length in characters.
        bands: Ascending threshold sequence.

    Returns:
        The band index.
    """
    for index, threshold in enumerate(bands):
        if length < threshold:
            return index
    return len(bands)


def _text(payload: RawPayload) -> str:
    """Return the payload's text under its shape's field (empty if absent)."""
    for field in TEXT_FIELDS:
        value = payload.data.get(field)
        if value is not None:
            return value if isinstance(value, str) else ""
    return ""


def _candidate(payload: RawPayload, bands: Sequence[int]) -> dict[str, Any]:
    """Render one eligible candidate entry for the manifest."""
    return {
        "source": payload.source,
        "source_hash": payload.source_hash,
        "source_id": payload.data.get("source_id") or "",
        "employer": payload.data.get("employer") or "",
        "key": payload.key,
        "band": band_of(len(_text(payload)), bands),
        "length_chars": len(_text(payload)),
    }


def _draw_band(
    candidates: list[dict[str, Any]], quota: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Draw up to ``quota`` candidates from one band, employers-diverse first.

    Args:
        candidates: One band's candidates for one source, in stable order.
        quota: Desired draws (may exceed availability; draws all).
        rng: The seeded RNG.

    Returns:
        The drawn candidates.
    """
    if len(candidates) <= quota:
        return list(candidates)
    order = list(candidates)
    rng.shuffle(order)
    chosen: list[dict[str, Any]] = []
    seen_employers: set[str] = set()
    # First pass: one posting per distinct employer where employer is known.
    for candidate in order:
        if len(chosen) >= quota:
            break
        employer = candidate["employer"]
        if employer and employer not in seen_employers:
            seen_employers.add(employer)
            chosen.append(candidate)
    # Second pass: fill the rest (already-chosen excluded).
    chosen_keys = {c["key"] for c in chosen}
    for candidate in order:
        if len(chosen) >= quota:
            break
        if candidate["key"] not in chosen_keys:
            chosen.append(candidate)
    return chosen


def _allocate(quota: int, band_sizes: list[int]) -> list[int]:
    """Spread a source's quota across its non-empty bands.

    Round-robin over the densest bands (index keeps ties stable): every pass
    hands one unit to each band with capacity left, so a fair quota spreads
    evenly and a scarce quota prefers dense bands, capped at availability.

    Args:
        quota: Total desired draws for the source.
        band_sizes: Candidate count per band index (0 = empty band).

    Returns:
        Per-band drawing quotas (sum ≤ quota; a size-0 band gets 0).

    Raises:
        ValueError: if ``quota`` is negative.
    """
    if quota < 0:
        raise ValueError(f"quota {quota!r} must be non-negative")
    filled = [i for i, size in enumerate(band_sizes) if size > 0]
    alloc = [0] * len(band_sizes)
    if not filled:
        return alloc
    order = sorted(filled, key=lambda i: (-band_sizes[i], i))
    assigned = 0
    while assigned < quota:
        progressed = False
        for i in order:
            if assigned >= quota:
                break
            if alloc[i] < band_sizes[i]:
                alloc[i] += 1
                assigned += 1
                progressed = True
        if not progressed:
            break
    return alloc


def build_manifest(
    store: RawStore,
    *,
    per_source: int,
    bands: Sequence[int],
    seed: int,
    sources: Iterable[str] | None = None,
    workers: int = 16,
) -> dict[str, Any]:
    """Build the tuning-sample manifest dict (not yet serialized).

    Args:
        store: The object store to read from.
        per_source: Quota of postings per source partition.
        bands: Ascending length-band thresholds.
        seed: Seed for the deterministic draw.
        sources: Source partitions to sample from; all allowlisted when None.
        workers: Concurrent fetch bound per source.

    Returns:
        Manifest dict: parameters plus the chosen postings, sorted by
        ``(source, source_hash)``.

    Raises:
        ValueError: if ``per_source`` is negative.
    """
    if per_source < 0:
        raise ValueError(f"per_source {per_source!r} must be non-negative")
    source_list = sorted(sources) if sources is not None else sorted(keys.RAW_SOURCES)
    rng = random.Random(seed)
    postings: list[dict[str, Any]] = []
    for source in source_list:
        fetched = fetch_all(store, source, workers=workers)
        candidates = [_candidate(p, bands) for p in fetched if p is not None]
        by_band: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            by_band[candidate["band"]].append(candidate)
        alloc = _allocate(
            per_source, [len(by_band.get(i, [])) for i in range(len(bands) + 1)]
        )
        for band_index, quota in enumerate(alloc):
            if quota:
                postings.extend(_draw_band(by_band[band_index], quota, rng))
    postings.sort(key=lambda c: (c["source"], c["source_hash"]))
    return {
        "params": {"per_source": per_source, "bands": list(bands), "seed": seed},
        "postings": postings,
    }


def serialize_manifest(manifest: dict[str, Any]) -> bytes:
    """Serialize a manifest to its canonical bytes (byte-stable across runs).

    The single manifest producer contract shared with the SQL-selection
    adapter (:mod:`backend.pipeline.selection`).

    Args:
        manifest: The :func:`build_manifest` (or selection) result.

    Returns:
        Canonical JSON bytes terminated by a newline.
    """
    return (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sample_bytes(
    store: RawStore,
    *,
    per_source: int,
    bands: Sequence[int],
    seed: int,
    sources: Iterable[str] | None = None,
    workers: int = 16,
) -> bytes:
    """Sample the raw layer and return the pinned manifest bytes.

    Two calls with identical parameters return identical bytes.

    Args:
        store: The object store to read from.
        per_source: Quota of postings per source partition.
        bands: Ascending length-band thresholds.
        seed: Seed for the deterministic draw.
        sources: Source partitions to sample from; all allowlisted when None.
        workers: Concurrent fetch bound per source.

    Returns:
        The canonical manifest bytes.
    """
    return serialize_manifest(
        build_manifest(
            store,
            per_source=per_source,
            bands=bands,
            seed=seed,
            sources=sources,
        )
    )


def main() -> None:
    """CLI entry point: write a pinned tuning-sample manifest.

    ``--output`` is the manifest path (configurable output path per
    AGENTS.md); a per-source summary prints to stdout.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Draw a seeded, source-stratified tuning sample over the "
        "raw lake (production selection is the SQL anti-join — D12)."
    )
    parser.add_argument(
        "--per-source", type=int, required=True, help="Quota per source."
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic draw seed.")
    parser.add_argument(
        "--bands",
        type=str,
        default=",".join(str(t) for t in DEFAULT_BANDS),
        help="Comma-separated ascending length-band thresholds (chars).",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(keys.RAW_SOURCES),
        help="Restrict sampling to one source partition (repeatable).",
    )
    parser.add_argument("--output", required=True, help="Manifest path to write.")
    args = parser.parse_args()

    bands = tuple(int(t) for t in args.bands.split(","))
    store = ObjectStore.from_env()
    data = sample_bytes(
        store,
        per_source=args.per_source,
        bands=bands,
        seed=args.seed,
        sources=args.source,
    )
    with open(args.output, "wb") as f:
        f.write(data)
    manifest = json.loads(data)  # the written bytes are the summary source
    counts: defaultdict[str, int] = defaultdict(int)
    for posting in manifest["postings"]:
        counts[posting["source"]] += 1
    for source in sorted(counts):
        print(f"{source}: {counts[source]}")
    print(f"total: {sum(counts.values())} -> {args.output}")


if __name__ == "__main__":
    main()
