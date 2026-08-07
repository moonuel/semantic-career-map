#!/usr/bin/env python3
"""Dump raw and clean text for diffs.

Usage:
    python scripts/002_boilerplate/diff_raw_vs_clean.py              # concatenate all postings into one diff pair
    python scripts/002_boilerplate/diff_raw_vs_clean.py <posting-id>  # single posting diff
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

JOBS_PATH = Path("data/jobs.json")
KEEP_SECTIONS = {"about_role", "responsibilities", "qualifications", "nice_to_have"}
RAW_PATH = Path("data/.diff_raw.txt")
CLEAN_PATH = Path("data/.diff_clean.txt")


def build_clean_text(sections: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ["about_role", "responsibilities", "qualifications", "nice_to_have"]:
        val = sections.get(key, "").strip()
        if val:
            parts.append(val)
    return "\n\n".join(parts)


def write_single(job: dict) -> None:
    """Write one posting to the diff pair files."""
    raw = job["raw_full_text"]
    clean = build_clean_text(job["sections"])
    RAW_PATH.write_text(raw, encoding="utf-8")
    CLEAN_PATH.write_text(clean, encoding="utf-8")
    excluded = sorted(set(job["sections"]) - KEEP_SECTIONS)
    print(f"Posting: {job['id']}")
    print(f"Company: {job['company']}")
    print(f"Title:   {job['title_raw']}")
    print(f"Excluded: {excluded if excluded else '(none)'}")
    print(
        f"Size:    {len(raw)} chars → {len(clean)} chars ({len(raw) - len(clean)} removed)"
    )
    print()
    print(f"Wrote:   {RAW_PATH}")
    print(f"Wrote:   {CLEAN_PATH}")
    print()
    print("To diff in VS Code:")
    print(
        "  Select both files in Explorer (Ctrl+click) → right-click → 'Compare Selected'"
    )


def write_all(jobs: list[dict]) -> None:
    """Concatenate all postings into one diff pair, separated by headers."""
    raw_parts: list[str] = []
    clean_parts: list[str] = []

    for job in jobs:
        header = f"\n{'─' * 70}\n## {job['id']} | {job['company']} | {job['title_raw']}\n{'─' * 70}\n"
        raw = job["raw_full_text"]
        clean = build_clean_text(job["sections"])
        excluded = sorted(set(job["sections"]) - KEEP_SECTIONS)

        if raw == clean:
            header += (
                "  (no boilerplate sections detected — raw and clean are identical)\n"
            )
        else:
            header += (
                f"  Excluded sections: {excluded if excluded else '(none)'}\n"
                f"  Raw: {len(raw)} chars → Clean: {len(clean)} chars "
                f"({len(raw) - len(clean)} removed)\n"
            )

        raw_parts.append(header)
        raw_parts.append(raw)
        clean_parts.append(header)
        clean_parts.append(clean)

        n_removed = len(raw) - len(clean)
        excluded_label = ",".join(excluded) if excluded else "(none)"
        print(
            f"{job['id']:<50s} {len(raw):>5d} → {len(clean):>5d}  {-n_removed:>+5d}  {excluded_label}"
        )

    RAW_PATH.write_text("\n".join(raw_parts), encoding="utf-8")
    CLEAN_PATH.write_text("\n".join(clean_parts), encoding="utf-8")

    total_raw = sum(len(j["raw_full_text"]) for j in jobs)
    total_clean = sum(len(build_clean_text(j["sections"])) for j in jobs)
    n_changed = sum(
        1 for j in jobs if j["raw_full_text"] != build_clean_text(j["sections"])
    )

    print()
    print(
        f"{len(jobs)} postings — {n_changed} changed, {len(jobs) - n_changed} unchanged"
    )
    print(
        f"Total: {total_raw} chars → {total_clean} chars ({total_raw - total_clean} removed)"
    )
    print()
    print(f"Wrote:   {RAW_PATH}")
    print(f"Wrote:   {CLEAN_PATH}")
    print()
    print("To diff in VS Code:")
    print(
        "  Select both files in Explorer (Ctrl+click) → right-click → 'Compare Selected'"
    )


def main() -> None:
    with open(JOBS_PATH, encoding="utf-8") as f:
        jobs = json.load(f)

    if len(sys.argv) >= 2:
        target = sys.argv[1]
        job = next((j for j in jobs if j["id"] == target), None)
        if job is None:
            print(f"Posting '{target}' not found.")
            available = "\n  ".join(j["id"] for j in jobs)
            print(f"\nAvailable:\n  {available}")
            sys.exit(1)
        write_single(job)
    else:
        write_all(jobs)


if __name__ == "__main__":
    main()
