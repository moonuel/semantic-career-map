"""Build the 39-posting working corpus for Exp 008.

Phase 1.4c — Experiment 008, Step 2. Merges the 27 postings migrated into
data/jobs.json (raw_full_text, role_category labels) with the 12 md-only
postings in data/selected-job-postings/ that the earlier migration never
loaded. md-only postings get raw text read straight from the .md file and
stay role-unlabeled (role_category null) until hand-labeled.

Writes scripts/008_boilerplate_only_cleaning/data/008_corpus.json — a
job-local artifact; jobs.json is never modified.

Usage:
    python scripts/008_boilerplate_only_cleaning/build_corpus.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
JOBS_PATH = PROJECT_ROOT / "data" / "jobs.json"
MD_DIR = PROJECT_ROOT / "data" / "selected-job-postings"
CORPUS_PATH = SCRIPT_DATA_DIR / "008_corpus.json"


def build_corpus() -> list[dict]:
    """Merge jobs.json postings with md-only postings into one sorted list."""
    with open(JOBS_PATH) as f:
        jobs = {j["id"]: j for j in json.load(f)}

    postings = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        posting_id = md_path.stem
        job = jobs.get(posting_id)
        raw_text = md_path.read_text()
        if job is not None:
            source = "jobs.json"
            # md content and jobs.json raw_full_text are strip-equal for the
            # 27 migrated postings; prefer jobs.json (the 003 pipeline read it).
            if job.get("raw_full_text", "").strip() == raw_text.strip():
                raw_text = job["raw_full_text"]
            role = job.get("role_category")
            company = job.get("company")
            title = job.get("title_raw")
            clean_text = job.get("clean_text")
        else:
            source = "md"
            role = None
            company = None
            title = None
            clean_text = None
        postings.append(
            {
                "posting_id": posting_id,
                "source": source,
                "source_file": md_path.name,
                "role_category": role,
                "company": company,
                "title_raw": title,
                "raw_full_text": raw_text,
                "clean_text": clean_text,
            }
        )
    return postings


def main() -> None:
    postings = build_corpus()
    n_jobs = sum(p["source"] == "jobs.json" for p in postings)
    n_md = sum(p["source"] == "md" for p in postings)
    artifact = {
        "experiment": "008_boilerplate_only_cleaning",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_postings": len(postings),
        "n_from_jobs_json": n_jobs,
        "n_from_md_only": n_md,
        "postings": postings,
    }
    with open(CORPUS_PATH, "w") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    print(
        f"Wrote {len(postings)} postings ({n_jobs} jobs.json + {n_md} md-only) → {CORPUS_PATH}"
    )
    unlabeled = [p["posting_id"] for p in postings if p["role_category"] is None]
    print(f"Role-unlabeled ({len(unlabeled)}): {unlabeled}")


if __name__ == "__main__":
    main()
