"""LLM-assisted golden-set curation for Exp 008.

Phase 1.4c — Experiment 008, Step 2. For every corpus posting not yet in
data/golden_cleaned.json (the shared golden set, read from the repo-root
data/ like every experiment), generates a boilerplate-only draft with the
new Exp 008 prompt, shows it next to the raw posting, and writes a golden
entry to data/golden_cleaned.json ONLY after an explicit per-posting sign
off ('a' = accept = sign-off). AGENTS.md requires the golden set to be
hand-maintained: drafts are untrusted starting text, never auto-saved
without review. The corpus is read from the experiment-local
scripts/008_boilerplate_only_cleaning/data/008_corpus.json.

The 5 existing golden entries are treated as already done. New entries
carry only ``posting_id`` + ``source: "manual"`` + ``text`` — the
job-context/role-context partitions belong to Exp 005 and are not curated
here.

Usage:
    python scripts/008_boilerplate_only_cleaning/draft_golden.py
    python scripts/008_boilerplate_only_cleaning/draft_golden.py --id bmo-analyst-data-scientist
    python scripts/008_boilerplate_only_cleaning/draft_golden.py --list
    python scripts/008_boilerplate_only_cleaning/draft_golden.py \
        --output /tmp/kilo/golden_dry.json --max 1   # dry pass, no repo writes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
from llm_client import GatewayClient, build_prompt_hash

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = SCRIPT_DATA_DIR / "008_corpus.json"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"  # shared golden set
DEFAULT_MODEL = "openai/gpt-5.4-nano"


def load_inputs() -> tuple[dict[str, str], list[str]]:
    """Return (golden_by_id, ordered pending posting ids)."""
    corpus = json.loads(CORPUS_PATH.read_text())
    golden = json.loads(GOLDEN_PATH.read_text())
    golden_ids = {g["posting_id"] for g in golden}
    pending = [
        p["posting_id"] for p in corpus["postings"] if p["posting_id"] not in golden_ids
    ]
    raw_by_id = {p["posting_id"]: p["raw_full_text"] for p in corpus["postings"]}
    return raw_by_id, pending


def render_entry(posting_id: str, text: str) -> dict:
    return {"posting_id": posting_id, "source": "manual", "text": text}


def upsert_golden(path: Path, entry: dict) -> None:
    """Insert or replace one golden entry on the given file path."""
    golden = json.loads(path.read_text()) if path.exists() else []
    golden = [g for g in golden if g["posting_id"] != entry["posting_id"]]
    golden.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
        f.write("\n")


def interactive_edit() -> str:
    """Read a multi-line golden replacement; terminate with a lone '.' line."""
    print(
        "Edit golden text. Enter each line; finish with a single '.' on its own line:"
    )
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLDEN_PATH,
        help="Golden file to sign off into (default: data/golden_cleaned.json)",
    )
    parser.add_argument(
        "--id", action="append", default=[], help="Restrict to posting id(s)"
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Cap postings in this pass"
    )
    parser.add_argument("--list", action="store_true", help="Show pending ids and exit")
    args = parser.parse_args()

    raw_by_id, pending = load_inputs()
    if args.id:
        pending = [pid for pid in pending if pid in set(args.id)]
    print(
        f"{len(pending)} postings pending curation (out of 39; 5 golden already present)"
    )
    if args.list:
        for pid in pending:
            print(f"  - {pid}")
        return
    if not pending:
        return

    prompt_hash = build_prompt_hash(
        prompts.NEW_SYSTEM_PROMPT, prompts.USER_MESSAGE_TEMPLATE
    )
    client = GatewayClient()

    done = 0
    for i, posting_id in enumerate(pending, start=1):
        if args.max is not None and done >= args.max:
            break
        raw = raw_by_id[posting_id]
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(pending)}] {posting_id}")
        print(f"{'-' * 80}")
        print(raw)
        print(f"{'-' * 80}")
        try:
            draft, _, _ = client.cached_call(
                prompt_hash,
                posting_id,
                args.model,
                prompts.NEW_SYSTEM_PROMPT,
                prompts.USER_MESSAGE_TEMPLATE.format(raw_full_text=raw),
            )
        except Exception as e:  # noqa: BLE001 — every fault class logged by caller
            print(f"Draft failed for {posting_id}: {e} (skipping)")
            continue
        print(
            f"--- DRAFT ({args.model}) — untrusted starting text ----------------------"
        )
        print(draft)
        print("--- END DRAFT ------------------------------------------------------")
        action = (
            input("[a]ccept (sign-off) / [e]dit then sign-off / [s]kip / [q]uit > ")
            .strip()
            .lower()
        )
        if action == "q":
            break
        if action == "s":
            continue
        if action == "a":
            text = draft
        elif action == "e":
            text = interactive_edit()
            if not text:
                print("Empty edit — nothing signed off.")
                continue
        else:
            continue
        upsert_golden(args.output, render_entry(posting_id, text))
        done += 1
        print(f"SIGNED OFF → {args.output} ({posting_id})")

    print(
        f"\nDone: {done} entries signed off this session.",
        f"{len(load_inputs()[1])} still pending.",
    )


if __name__ == "__main__":
    main()
