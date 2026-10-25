"""Port curation results from the golden workpad into data/golden_cleaned.json.

Phase 1.4c — Experiment 008, Step 2c. The golden set was curated in the
workpad data/008_golden_workpad.md (39 sections, one per posting) instead
of the draft_golden.py CLI loop: the pad was stripped in place and
finalized in a user diff pass, so the pad itself is the sign-off record.
This script is then purely mechanical — it parses `# <posting_id>`
sections and upserts them into data/golden_cleaned.json as
{posting_id, source: "manual", text}.

Existing golden entries always win (signed-off text is immutable). Pad
sections for already-golden postings are reported as skipped with a
differs Y/N flag so a supersession can be done deliberately by deleting
the golden entry and re-running.

Run with --dry first (default) to inspect the report; add --write to
actually write data/golden_cleaned.json.

Usage:
    python scripts/008_boilerplate_only_cleaning/pad_to_golden.py            # dry
    python scripts/008_boilerplate_only_cleaning/pad_to_golden.py --write    # write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DATA_DIR = Path(__file__).resolve().parent / "data"
PAD_PATH = SCRIPT_DATA_DIR / "008_golden_workpad.md"
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"  # shared golden set


def parse_workpad(pad_path: Path) -> dict[str, str]:
    """Parse `# <posting_id>` sections into {posting_id: cleaned_text}."""
    sections: dict[str, str] = {}
    current_id: str | None = None
    lines: list[str] = []
    for line in pad_path.read_text().splitlines():
        if line.startswith("# "):
            if current_id is not None:
                sections[current_id] = "\n".join(lines).strip()
            current_id = line[2:].strip()
            lines = []
        elif current_id is not None:
            lines.append(line)
    if current_id is not None:
        sections[current_id] = "\n".join(lines).strip()
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pad", type=Path, default=PAD_PATH, help="Workpad to read from"
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=GOLDEN_PATH,
        help="Golden file to write (default: data/golden_cleaned.json)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the golden file (default: dry run, report only)",
    )
    args = parser.parse_args()

    sections = parse_workpad(args.pad)
    golden_path: Path = args.golden
    golden = json.loads(golden_path.read_text()) if golden_path.exists() else []
    golden_by_id = {g["posting_id"]: g["text"] for g in golden}

    added: list[str] = []
    skipped: list[tuple[str, bool]] = []
    for posting_id, text in sections.items():
        if posting_id in golden_by_id:
            skipped.append((posting_id, text.strip() == golden_by_id[posting_id]))
        else:
            added.append(posting_id)
    in_golden_not_pad = sorted(set(golden_by_id) - set(sections))

    print(f"Workpad sections: {len(sections)}")
    print(f"Golden entries before: {len(golden)}")
    print(f"To add: {len(added)}")
    for posting_id in added:
        print(f"  + {posting_id}")
    print(f"Skipped (golden exists): {len(skipped)}")
    for posting_id, same in skipped:
        print(f"  = {posting_id} (pad version identical: {'Y' if same else 'N'})")
    print(f"Golden ids absent from pad: {len(in_golden_not_pad)}")
    for posting_id in in_golden_not_pad:
        print(f"  ! {posting_id}")

    if not args.write:
        print("\nDry run — nothing written. Re-run with --write to apply.")
        return

    for posting_id, text in sections.items():
        if posting_id in golden_by_id:
            continue
        golden.append({"posting_id": posting_id, "source": "manual", "text": text})
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nWrote {len(golden)} entries → {golden_path}")


if __name__ == "__main__":
    main()
