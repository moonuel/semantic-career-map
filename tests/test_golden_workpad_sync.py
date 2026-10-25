"""Guard: the committed golden JSON stays in lockstep with the workpad.

``data/golden_cleaned.json`` is the eval ground truth, generated from the
sign-off workpad (``scripts/008_boilerplate_only_cleaning/data/008_golden_workpad.md``)
via ``pad_to_golden.py`` — the workpad is the sole edit source (the 008
report's golden-set flow). Editing either copy alone silently forks the
ground truth; this test fails on drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAD_SCRIPT = (
    PROJECT_ROOT / "scripts" / "008_boilerplate_only_cleaning" / "pad_to_golden.py"
)
PAD_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "008_boilerplate_only_cleaning"
    / "data"
    / "008_golden_workpad.md"
)
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_cleaned.json"


def _parse_workpad() -> dict[str, str]:
    """Load the (stdlib-only) pad parser and parse the committed workpad."""
    spec = importlib.util.spec_from_file_location("pad_to_golden", PAD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_workpad(PAD_PATH)


def test_golden_json_matches_workpad() -> None:
    """Every golden entry must equal its workpad section — no silent forks.

    Golden edited directly (bypassing the workpad) or a workpad edit not
    ported with ``pad_to_golden.py`` fails here with the drifted ids.
    """
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    pad = _parse_workpad()
    golden_texts = {g["posting_id"]: g["text"] for g in golden}
    drift: dict[str, str] = {}
    for posting_id, text in golden_texts.items():
        if posting_id not in pad:
            drift[posting_id] = "missing from workpad"
        elif pad[posting_id] != text:
            drift[posting_id] = "text drift (pad vs golden)"
    assert not drift, f"golden set forked from the workpad: {drift}"
