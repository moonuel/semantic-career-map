"""Fidelity tests: extracted cleaning prompts match the Exp 008 originals.

Task 3.1 verification clause: "prompt text matches original". The test
parses ``scripts/008_boilerplate_only_cleaning/prompts.py`` (the
immutable experiment artifact) and asserts the extracted constants are
byte-equal to the originals' evaluated string values.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from backend.pipeline.cleaning import prompts

ORIGINS = {
    "NEW_SYSTEM_PROMPT": prompts.BOILERPLATE_ONLY_SYSTEM_PROMPT,
    "USER_MESSAGE_TEMPLATE": prompts.USER_MESSAGE_TEMPLATE,
}
ORIGINAL_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "008_boilerplate_only_cleaning"
    / "prompts.py"
)


def _original_literal(name: str) -> Any:
    """Return the evaluated value of a module-level assignment.

    Args:
        name: Assignment target name in the original prompts module.

    Returns:
        The literal's evaluated value.

    Raises:
        AssertionError: if no assignment to ``name`` exists.
    """
    tree = ast.parse(ORIGINAL_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name!r} not found in {ORIGINAL_PATH}")


def test_system_prompt_matches_original_verbatim() -> None:
    assert prompts.BOILERPLATE_ONLY_SYSTEM_PROMPT == _original_literal(
        "NEW_SYSTEM_PROMPT"
    )


def test_user_template_matches_original_verbatim() -> None:
    assert prompts.USER_MESSAGE_TEMPLATE == _original_literal("USER_MESSAGE_TEMPLATE")


def test_template_has_single_raw_full_text_placeholder() -> None:
    assert prompts.USER_MESSAGE_TEMPLATE.count("{raw_full_text}") == 1


def test_system_prompt_is_nonempty_rules_text() -> None:
    text = prompts.BOILERPLATE_ONLY_SYSTEM_PROMPT
    assert len(text) > 1000
    assert "Strip completely:" in text
    assert "Preserve verbatim" in text
