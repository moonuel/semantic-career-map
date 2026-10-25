"""Tests for the authoritative configuration hash (task 3.3)."""

from __future__ import annotations

import hashlib

from backend.pipeline.cleaning.configuration import configuration_hash


def test_matches_task_33_definition() -> None:
    h = configuration_hash("m", "sys", "tpl")
    assert h == hashlib.sha256(b"m\nsys\ntpl").hexdigest()
    assert len(h) == 64
    assert h == h.lower()


def test_identical_config_yields_identical_hash() -> None:
    assert configuration_hash("m", "sys", "tpl") == configuration_hash(
        "m", "sys", "tpl"
    )


def test_any_field_change_changes_the_hash() -> None:
    base = configuration_hash("m", "sys", "tpl")
    assert configuration_hash("m2", "sys", "tpl") != base
    assert configuration_hash("m", "sys2", "tpl") != base
    assert configuration_hash("m", "sys", "tpl2") != base
