"""Tests for the ``PostingMetadata`` schema builder in the migration script.

Hermetic: no network, no store. Exercises the metadata contract only — the
schema is the durable artifact the migration and future ingestion share.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from backend.storage import keys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts" / "migrate_selected_postings"),
)

from ingest import PostingMetadata, content_sha256, split_slug

# --- split_slug --------------------------------------------------------------


class TestSplitSlug:
    """Filename stem -> (employer, title)."""

    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("affirm-ai-solutions-engineer", ("affirm", "ai-solutions-engineer")),
            ("bmo-analyst-data-scientist", ("bmo", "analyst-data-scientist")),
            ("canada-life-data-analyst", ("canada-life", "data-analyst")),
            ("fgf-brands-ai-engineer-new-grad", ("fgf-brands", "ai-engineer-new-grad")),
            (
                "industrial-and-commercial-bank-of-china-data-science-analyst",
                ("industrial-and-commercial-bank-of-china", "data-science-analyst"),
            ),
            ("motion-recruitment-ml-engineer", ("motion-recruitment", "ml-engineer")),
            (
                "ontario-teachers-pension-plan-data-scientist",
                ("ontario-teachers-pension-plan", "data-scientist"),
            ),
            (
                "promise-robotics-software-developer",
                ("promise-robotics", "software-developer"),
            ),
            (
                "venterra-realty-data-decision-scientist",
                ("venterra-realty", "data-decision-scientist"),
            ),
        ],
    )
    def test_splits_known_stems(self, stem: str, expected: tuple[str, str]) -> None:
        assert split_slug(stem) == expected

    @pytest.mark.parametrize("bad", ["", "affirm", "affirm-", "-title"])
    def test_rejects_unparseable(self, bad: str) -> None:
        with pytest.raises(ValueError):
            split_slug(bad)

    def test_every_corpus_filename_parses(self) -> None:
        source = Path(__file__).resolve().parents[1] / "data" / "selected-job-postings"
        if not source.is_dir():
            pytest.skip("corpus not present")
        stems = [p.stem for p in source.glob("*.md")]
        assert len(stems) == 37
        for stem in stems:
            employer, title = split_slug(stem)
            assert employer and title


# --- content hash ------------------------------------------------------------


class TestContentHash:
    def test_is_deterministic_hex64(self) -> None:
        h = content_sha256(b"hello")
        assert h == content_sha256(b"hello")
        assert keys._HASH_RE.fullmatch(h) is not None

    def test_differs_on_content(self) -> None:
        assert content_sha256(b"a") != content_sha256(b"b")


# --- PostingMetadata ---------------------------------------------------------


class TestPostingMetadata:
    def _make(self, **overrides: object) -> PostingMetadata:
        base = {
            "id": "0190f3a2-7b41-7c9e-8a3d-5f6e1b2c4d70",
            "source": None,
            "employer": "affirm",
            "title": "ai-solutions-engineer",
            "ingested_at": "2026-10-04T00:00:00Z",
            "content_sha256": "a" * 64,
        }
        base.update(overrides)
        return PostingMetadata(**base)  # type: ignore[arg-type]

    def test_from_source_mints_id_and_hash(self) -> None:
        meta = PostingMetadata.from_source(
            source_bytes=b"body", stem="affirm-ml-engineer-2"
        )
        assert keys._ID_RE.fullmatch(meta.id) is not None
        assert meta.employer == "affirm"
        assert meta.title == "ml-engineer-2"
        assert meta.content_sha256 == content_sha256(b"body")
        assert meta.source is None
        assert meta.ingested_at.endswith("Z")

    def test_json_round_trip_is_lossless(self) -> None:
        meta = self._make()
        assert PostingMetadata.from_json(meta.to_json()) == meta

    def test_to_json_is_canonical_and_sorted(self) -> None:
        meta = self._make()
        obj = json.loads(meta.to_json())
        assert list(obj) == sorted(obj)
        assert meta.to_json() == meta.to_json()

    def test_dropped_raw_filename_field_is_absent(self) -> None:
        obj = json.loads(self._make().to_json())
        assert "raw_filename" not in obj
        assert set(obj) == {
            "id",
            "source",
            "employer",
            "title",
            "ingested_at",
            "content_sha256",
        }

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("id", "not-a-uuid"),
            ("id", "0190f3a2-7b41-4c9e-8a3d-5f6e1b2c4d70"),
            ("content_sha256", "short"),
            ("content_sha256", "A" * 64),
            ("employer", ""),
            ("title", ""),
        ],
        ids=[
            "bad-id",
            "wrong-version",
            "short-hash",
            "uppercase-hash",
            "empty-employer",
            "empty-title",
        ],
    )
    def test_validate_rejects_bad_fields(self, field: str, bad: str) -> None:
        with pytest.raises(ValueError):
            self._make(**{field: bad}).validate()

    def test_from_json_rejects_unknown_field(self) -> None:
        with pytest.raises(TypeError):
            PostingMetadata.from_json(json.dumps({"id": "x", "extra": 1}))

    def test_frozen(self) -> None:
        meta = self._make()
        with pytest.raises(AttributeError):
            meta.employer = "other"  # type: ignore[misc]
