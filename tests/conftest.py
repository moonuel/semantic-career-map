"""Shared test doubles for storage-backed unit tests.

``MemorySink`` is the in-process stand-in for
:class:`backend.storage.store.ObjectStore`: same protocol surface
(``put``/``exists``/``get``/``list_keys``), zero I/O. The object store is the
only storage backend — local dev runs Garage, CI and prod run dev/prod S3 —
so tests never touch a filesystem mirror; they use this.
"""

from __future__ import annotations

import pytest

from backend.storage import keys


class MemorySink:
    """Dict-backed storage double with the ``ObjectStore`` method surface."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        """Store ``data`` at ``key`` (validated, overwrite)."""
        keys.validate_key(key)
        self.objects[key] = data

    def exists(self, key: str) -> bool:
        """Return whether ``key`` is present (validated)."""
        keys.validate_key(key)
        return key in self.objects

    def get(self, key: str) -> bytes:
        """Return the bytes at ``key``; ``FileNotFoundError`` when absent."""
        keys.validate_key(key)
        if key not in self.objects:
            raise FileNotFoundError(key)
        return self.objects[key]

    def list_keys(self, prefix: str) -> list[str]:
        """Return every stored key under ``prefix``, sorted."""
        return sorted(k for k in self.objects if k.startswith(prefix))


@pytest.fixture
def memory_sink() -> MemorySink:
    """A fresh empty ``MemorySink`` per test."""
    return MemorySink()


class JournalSink(MemorySink):
    """``MemorySink`` recording put order for write-protocol assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.journal: list[str] = []

    def put(self, key: str, data: bytes) -> None:
        super().put(key, data)
        self.journal.append(key)
