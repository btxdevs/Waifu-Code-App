"""Read-freshness tracker.

Port of Assets/Scripts/ChatTools/Workspace/ReadFileStateTracker.cs.
Edit/Write gate their operations behind "the LLM must have read this file at least
once, and the file's mtime hasn't advanced since". This tracker stores that state
in-memory for the lifetime of the process.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ReadStateEntry:
    """One stored read. `content` is only set for whole-file text reads (used by
    Write's "content equality" fallback when mtime has advanced). `timestamp` is
    the mtime at read time, in nanoseconds — directly comparable across calls."""
    content: str | None = None
    timestamp: int = 0
    offset: int | None = None
    limit: int | None = None


class ReadFileStateTracker:
    """Process-lifetime in-memory store keyed by absolute path. Case-folded on
    Windows-style filesystems to match Path.GetFullPath casing quirks."""

    def __init__(self) -> None:
        self._entries: dict[str, ReadStateEntry] = {}

    def get(self, absolute_path: str) -> ReadStateEntry | None:
        if not absolute_path:
            return None
        return self._entries.get(self._key(absolute_path))

    def set(self, absolute_path: str, entry: ReadStateEntry) -> None:
        if not absolute_path:
            return
        self._entries[self._key(absolute_path)] = entry

    def clear(self) -> None:
        self._entries.clear()

    @staticmethod
    def mtime_ticks(absolute_path: str) -> int:
        """File's last-modified time as a comparable int. Mirrors GetMTimeTicks
        on the C# side. Returns 0 on stat failure."""
        try:
            return os.stat(absolute_path).st_mtime_ns
        except OSError:
            return 0

    @staticmethod
    def _key(absolute_path: str) -> str:
        # Casefold so case-insensitive filesystems (Windows, default macOS HFS+)
        # don't end up with two entries for the same file.
        return os.path.normpath(absolute_path).casefold()
