"""Oversized tool results → disk (Claude Code's maxResultSizeChars pattern).

When a tool's output blows its LLM-facing cap, the FULL text is written to a
spill file and the model gets a truncated preview plus the file's path — so
nothing is lost (it can Read the file), but the conversation history only ever
carries the preview. Used by the shell output formatter and WebFetch.

The spill dir lives under the OS temp dir and is treated as always-readable by
the workspace sandbox (see workspace.resolve_path) — per-chat root overrides
can't accidentally cut the model off from its own spill files. A best-effort
janitor drops files older than a few days at ToolManager build time.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

SPILL_DIR = Path(tempfile.gettempdir()) / "WaifuCode" / "tool_results"

# Hard cap on a single spill file, to bound disk use from a runaway command.
MAX_SPILL_CHARS = 8 * 1024 * 1024

_JANITOR_MAX_AGE_SECONDS = 3 * 24 * 3600


def spill_dir() -> str:
    return str(SPILL_DIR)


def spill_text(text: str, tool_name: str) -> str | None:
    """Write `text` to a fresh spill file and return its absolute path, or None
    when the write fails (callers fall back to plain truncation)."""
    if not text:
        return None
    try:
        SPILL_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d_%H%M%S')}_{(tool_name or 'tool').lower()}_{uuid.uuid4().hex[:6]}.txt"
        path = SPILL_DIR / name
        path.write_text(text[:MAX_SPILL_CHARS], encoding="utf-8", errors="replace")
        return str(path)
    except OSError as e:
        print(f"[spill] write failed: {e}", file=sys.stderr)
        return None


def cleanup_spill_dir() -> None:
    """Best-effort janitor: drop spill files older than the retention window.
    Called once at ToolManager build (app start / settings reload)."""
    try:
        cutoff = time.time() - _JANITOR_MAX_AGE_SECONDS
        for entry in os.scandir(SPILL_DIR):
            try:
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    os.unlink(entry.path)
            except OSError:
                continue
    except OSError:
        pass  # dir doesn't exist yet — nothing to clean
