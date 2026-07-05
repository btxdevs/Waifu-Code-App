"""App-owned long-term memory — two scopes.

* Character memory — facts about a specific character (and the user's history with them),
  keyed by the character's stable id. Private to that character.
* Project memory — facts about a workfolder, keyed by the NORMALIZED workspace root path and
  shared by every character whose active workspace resolves to that root.

Each scope is one JSON file holding a list of MemoryEntry. Recall mirrors Claude Code's own
two-tier scheme: the chat manager injects a compact INDEX of every in-scope entry into the system
prompt each turn (cheap, always-on), plus the full body of the entries most relevant to the current
message (a small keyword-overlap pass — embeddings can replace it later). The LLM writes entries via
the Remember / Forget tools and can pull a specific body with RecallMemory.

Atomic writes (temp-file + rename) and filename sanitization mirror character_store.py / save_load.py.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path


from .app_paths import APP_ROOT

# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
_DEFAULT_MEMORY_DIR = APP_ROOT / "memory"

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
# Generic words that don't help relevance scoring.
_STOPWORDS = frozenset(
    "the a an and or of to in on at for with is are was were be been being it its this that these "
    "those you your yours i me my we us our he she they them his her their as by from but not no if "
    "do does did so just can could would should will what when where who how why about into out up".split()
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_memory_id() -> str:
    return uuid.uuid4().hex


def _tokens(text: str) -> set[str]:
    """Content words of `text`, lowercased, minus stopwords and 1-2 char noise."""
    return {w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2 and w not in _STOPWORDS}


@dataclass
class MemoryEntry:
    """One remembered fact. `description` is the relevance key shown in the index; `text` is the
    body surfaced when relevant. `name` is a short title the LLM also uses to Forget / RecallMemory."""
    id: str = ""
    name: str = ""
    description: str = ""
    text: str = ""
    pinned: bool = False           # exempt from cap eviction
    created_at_utc: str = ""
    updated_at_utc: str = ""

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "text": self.text,
            "pinned": self.pinned,
            "createdAtUtc": self.created_at_utc,
            "updatedAtUtc": self.updated_at_utc,
        }

    @classmethod
    def from_wire(cls, d: dict) -> "MemoryEntry":
        return cls(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or ""),
            description=str(d.get("description") or ""),
            text=str(d.get("text") or ""),
            pinned=bool(d.get("pinned")),
            created_at_utc=str(d.get("createdAtUtc") or ""),
            updated_at_utc=str(d.get("updatedAtUtc") or ""),
        )


class MemoryStore:
    """CRUD + two-tier render over per-scope memory files. One file per key, holding a list of
    MemoryEntry. Subclasses define how a key maps to a filename stem (and what `key` means)."""

    MAX_ENTRIES = 60     # per scope; oldest non-pinned entries are evicted past this
    TOP_K_BODIES = 4     # full bodies surfaced per scope by the relevance pass

    def __init__(self, subdir: str, memory_dir: Path | str | None = None):
        base = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self.dir = base / subdir
        self.dir.mkdir(parents=True, exist_ok=True)

    # --- keying (overridable) -------------------------------------------------
    def _stem(self, key: str) -> str:
        return _SAFE_RE.sub("_", (key or "").strip()) or "default"

    def _path_for(self, key: str) -> Path:
        return self.dir / f"{self._stem(key)}.json"

    # --- load / save ----------------------------------------------------------
    def load(self, key: str) -> list[MemoryEntry]:
        path = self._path_for(key)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [MemoryEntry.from_wire(e) for e in (data.get("entries") or []) if isinstance(e, dict)]
        except Exception as e:  # noqa: BLE001 — a corrupt file shouldn't crash the chat
            print(f"[MemoryStore] Failed to read '{path.name}': {e}", file=sys.stderr)
            return []

    def _save(self, key: str, entries: list[MemoryEntry]) -> None:
        path = self._path_for(key)
        payload = {"key": str(key), "entries": [e.to_wire() for e in entries]}
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem

    # --- mutations ------------------------------------------------------------
    def add(self, key: str, name: str, description: str, text: str, pinned: bool = False) -> MemoryEntry:
        entries = self.load(key)
        now = _utc_now()
        entry = MemoryEntry(
            id=new_memory_id(), name=(name or "").strip(), description=(description or "").strip(),
            text=(text or "").strip(), pinned=pinned, created_at_utc=now, updated_at_utc=now,
        )
        entries.append(entry)
        # Cap: drop the oldest non-pinned entries past MAX_ENTRIES, preserving the rest's order.
        if len(entries) > self.MAX_ENTRIES:
            over = len(entries) - self.MAX_ENTRIES
            evictable = sorted((e for e in entries if not e.pinned), key=lambda e: e.updated_at_utc)
            drop = {e.id for e in evictable[:over]}
            entries = [e for e in entries if e.id not in drop]
        self._save(key, entries)
        return entry

    def remove_by_name(self, key: str, name: str) -> int:
        """Remove every entry whose name matches `name` (case-insensitive). Returns the count removed."""
        target = (name or "").strip().casefold()
        if not target:
            return 0
        entries = self.load(key)
        kept = [e for e in entries if e.name.strip().casefold() != target]
        removed = len(entries) - len(kept)
        if removed:
            self._save(key, kept)
        return removed

    def get_by_name(self, key: str, name: str) -> MemoryEntry | None:
        target = (name or "").strip().casefold()
        if not target:
            return None
        for e in self.load(key):
            if e.name.strip().casefold() == target:
                return e
        return None

    # --- render (two-tier: index + relevant bodies) ---------------------------
    def render(self, key: str, query: str, header: str) -> str:
        entries = self.load(key)
        if not entries:
            return ""
        lines = [header]
        for e in entries:
            lines.append(f"  - {e.name}: {e.description}" if e.description else f"  - {e.name}")
        bodies = self._relevant(entries, query)
        if bodies:
            lines.append("")
            lines.append("Relevant right now (full detail):")
            for e in bodies:
                lines.append(f"  > {e.name}: {e.text}")
        return "\n".join(lines)

    def _relevant(self, entries: list[MemoryEntry], query: str) -> list[MemoryEntry]:
        q = _tokens(query)
        if not q:
            return []
        scored: list[tuple[int, str, MemoryEntry]] = []
        for e in entries:
            # Weight name+description (the recall key) over the body.
            overlap = len(q & _tokens(f"{e.name} {e.description}")) * 2 + len(q & _tokens(e.text))
            if overlap > 0:
                scored.append((overlap, e.updated_at_utc, e))
        # Highest overlap first; most-recently-updated wins ties (stable sort + recency pre-sort).
        scored.sort(key=lambda t: t[1], reverse=True)
        scored.sort(key=lambda t: t[0], reverse=True)
        return [e for _, _, e in scored[: self.TOP_K_BODIES]]


class CharacterMemoryStore(MemoryStore):
    """Per-character memory, keyed by the character's stable id (the sanitized id is the filename)."""

    def __init__(self, memory_dir: Path | str | None = None):
        super().__init__("characters", memory_dir)


class ProjectMemoryStore(MemoryStore):
    """Per-workspace memory, keyed by the workspace root PATH. The filename is a hash of the
    normalized path (the raw path is stored inside the file for readability), so different spellings
    of the same folder share one memory and any character working there sees it."""

    def __init__(self, memory_dir: Path | str | None = None):
        super().__init__("projects", memory_dir)

    @staticmethod
    def normalize_path(path: str) -> str:
        s = str(path or "").strip().replace("\\", "/").rstrip("/")
        if not s:
            return ""
        try:
            s = str(Path(s).resolve()).replace("\\", "/").rstrip("/")
        except Exception:  # noqa: BLE001 — a non-existent/odd path still keys deterministically
            pass
        return s.casefold() if os.name == "nt" else s  # Windows FS is case-insensitive

    def _stem(self, key: str) -> str:
        norm = self.normalize_path(key)
        if not norm:
            return "default"
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
