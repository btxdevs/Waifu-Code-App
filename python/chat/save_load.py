"""Disk persistence for chat sessions. Port of
Assets/Scripts/Chat/SaveLoadManager.cs + ChatSaveData.cs.

Saves live in `saves/` (configurable), one folder per chat grouped by character:

  * `saves/<characterId>/<slot>/chat.json`         — the full session snapshot
  * `saves/<characterId>/<slot>/report_<id>.md`    — one file per WriteReport markdown body
  * `saves/<characterId>/<slot>/image_<hash>.<ext>`— externalized image attachments

Slot names are globally unique (chat_<ms>), so slot-keyed lookups resolve the folder by
scanning `*/<slot>/chat.json` (cached). Atomic writes via temp-file + rename so a crash
mid-write can't leave a half-written save. Report ids are sanitized to
letters/digits/-/_ to make path traversal impossible even with a malicious LLM.
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .models import ChatMessage, SAVED_IMAGE_URL_PREFIX


from .app_paths import APP_ROOT

# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
_DEFAULT_SAVES_DIR = APP_ROOT / "saves"


# ----------------------------------------------------------------------------
# Save schema (mirrors Assets/Scripts/Chat/ChatSaveData.cs)
# ----------------------------------------------------------------------------

@dataclass
class TodoItemSnapshot:
    content: str = ""
    active_form: str = ""
    # "pending" | "in_progress" | "completed" — wire-format strings, not enums.
    status: str = "pending"

    def to_wire(self) -> dict:
        return {"content": self.content, "activeForm": self.active_form, "status": self.status}

    @classmethod
    def from_wire(cls, d: dict) -> "TodoItemSnapshot":
        return cls(
            content=str(d.get("content") or ""),
            active_form=str(d.get("activeForm") or ""),
            status=str(d.get("status") or "pending"),
        )


@dataclass
class TodoSnapshotEntry:
    id: str = ""
    history_index: int = -1
    items: list[TodoItemSnapshot] = field(default_factory=list)

    def to_wire(self) -> dict:
        return {"id": self.id, "historyIndex": self.history_index,
                "items": [i.to_wire() for i in self.items]}

    @classmethod
    def from_wire(cls, d: dict) -> "TodoSnapshotEntry":
        return cls(
            id=str(d.get("id") or ""),
            history_index=int(d.get("historyIndex", -1)),
            items=[TodoItemSnapshot.from_wire(i) for i in (d.get("items") or []) if isinstance(i, dict)],
        )


@dataclass
class ReportEntry:
    id: str = ""
    title: str = ""
    history_index: int = -1

    def to_wire(self) -> dict:
        return {"id": self.id, "title": self.title, "historyIndex": self.history_index}

    @classmethod
    def from_wire(cls, d: dict) -> "ReportEntry":
        return cls(
            id=str(d.get("id") or ""),
            title=str(d.get("title") or ""),
            history_index=int(d.get("historyIndex", -1)),
        )


@dataclass
class TurnSnapshot:
    """State captured right before a user turn was submitted. Used to roll back N turns
    and resume with the original message text re-inserted into the input field."""
    user_message: str = ""
    history_length: int = 0
    outfit_name: str = ""
    # Stable KK coordinate id of the outfit at this turn. Preferred over outfit_name on restore
    # (names are user-editable); -1 = unknown (legacy snapshot / non-KK).
    outfit_index: int = -1
    current_status: str = ""
    emotion_labels: list[str] = field(default_factory=list)
    # Absolute paths of files attached to this turn's message, so a rollback can re-offer them
    # in the composer (the user re-sends the same attachments along with the refilled text).
    attachments: list[str] = field(default_factory=list)
    # Absolute paths of image files to show as bubble thumbnails — the preprocessed saved copy
    # for images we externalize, else the original. Persisted so thumbnails survive reload.
    image_thumbs: list[str] = field(default_factory=list)

    def to_wire(self) -> dict:
        return {
            "userMessage": self.user_message,
            "historyLength": self.history_length,
            "outfitName": self.outfit_name,
            "outfitIndex": self.outfit_index,
            "currentStatus": self.current_status,
            "emotionLabels": list(self.emotion_labels),
            "attachments": list(self.attachments),
            "imageThumbs": list(self.image_thumbs),
        }

    @classmethod
    def from_wire(cls, d: dict) -> "TurnSnapshot":
        idx = d.get("outfitIndex")
        return cls(
            user_message=str(d.get("userMessage") or ""),
            history_length=int(d.get("historyLength", 0)),
            outfit_name=str(d.get("outfitName") or ""),
            outfit_index=int(idx) if isinstance(idx, (int, float)) else -1,
            current_status=str(d.get("currentStatus") or ""),
            emotion_labels=[str(e) for e in (d.get("emotionLabels") or []) if isinstance(e, str)],
            attachments=[str(a) for a in (d.get("attachments") or []) if isinstance(a, str) and a],
            image_thumbs=[str(a) for a in (d.get("imageThumbs") or []) if isinstance(a, str) and a],
        )


@dataclass
class ChatSaveData:
    """One slot's full persisted state. Bump `version` if the schema changes incompatibly."""
    version: int = 1
    # Stable id of the character this chat belongs to (authoritative). character_name is kept
    # as a display snapshot so the saves list can label a chat even if the character is gone.
    character_id: str = ""
    character_name: str = ""
    outfit_name: str = ""
    # Stable KK coordinate id of the current outfit. Preferred over outfit_name on resume (names
    # are user-editable); -1 = unknown (legacy save / non-KK).
    outfit_index: int = -1
    user_name: str = ""
    current_status: str = "Nothing in particular."
    emotion_labels: list[str] = field(default_factory=list)
    history: list[ChatMessage] = field(default_factory=list)
    turn_snapshots: list[TurnSnapshot] = field(default_factory=list)
    reports: list[ReportEntry] = field(default_factory=list)
    todo_snapshots: list[TodoSnapshotEntry] = field(default_factory=list)
    # ---- Per-chat settings chosen in the new-chat dialog (see ChatManager._on_create_new). ----
    # Whether the assistant speaks its replies (voice mode) for this chat. Gated at runtime by
    # voice availability — a character with no voice for the active provider stays in no-voice mode.
    voice_mode: bool = True
    # Head/eye look-at tracking for this chat (the avatar follows the camera). Default on; toggled
    # live from the sidebar (Chat.SetHeadTracking / Chat.SetEyeTracking) and applied in Unity.
    head_tracking: bool = True
    eye_tracking: bool = True
    # Active TTS provider for this chat ("pocket"/"elevenlabs"); "" = follow the global config.
    # Switched in on session start so each chat can use a different voice engine.
    voice_provider: str = ""
    # Id of the LLM config (in llm.config.json's set) this chat uses; "" = follow the default.
    # Applied on session start so each chat can target a different model/endpoint.
    llm_config_id: str = ""
    # Workspace folders the file tools may reach for THIS chat — REPLACES the global config's
    # allowedRoots while this chat is active. Empty = fall back to the global config roots.
    workspace_roots: list[str] = field(default_factory=list)
    # Folders granted to the workspace sandbox via file attachments (outside the configured
    # work folder). Re-applied on load so the AI can still reach attached files after a reload.
    extra_workspace_roots: list[str] = field(default_factory=list)
    # Per-chat Chat-mode camera + transparent-window framing (zoom / pan / rotation / window position).
    # OPAQUE to Python — Unity defines the shape (sent via Chat.ViewState, restored via
    # Session.Begin.chatView); we just round-trip the dict so the field set can evolve Unity-side.
    chat_view: dict = field(default_factory=dict)
    # ISO-8601 UTC; set by SaveLoadManager on write.
    saved_at_utc: str = ""

    def to_wire(self) -> dict:
        return {
            "version": self.version,
            "characterId": self.character_id,
            "characterName": self.character_name,
            "outfitName": self.outfit_name,
            "outfitIndex": self.outfit_index,
            "userName": self.user_name,
            "currentStatus": self.current_status,
            "emotionLabels": list(self.emotion_labels),
            "history": [m.to_wire() for m in self.history if m is not None],
            "turnSnapshots": [t.to_wire() for t in self.turn_snapshots],
            "reports": [r.to_wire() for r in self.reports],
            "todoSnapshots": [t.to_wire() for t in self.todo_snapshots],
            "voiceMode": self.voice_mode,
            "headTracking": self.head_tracking,
            "eyeTracking": self.eye_tracking,
            "voiceProvider": self.voice_provider,
            "llmConfigId": self.llm_config_id,
            "workspaceRoots": list(self.workspace_roots),
            "extraWorkspaceRoots": list(self.extra_workspace_roots),
            "chatView": dict(self.chat_view),
            "savedAtUtc": self.saved_at_utc,
        }

    @classmethod
    def from_wire(cls, d: dict) -> "ChatSaveData":
        return cls(
            version=int(d.get("version", 1)),
            character_id=str(d.get("characterId") or ""),
            character_name=str(d.get("characterName") or ""),
            outfit_name=str(d.get("outfitName") or ""),
            outfit_index=int(d["outfitIndex"]) if isinstance(d.get("outfitIndex"), (int, float)) else -1,
            user_name=str(d.get("userName") or ""),
            current_status=str(d.get("currentStatus") or "Nothing in particular."),
            emotion_labels=[str(e) for e in (d.get("emotionLabels") or []) if isinstance(e, str)],
            history=[ChatMessage.from_wire(m) for m in (d.get("history") or []) if isinstance(m, dict)],
            turn_snapshots=[TurnSnapshot.from_wire(t) for t in (d.get("turnSnapshots") or []) if isinstance(t, dict)],
            reports=[ReportEntry.from_wire(r) for r in (d.get("reports") or []) if isinstance(r, dict)],
            todo_snapshots=[TodoSnapshotEntry.from_wire(t) for t in (d.get("todoSnapshots") or []) if isinstance(t, dict)],
            voice_mode=bool(d.get("voiceMode", True)),
            head_tracking=bool(d.get("headTracking", True)),
            eye_tracking=bool(d.get("eyeTracking", True)),
            voice_provider=str(d.get("voiceProvider") or ""),
            llm_config_id=str(d.get("llmConfigId") or ""),
            workspace_roots=[str(r) for r in (d.get("workspaceRoots") or []) if isinstance(r, str) and r],
            extra_workspace_roots=[str(r) for r in (d.get("extraWorkspaceRoots") or []) if isinstance(r, str) and r],
            chat_view=dict(d["chatView"]) if isinstance(d.get("chatView"), dict) else {},
            saved_at_utc=str(d.get("savedAtUtc") or ""),
        )


# ----------------------------------------------------------------------------
# Save slot metadata (used by the renderer's "load chat" picker)
# ----------------------------------------------------------------------------

@dataclass
class SaveSlotMetadata:
    slot: str = ""
    character_id: str = ""
    character_name: str = ""
    user_name: str = ""
    saved_at_utc: str = ""
    last_message_text: str = ""
    # Per-chat workspace folders (the file tools' allowed roots for this chat). Empty = the chat
    # follows the global config roots. Surfaced in the home-page saves list + search.
    workspace_roots: list[str] = field(default_factory=list)

    def to_wire(self) -> dict:
        return {
            "slot": self.slot,
            "characterId": self.character_id,
            "characterName": self.character_name,
            "userName": self.user_name,
            "savedAtUtc": self.saved_at_utc,
            "lastMessageText": self.last_message_text,
            "workspaceRoots": list(self.workspace_roots),
        }


# ----------------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------------

_EMOTION_TAG_REGEX = re.compile(r"\[[^\[\]]*\]")
_SAFE_ID_REGEX = re.compile(r"[^A-Za-z0-9_\-]")

# Image-block externalization: inline base64 image_url blocks are written to side files so the
# JSON save doesn't bloat with many images. The on-disk save references them via this prefix.
_DATA_URL_RE = re.compile(r"^data:(image/[A-Za-z0-9.+\-]+);base64,(.*)$", re.DOTALL)
_MIME_TO_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif",
                "image/webp": "webp", "image/bmp": "bmp"}
_EXT_TO_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}
_SAVED_IMAGE_PREFIX = SAVED_IMAGE_URL_PREFIX


class SaveLoadManager:
    """All disk IO for saves + reports. Single instance per process — pass it to ChatManager."""

    def __init__(self, saves_dir: str | os.PathLike | None = None, default_slot: str = "default",
                 verbose: bool = False):
        self.saves_dir = Path(saves_dir) if saves_dir else _DEFAULT_SAVES_DIR
        self.default_slot = default_slot
        self.verbose = verbose
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        # slot → its folder (saves/<charId>/<slot>). Slots are globally unique, so a scan
        # can always rebuild an entry; the cache just avoids re-globbing on every access.
        self._slot_dirs: dict[str, Path] = {}

    # ---- slot folders --------------------------------------------------------

    def _slot_dir(self, slot: str | None) -> Path | None:
        """The existing folder for `slot` (saves/<charId>/<slot>), or None when the slot
        has never been saved. Resolved by scanning `*/<slot>/chat.json` and cached."""
        s = _sanitize_id(slot or self.default_slot)
        cached = self._slot_dirs.get(s)
        if cached is not None and (cached / "chat.json").exists():
            return cached
        try:
            for f in self.saves_dir.glob(f"*/{s}/chat.json"):
                self._slot_dirs[s] = f.parent
                return f.parent
        except OSError:
            pass
        return None

    def _slot_dir_for_write(self, slot: str | None, character_id: str = "") -> Path:
        """The folder to WRITE into for `slot`: the existing one, else a fresh
        saves/<charId>/<slot> (charId falls back to 'unknown' — side files written
        before the first chat.json save land somewhere deterministic)."""
        existing = self._slot_dir(slot)
        if existing is not None:
            return existing
        s = _sanitize_id(slot or self.default_slot)
        char = _sanitize_id(character_id) if character_id else "unknown"
        folder = self.saves_dir / char / s
        self._slot_dirs[s] = folder
        return folder

    # ---- save / load ---------------------------------------------------------

    def get_save_path(self, slot: str | None = None) -> Path | None:
        """The slot's chat.json, or None when the slot has never been saved."""
        folder = self._slot_dir(slot)
        return folder / "chat.json" if folder is not None else None

    def has_save(self, slot: str | None = None) -> bool:
        return self.get_save_path(slot) is not None

    def save(self, data: ChatSaveData, slot: str | None = None) -> None:
        if data is None:
            return
        data.saved_at_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
        folder = self._slot_dir_for_write(slot, data.character_id)
        path = folder / "chat.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            wire = data.to_wire()
            # Move inline base64 images out to side files so the JSON stays small.
            self._externalize_images(wire, slot, folder)
            tmp.write_text(json.dumps(wire, indent=2), encoding="utf-8")
            os.replace(tmp, path)  # atomic on POSIX; close-to-atomic on Windows
            if self.verbose:
                print(f"[SaveLoadManager] Saved → {path}", file=sys.stderr)
        except OSError as e:
            print(f"[SaveLoadManager] Save failed: {e}", file=sys.stderr)

    def load(self, slot: str | None = None) -> ChatSaveData | None:
        path = self.get_save_path(slot)
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                print(f"[SaveLoadManager] {path}: root must be an object", file=sys.stderr)
                return None
            # Note: externalized image refs (saved-image:) are left as-is in the loaded history.
            # They're resolved to base64 lazily, only when a message is actually sent to the LLM
            # (see resolve_saved_image_url + history_to_wire), so reload stays cheap.
            loaded = ChatSaveData.from_wire(data)
            if self.verbose:
                print(f"[SaveLoadManager] Loaded ← {path}", file=sys.stderr)
            return loaded
        except (OSError, json.JSONDecodeError) as e:
            print(f"[SaveLoadManager] Load failed (will start fresh): {e}", file=sys.stderr)
            return None

    def delete_save(self, slot: str | None = None) -> None:
        """Remove the slot's whole folder (save + reports + images go together). An
        emptied character folder is removed too."""
        folder = self._slot_dir(slot)
        self._slot_dirs.pop(_sanitize_id(slot or self.default_slot), None)
        if folder is None:
            return
        try:
            shutil.rmtree(folder, ignore_errors=True)
            parent = folder.parent
            if parent != self.saves_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as e:
            print(f"[SaveLoadManager] Delete save failed: {e}", file=sys.stderr)

    def rebind_slot(self, slot: str, character_id: str, character_name: str) -> str | None:
        """Re-bind a saved chat to another character: rewrites characterId/Name inside
        chat.json, moves the slot folder under the new character's directory, and fixes
        the absolute thumbnail paths that pointed into the old location. Returns an
        error string, or None on success."""
        folder = self._slot_dir(slot)
        if folder is None:
            return "Save not found."
        dest = self.saves_dir / _sanitize_id(character_id or "unknown") / folder.name
        try:
            raw = json.loads((folder / "chat.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            return f"Couldn't read the save: {e}"
        raw["characterId"] = character_id
        raw["characterName"] = character_name
        old_prefix = str(folder).replace("\\", "/")
        new_prefix = str(dest).replace("\\", "/")
        for snap in raw.get("turnSnapshots") or []:
            thumbs = snap.get("imageThumbs") if isinstance(snap, dict) else None
            if not isinstance(thumbs, list):
                continue
            for i, tp in enumerate(thumbs):
                if isinstance(tp, str) and tp.replace("\\", "/").startswith(old_prefix + "/"):
                    thumbs[i] = new_prefix + tp.replace("\\", "/")[len(old_prefix):]
        try:
            if dest != folder:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(dest))
            (dest / "chat.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
        except OSError as e:
            return f"Rebind failed: {e}"
        self._slot_dirs[folder.name] = dest
        try:
            parent = folder.parent
            if parent != self.saves_dir and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
        return None

    def list_save_slots(self) -> list[SaveSlotMetadata]:
        """Scans the saves dir, returns metadata for each slot, sorted by savedAtUtc desc."""
        out: list[SaveSlotMetadata] = []
        try:
            for f in self.saves_dir.glob("*/*/chat.json"):
                slot = f.parent.name
                self._slot_dirs[slot] = f.parent
                try:
                    raw = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    print(f"[SaveLoadManager] Skipping unreadable {f}: {e}", file=sys.stderr)
                    continue
                if not isinstance(raw, dict):
                    continue
                last_msg = self._extract_last_message(raw.get("history") or [])
                out.append(SaveSlotMetadata(
                    slot=slot,
                    character_id=str(raw.get("characterId") or ""),
                    character_name=str(raw.get("characterName") or ""),
                    user_name=str(raw.get("userName") or ""),
                    saved_at_utc=str(raw.get("savedAtUtc") or ""),
                    last_message_text=last_msg,
                    workspace_roots=[str(r) for r in (raw.get("workspaceRoots") or []) if isinstance(r, str) and r],
                ))
        except OSError as e:
            print(f"[SaveLoadManager] Failed to scan save slots: {e}", file=sys.stderr)

        out.sort(key=lambda m: m.saved_at_utc, reverse=True)
        return out

    @staticmethod
    def _extract_last_message(history: list) -> str:
        for m in reversed(history):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content:
                # Strip inline emotion tags so "[Joy]Hello" surfaces as "Hello".
                stripped = _EMOTION_TAG_REGEX.sub("", content).strip()
                if len(stripped) > 60:
                    stripped = stripped[:57] + "..."
                return stripped
        return ""

    # ---- image attachment externalization -----------------------------------

    def _externalize_images(self, wire: dict, slot: str | None, folder: Path) -> None:
        """Write inline base64 `image_url` blocks in the wire history out to side files
        (`image_{hash}.{ext}` in the slot's folder) and replace them with
        `saved-image:{slot}/{filename}` refs, so the JSON save doesn't bloat when a chat
        has many image attachments. Idempotent — identical bytes hash to the same file —
        and prunes image files no longer referenced.

        Operates on the freshly-built wire dict; the block dicts are copied before rewriting so
        the in-memory session history keeps its full base64 (the live LLM path is untouched)."""
        s = _sanitize_id(slot or self.default_slot)
        history = wire.get("history")
        if not isinstance(history, list):
            return
        referenced: set[str] = set()
        for msg in history:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = None
            for i, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                iu = block.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else ""
                if not isinstance(url, str):
                    continue
                if url.startswith(_SAVED_IMAGE_PREFIX):
                    referenced.add(os.path.basename(url[len(_SAVED_IMAGE_PREFIX):]))
                    continue
                m = _DATA_URL_RE.match(url)
                if not m:
                    continue
                try:
                    raw = base64.b64decode(m.group(2))
                except Exception:
                    continue
                ext = _MIME_TO_EXT.get(m.group(1), "img")
                fname = f"image_{hashlib.sha1(raw).hexdigest()[:16]}.{ext}"
                try:
                    (folder / fname).write_bytes(raw)
                except OSError as e:
                    print(f"[SaveLoadManager] image externalize failed: {e}", file=sys.stderr)
                    continue
                referenced.add(fname)
                if new_content is None:
                    new_content = list(content)
                new_content[i] = {"type": "image_url",
                                  "image_url": {"url": f"{_SAVED_IMAGE_PREFIX}{s}/{fname}"}}
            if new_content is not None:
                msg["content"] = new_content
        # Also keep image files referenced only as bubble thumbnails (e.g. vision-off images that
        # have no history image block) — they're recorded in turn snapshots' imageThumbs.
        snaps = wire.get("turnSnapshots")
        if isinstance(snaps, list):
            for snap in snaps:
                for tp in (snap.get("imageThumbs") or []) if isinstance(snap, dict) else []:
                    if isinstance(tp, str):
                        referenced.add(os.path.basename(tp))
        # Prune image files for this slot that are no longer referenced (rollback / context strip).
        try:
            for f in folder.glob("image_*"):
                if f.name not in referenced:
                    try:
                        f.unlink()
                    except OSError:
                        pass
        except OSError:
            pass

    def store_image(self, raw: bytes, slot: str | None = None, ext: str = "jpg") -> str:
        """Write image bytes to a side file using the SAME hash-based name as
        _externalize_images (so the two coincide and aren't duplicated/pruned). Returns the
        absolute path, or "" on failure. Used to persist an attachment's preprocessed image at
        submit time so its bubble thumbnail survives even if the original file moves."""
        folder = self._slot_dir_for_write(slot)
        fname = f"image_{hashlib.sha1(raw).hexdigest()[:16]}.{ext}"
        path = folder / fname
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        except OSError as e:
            print(f"[SaveLoadManager] store_image failed: {e}", file=sys.stderr)
            return ""
        return str(path)

    def resolve_saved_image_url(self, url: str) -> str | None:
        """If `url` is a `saved-image:{slot}/{filename}` reference, read the side file and
        return a base64 data URL; otherwise None (the caller leaves the block unchanged).
        Called at LLM-send time via history_to_wire's image_resolver, so image bytes are only
        read from disk when actually sent — not on every chat load. The slot embedded in the
        ref locates the folder, so no session context is needed."""
        if not isinstance(url, str) or not url.startswith(_SAVED_IMAGE_PREFIX):
            return None
        slot, sep, fname = url[len(_SAVED_IMAGE_PREFIX):].partition("/")
        if not sep:
            return None
        folder = self._slot_dir(slot)
        if folder is None:
            return None
        fname = os.path.basename(fname)
        try:
            raw = (folder / fname).read_bytes()
        except OSError:
            return None
        ext = Path(fname).suffix.lstrip(".").lower()
        mime = _EXT_TO_MIME.get(ext, "image/jpeg")
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    # ---- reports ------------------------------------------------------------

    def get_report_path(self, report_id: str, slot: str | None = None) -> Path:
        safe = _sanitize_id(report_id)
        return self._slot_dir_for_write(slot) / f"report_{safe}.md"

    def has_report(self, report_id: str, slot: str | None = None) -> bool:
        return bool(report_id) and self.get_report_path(report_id, slot).exists()

    def save_report(self, report_id: str, content: str, slot: str | None = None) -> None:
        if not report_id:
            return
        path = self.get_report_path(report_id, slot)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(content or "", encoding="utf-8")
            os.replace(tmp, path)
            if self.verbose:
                print(f"[SaveLoadManager] Wrote report → {path}", file=sys.stderr)
        except OSError as e:
            print(f"[SaveLoadManager] SaveReport failed: {e}", file=sys.stderr)

    def load_report(self, report_id: str, slot: str | None = None) -> str | None:
        if not report_id:
            return None
        path = self.get_report_path(report_id, slot)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[SaveLoadManager] LoadReport failed: {e}", file=sys.stderr)
            return None

    def delete_report(self, report_id: str, slot: str | None = None) -> None:
        if not report_id:
            return
        path = self.get_report_path(report_id, slot)
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            print(f"[SaveLoadManager] DeleteReport failed: {e}", file=sys.stderr)


def _sanitize_id(report_id: str) -> str:
    if not report_id:
        return "report"
    cleaned = _SAFE_ID_REGEX.sub("", report_id)
    return cleaned or "report"
