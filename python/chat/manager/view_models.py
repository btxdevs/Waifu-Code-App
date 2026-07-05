"""UI-side DTOs + label/tag helpers shared by the ChatManager mixins. Wire format matches the C# nested types."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


# File extensions treated as images (used when classifying attachments in _turn).
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


# ---------- UI-side DTOs (mirror the C# nested types so the renderer wire format stays identical) ----------

@dataclass
class ReportRef:
    id: str
    title: str


@dataclass
class AttachmentRef:
    """One attachment chip on a user bubble. `kind` is "image" or "file"; for images `path`
    is the preprocessed saved thumbnail copy (falls back to the original file)."""
    name: str
    path: str
    kind: str


@dataclass
class TodoItemRef:
    content: str
    active_form: str
    status: str


@dataclass
class HistoryEntry:
    """One UI row in the chat scroll-back. `role` is one of:
      * "user"          — user typed this message
      * "assistant"     — character spoke this content
      * "tool_activity" — compact inline event for a tool call ("Wrote a report: …",
                          "Edited file: …"). `text` holds the human-readable label,
                          `tool_name` carries the tool identifier for renderer styling.
      * "user_action"   — something the USER did to the avatar (e.g. a touch/caress). Its own
                          category — rendered on the right like the user's messages and kept
                          separate from the assistant's tool rows. `tool_name` picks the icon.
    Widgets (reports, todos) attach to the tool_activity row whose tool produced them.
    """
    role: str
    speaker: str
    text: str
    can_rollback: bool
    turn_index: int
    reports: list[ReportRef] = field(default_factory=list)
    todos: list[TodoItemRef] | None = None
    tool_name: str | None = None
    # Attachments on this (user) message, rendered as one compact chip row — every kind gets a
    # chip (image chips carry a tiny thumbnail the renderer loads via readImageDataUrl).
    attachments: list[AttachmentRef] = field(default_factory=list)
    # Index into session.history this row was built from. Lets the renderer's
    # inline edit map a bubble back to its source ChatMessage. -1 for rows with
    # no editable backing message (tool_activity). For live-streamed rows the
    # manager predicts the index (the message is appended to history immediately
    # after the row is pushed — see _on_player_submit / _on_orch_token).
    history_index: int = -1

    def to_wire(self) -> dict:
        out: dict = {
            "role": self.role,
            "speaker": self.speaker,
            "text": self.text,
            "canRollback": self.can_rollback,
            "turnIndex": self.turn_index,
            "historyIndex": self.history_index,
            "reports": [{"id": r.id, "title": r.title} for r in self.reports],
            # The C# version sends null (not []) when there's no snapshot for this turn —
            # the renderer treats null vs empty distinctly. Preserve.
            "todos": ([
                {"content": t.content, "activeForm": t.active_form, "status": t.status}
                for t in self.todos
            ] if self.todos else None),
        }
        if self.tool_name:
            out["toolName"] = self.tool_name
        if self.attachments:
            out["attachments"] = [
                {"name": a.name, "path": a.path, "kind": a.kind} for a in self.attachments
            ]
        return out


# The "📎 Attached image/file: …" note lines _process_attachments appends to the turn text so
# the LLM knows what was attached (and where). The UI shows attachment CHIPS instead, so these
# lines are stripped from the displayed bubble text.
_ATTACHMENT_NOTE_REGEX = re.compile(r"^\U0001F4CE Attached (?:image|file): .*$", re.MULTILINE)


def strip_attachment_notes(text: str) -> str:
    """Remove the appended 📎 attachment note lines from a user message's display text."""
    if not text or "\U0001F4CE" not in text:
        return text
    return _ATTACHMENT_NOTE_REGEX.sub("", text).strip()


def build_attachment_refs(attachments: list[str], image_thumbs: list[str]) -> list[AttachmentRef]:
    """Chip refs for a turn's attachments. `image_thumbs` holds one saved-thumbnail path per
    IMAGE attachment, in attachment order (see _process_attachments) — pair them back up here;
    non-image files chip their original path."""
    refs: list[AttachmentRef] = []
    thumbs = iter(image_thumbs or [])
    for raw in attachments or []:
        name = os.path.basename(raw) or raw
        if os.path.splitext(raw)[1].lower() in IMAGE_EXTS:
            refs.append(AttachmentRef(name=name, path=next(thumbs, "") or raw, kind="image"))
        else:
            refs.append(AttachmentRef(name=name, path=raw, kind="file"))
    return refs


def _tool_activity_label(tool_name: str, args: dict) -> str:
    """Human-readable label for a tool_activity row. Each tool gets a specific
    phrasing pulling the most relevant argument. Falls back to a generic
    "Used the <name> tool" for unknown names."""
    def _short(s: str, n: int = 60) -> str:
        s = str(s or "").strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    name = tool_name or "tool"
    a = args or {}
    match name:
        case "ReportWrite":
            return f"Wrote report: {_short(a.get('title') or 'Untitled', 80)}"
        case "Read":
            return f"Read file: {_short(a.get('file_path') or '?', 80)}"
        case "Write":
            return f"Wrote file: {_short(a.get('file_path') or '?', 80)}"
        case "Edit":
            return f"Edited file: {_short(a.get('file_path') or '?', 80)}"
        case "Open":
            return f"Opened file: {_short(a.get('path') or a.get('file_path') or '?', 80)}"
        case "Glob":
            return f"Searched files: {_short(a.get('pattern') or '?', 80)}"
        case "Grep":
            return f"Grepped: {_short(a.get('pattern') or '?', 80)}"
        case "Bash":
            return f"Ran command: {_short(a.get('command') or '?', 80)}"
        case "PowerShell":
            return f"Ran PowerShell: {_short(a.get('command') or '?', 80)}"
        case "WebFetch":
            return f"Fetched page: {_short(a.get('url') or '?', 80)}"
        case "WebSearch":
            return f"Searched web: {_short(a.get('query') or '?', 80)}"
        case "WebPageRead":
            return f"Read web page: {_short(a.get('url') or '?', 80)}"
        case "WebPageOutline":
            return f"Outlined page: {_short(a.get('url') or '?', 80)}"
        case "ChangeOutfit" | "change_outfit":
            return f"Changed outfit to: {_short(a.get('outfitName') or '?', 60)}"
        case "TodoWrite":
            return "Updated todos"
        case "UwUAgent":
            label = f"Summoned an UwU helper: {_short(a.get('description') or a.get('agent_type') or '?', 80)}"
            return label + (" (background)" if a.get("run_in_background") else "")
        case "CheckUwUHelpers":
            return "Checked on the UwU helpers"
        case "DismissUwUHelper":
            return f"Dismissed UwU helper: {_short(a.get('task_id') or '?', 20)}"
        case "AskUserQuestion":
            # Args are {"questions": [{"question", "header", ...}, ...]} — there is no
            # top-level "question" key, so read the first entry of the array.
            qs = a.get("questions")
            first = qs[0] if isinstance(qs, list) and qs and isinstance(qs[0], dict) else {}
            text = first.get("question") or first.get("header") or "?"
            if isinstance(qs, list) and len(qs) > 1:
                text = f"{text} (+{len(qs) - 1} more)"
            return f"Asked: {_short(text, 80)}"
        case _:
            return f"Used the {name} tool"


# Tag-stripping regex used when building the "last message" preview for the saves list.
_EMOTION_TAG_REGEX = re.compile(r"\[[^\[\]]*\]")

# A tag plus the whitespace straddling it. Removing a bare tag would leave a double space
# between sentences ("end. [Tag] Next" → "end.  Next"); collapsing the surrounding whitespace
# to a single space (or none, when the tag had no whitespace around it) avoids that.
_TAG_WITH_WS_REGEX = re.compile(r"(\s*)\[[^\[\]]*\](\s*)")


def strip_emotion_tags(text: str) -> str:
    """Remove [LABEL] tags from display text, collapsing whitespace at each seam so stripped
    inline tags don't leave double spaces between sentences."""
    if not text:
        return text
    return _TAG_WITH_WS_REGEX.sub(lambda m: " " if (m.group(1) or m.group(2)) else "", text)

# Matches a leading run of [LABEL] emotion tags + any whitespace right after them,
# e.g. "[Joy]\n" or "[Joy][Sadness] ". Used to preserve the avatar-driving tags when
# the user edits the visible (tag-stripped) text of an assistant message.
_LEADING_EMOTION_TAGS_REGEX = re.compile(r"^(?:\s*\[[^\[\]]*\])+\s*")


def _leading_emotion_tags(content: str) -> str:
    """Return the leading [LABEL] tag prefix (incl. trailing whitespace) of an
    assistant message, or '' if it doesn't start with one."""
    if not content:
        return ""
    m = _LEADING_EMOTION_TAGS_REGEX.match(content)
    return m.group(0) if m else ""
