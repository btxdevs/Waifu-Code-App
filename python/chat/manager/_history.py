"""Rebuilds the renderer's chat scroll-back (history entries) from the session + bookkeeping."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid

from ..character_store import CharacterStore, CharacterRecord, new_character_id
from ..config import ChatBackendConfig
from ..llm_client import LlmClient
from ..models import ChatMessage, EmotionEntry, StructuredReply, ToolSchema
from ..orchestrator import (
    ChatOrchestrator, ChatSession, CharacterInfo, OrchestratorEvents,
    ToolExecutionResult, ToolRunner,
)
from ..save_load import (
    ChatSaveData, ReportEntry, SaveLoadManager, TodoSnapshotEntry, TodoItemSnapshot, TurnSnapshot,
)
from ..speech import SentenceSpeechPipeline
from ..text import EmotionStreamFilter
from .protocol import *  # noqa: F401,F403  (envelope-type + callable-alias constants)
from .view_models import (
    HistoryEntry, ReportRef, TodoItemRef, strip_emotion_tags,
    _tool_activity_label, _leading_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
    build_attachment_refs, strip_attachment_notes,
)


class HistoryMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    def _build_history_entries(self, session: ChatSession) -> list[HistoryEntry]:
        """Walks session.history → ordered list of UI rows: user bubbles, assistant
        speech bubbles, and tool_activity events.

        For each assistant message in history we emit (in order):
          1. an assistant bubble — only if the cleaned content is non-empty
          2. one tool_activity row per tool_call on that message
        Tool result rows are skipped from the UI; they belong to the same turn and
        their effect is already represented by the tool_activity row that "preceded"
        them. Inline [LABEL] tags are stripped from visible assistant text.

        Widgets attach to the matching tool_activity row: WriteReport gets the
        "view report" button, TodoWrite gets the todo list. Reports/todos are matched
        by the history_index of the assistant message that contained the tool_call —
        register_report / register_todo_snapshot record that index at registration time.
        """
        n = len(session.history)
        entries: list[HistoryEntry] = []
        # turn_index counts ONLY user rows since rollback keys off user messages.
        turn_idx = 0
        for i in range(n):
            m = session.history[i]
            if m is None:
                continue

            if m.role == "user":
                # Hidden content-blocks user rows (no plain `content`): image attachments (Read tool) and
                # injected touch / "touching ended" notes. They exist so the LLM sees them, but aren't
                # chat bubbles.
                if m.content_blocks and not m.content:
                    # A touch note (or a user outfit change) renders as a rewindable
                    # "user_action" row (an event line, kept separate from the assistant's
                    # tool rows) and IS a turn, so it advances turn_idx in lockstep with
                    # _turn_snapshots. A background task report (task_done) renders the same
                    # way but is NOT rewindable — no snapshot was taken for it, so it must
                    # not consume a turn_idx either. Everything else stays invisible and
                    # doesn't count against turn_idx (would shift rollback indices).
                    if m.touch_zone or m.outfit_change:
                        entries.append(HistoryEntry(
                            role="user_action",
                            speaker=session.display_name,
                            text=(self._touch_label(m.touch_zone) if m.touch_zone
                                  else self._outfit_change_label(m.outfit_change)),
                            can_rollback=turn_idx < len(self._turn_snapshots),
                            turn_index=turn_idx,
                            reports=[], todos=None,
                            tool_name="Touch" if m.touch_zone else "ChangeOutfit",
                            history_index=i,
                        ))
                        turn_idx += 1
                    elif m.task_done:
                        entries.append(HistoryEntry(
                            role="user_action",
                            speaker=session.display_name,
                            # task_done carries the ready-made row label (see _tasks.py).
                            text=m.task_done,
                            can_rollback=False,
                            turn_index=-1,
                            reports=[], todos=None,
                            tool_name="UwUAgent",
                            history_index=i,
                        ))
                    continue
                text = strip_emotion_tags(m.content or "").strip() if m.content else ""
                can_rollback = turn_idx < len(self._turn_snapshots)
                # Attachment chips for this turn, from the persisted snapshot. The 📎 note lines
                # _process_attachments appended stay in history for the LLM but come OUT of the
                # displayed text — the chips carry that information. Rows with no snapshot data
                # (legacy saves) keep their note lines so nothing disappears.
                snap = (self._turn_snapshots[turn_idx]
                        if turn_idx < len(self._turn_snapshots) else None)
                atts = build_attachment_refs(snap.attachments, snap.image_thumbs) if snap else []
                if atts:
                    text = strip_attachment_notes(text)
                entries.append(HistoryEntry(
                    role="user",
                    speaker=session.user_name,
                    text=text,
                    can_rollback=can_rollback,
                    turn_index=turn_idx,
                    reports=[], todos=None,
                    history_index=i,
                    attachments=atts,
                ))
                turn_idx += 1
                continue

            if m.role != "assistant":
                # Skip system / tool rows.
                continue

            cleaned = strip_emotion_tags(m.content or "").strip() if m.content else ""
            if cleaned:
                entries.append(HistoryEntry(
                    role="assistant",
                    speaker=session.display_name,
                    text=cleaned,
                    can_rollback=False,
                    turn_index=-1,
                    reports=[], todos=None,
                    history_index=i,
                ))
            # Emit one tool_activity row per tool_call on THIS assistant message. Reports
            # and todos registered while this assistant row was the latest in history
            # point at index `i`; we route the report widget onto WriteReport rows and
            # the todo snapshot onto TodoWrite rows specifically.
            if m.tool_calls:
                round_reports = [r for r in self._reports if r.history_index == i]
                round_todos = [t for t in self._todo_snapshots if t.history_index == i]
                # Track which report we've already attached so multiple WriteReport calls
                # in the same round each get their own report button (one per row).
                report_cursor = 0
                todo_cursor = 0
                for tc in m.tool_calls:
                    name = tc.function.name or "tool"
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    label = _tool_activity_label(name, args)
                    reports_here: list[ReportRef] = []
                    todos_here: list[TodoItemRef] | None = None
                    if name == "ReportWrite" and report_cursor < len(round_reports):
                        r = round_reports[report_cursor]
                        report_cursor += 1
                        reports_here.append(ReportRef(id=r.id, title=r.title))
                    if name == "TodoWrite" and todo_cursor < len(round_todos):
                        t = round_todos[todo_cursor]
                        todo_cursor += 1
                        todos_here = [TodoItemRef(content=it.content,
                                                  active_form=it.active_form,
                                                  status=it.status)
                                      for it in t.items]
                    entries.append(HistoryEntry(
                        role="tool_activity",
                        speaker=session.display_name,
                        text=label,
                        can_rollback=False,
                        turn_index=-1,
                        reports=reports_here,
                        todos=todos_here,
                        tool_name=name,
                    ))
        return entries

    def _push_history(self, session: ChatSession) -> None:
        entries = [h.to_wire() for h in self._build_history_entries(session)]
        self._push_envelope(T_CHAT_HISTORY, {"entries": entries})
