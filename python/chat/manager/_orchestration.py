"""Orchestrator wiring + event handlers, speech-pipeline callbacks, tool runner, Unity RPC."""
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
    HistoryEntry, ReportRef, TodoItemRef,
    _tool_activity_label, _leading_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
)


class OrchestrationMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    async def _ensure_orchestrator(self, character: CharacterInfo) -> None:
        """Constructs the orchestrator on first use. Tools live in Python now —
        ToolManager is built once and reused across sessions, with per-session
        schemas rebuilt at every turn (so e.g. ChangeOutfit's outfit enum stays
        accurate when the character switches)."""
        if self.orchestrator is None:
            # Lazy import to avoid pulling chat.tools at module load time.
            from ..tools import build_tool_manager
            self._tool_manager = build_tool_manager(
                chat_manager=self,
                ask_modal=self._ask_modal_fn,
                image_processor=self._image_processor_fn,
                ocr_processor=self._ocr_processor_fn,
                supports_vision=bool(self.config.llm.supports_vision),
                vision_max_edge_pixels=int(self.config.vision_max_image_edge_pixels),
                vision_jpeg_quality=int(self.config.vision_jpeg_quality),
            )
            # Snapshot the GLOBAL config's allowed roots (the tool manager loads them from
            # app.config.json). This is the fallback a chat with no per-chat workspace
            # override falls back to (and what reload_config refreshes from the settings panel).
            ws = getattr(self._tool_manager, "workspace", None)
            if ws is not None:
                self._base_workspace_roots = [str(r) for r in (ws.allowed_roots or []) if r]
            self.speech = SentenceSpeechPipeline(
                tts_synthesize=self._tts_synthesize,
                on_apply_emotion=self._send_apply_emotion,
                on_speaking_edge=self._on_speaking_edge,
                allowed_emotions=character.available_emotions,
                verbose=self.verbose,
            )
            self.orchestrator = ChatOrchestrator(
                config=self.config,
                llm_client=self.llm,
                tool_runner=self._tool_runner,
                tool_schemas=[],  # populated per-session by _refresh_tool_schemas below
                speech=self.speech,
                # The system prompt's {{environment}} line surfaces the workspace
                # root rather than the Python process cwd — pull it dynamically
                # so settings-panel edits are visible to the LLM on the next turn.
                workspace_root_getter=self._current_workspace_root,
                # Long-term memory injected into the system prompt each turn: a compact index of every
                # in-scope memory + the bodies most relevant to the current message (two-tier recall).
                memory_getter=self._memory_prompt_section,
                # Live "Writing report…" / "Searching the web…" labels come straight from
                # each tool's own activity_label (single source of truth).
                activity_label_resolver=self._tool_manager.activity_label,
                # Per-tool flag views (defer-until-speech, parallel-safe) come live from
                # the ToolManager so the orchestrator's checks can't drift from the
                # tools' own attributes (the old hardcoded set had exactly that bug).
                deferred_tools_getter=self._tool_manager.deferred_tools,
                concurrency_safe_getter=self._tool_manager.concurrency_safe_tools,
            )
            self.orchestrator.events = OrchestratorEvents(
                on_token=self._on_orch_token,
                on_emotion_changed=None,  # speech pipeline handles per-sentence; live bubble pulls from session
                on_tool_activity=self._on_orch_tool_activity,
                on_executed_tool=self._on_orch_executed_tool,
                on_turn_complete=self._on_orch_turn_complete,
                on_error=self._on_orch_error,
            )
        # ToolManager schemas are session-aware (ChangeOutfit reads outfit names
        # off the active character). Rebuild now that we have a session-bound
        # character.
        self._refresh_tool_schemas()

    def _current_workspace_root(self) -> str | None:
        """First allowed root of the live workspace config, or None if the tool
        manager isn't up yet / no roots are configured. Used by the orchestrator
        to render the system prompt's cwd."""
        if self._tool_manager is None or self._tool_manager.workspace is None:
            return None
        roots = self._tool_manager.workspace.allowed_roots
        return roots[0] if roots else None

    # ------------------------------------------------------------------------
    # Long-term memory (character + project). Stores live on the manager; the Remember / Forget /
    # RecallMemory tools call these, and _memory_prompt_section renders both scopes into the prompt.
    # ------------------------------------------------------------------------
    def _active_project_key(self) -> str | None:
        """The project-memory key for this chat = its primary workspace root path, or None when the
        chat has no workspace at all (→ no project memory, per design)."""
        return self._current_workspace_root() or None

    def _resolve_memory_scope(self, scope: str, session):
        """(store, key, label) for a scope, or (None, error_message, '') when it isn't available."""
        if scope == "character":
            cid = session.character.id if (session is not None and session.character is not None) else ""
            if not cid:
                return None, "No active character to attach this memory to.", ""
            return self._char_memory, cid, "character"
        if scope == "project":
            key = self._active_project_key()
            if not key:
                return None, "This chat has no workspace/project set, so there's no project memory.", ""
            return self._project_memory, key, "project"
        return None, f"Unknown memory scope '{scope}' (use 'character' or 'project').", ""

    def memory_add(self, scope: str, session, name: str, description: str, text: str) -> tuple[bool, str]:
        store, key, label = self._resolve_memory_scope(scope, session)
        if store is None:
            return False, key  # `key` carries the error message here
        e = store.add(key, name, description, text)
        return True, f"Saved to {label} memory: '{e.name}'."

    def memory_remove(self, scope: str, session, name: str) -> tuple[bool, str]:
        store, key, label = self._resolve_memory_scope(scope, session)
        if store is None:
            return False, key
        n = store.remove_by_name(key, name)
        if n == 0:
            return False, f"No {label} memory named '{name}'."
        return True, f"Forgot {label} memory '{name}'."

    def memory_get(self, scope: str, session, name: str):
        store, key, _ = self._resolve_memory_scope(scope, session)
        if store is None:
            return None
        return store.get_by_name(key, name)

    def _memory_prompt_section(self, session, query: str) -> str:
        """Render both in-scope memory blocks (index + relevant bodies) for the system prompt.
        Empty string when there's nothing to show. Called by the orchestrator each turn."""
        parts: list[str] = []
        if session is not None and session.character is not None and session.character.id:
            s = self._char_memory.render(
                session.character.id, query,
                header="What you remember about this character and your history with the user:")
            if s:
                parts.append(s)
        key = self._active_project_key()
        if key:
            s = self._project_memory.render(
                key, query,
                header="What you remember about the current project/workspace (shared across characters):")
            if s:
                parts.append(s)
        return "\n\n".join(parts)

    def _refresh_tool_schemas(self) -> None:
        """Rebuild the LLM-facing tool schema list off the live session. Called on
        every session change. The orchestrator reads this at the top of each
        round, but it doesn't change inside a turn so once per session is enough."""
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        if not hasattr(self, "_tool_manager") or self._tool_manager is None:
            return
        from ..models import ToolSchema as ToolSchemaWire
        entries = self._tool_manager.build_schemas(self.orchestrator.session)
        self.orchestrator.tool_schemas = [
            ToolSchemaWire(name=e.name, description=e.description, parameters=e.parameters)
            for e in entries
        ]

    def _on_orch_token(self, delta: str) -> None:
        """Stream a token. The first visible token of each round pushes a fresh
        assistant entry; subsequent tokens append deltas to it. Emotion tags
        ([LABEL]) are stripped via a per-round EmotionStreamFilter — the orchestrator
        stores raw tags in session.history (LLM context) but the renderer only ever
        sees clean text. The token also feeds the speech pipeline so TTS can run in
        parallel."""
        if self.voice_enabled and self.speech is not None:
            # Speech pipeline owns its own emotion filter for per-sentence emotion
            # binding; raw delta is fine here.
            self.speech.feed_token(delta)

        if self._needs_new_assistant_entry and self._text_filter is None:
            # New round — fresh filter so partial-tag state doesn't leak between rounds.
            labels = self.orchestrator.effective_emotion_labels() if self.orchestrator else []
            self._text_filter = EmotionStreamFilter(
                allowed_labels=labels or None,
                on_emotion=self._on_text_filter_emotion,
            )

        cleaned = self._text_filter.feed(delta) if self._text_filter else delta
        if not cleaned:
            # Filter consumed a partial tag — no visible output yet. Wait for more.
            return

        if self._needs_new_assistant_entry:
            # Start of a new assistant message: drop the whitespace left over after the
            # stripped leading [LABEL] tag (the model writes "[Neutral] text" / "[Neutral]\ntext"),
            # so neither the bubble nor lip-sync begins with a stray space/newline.
            cleaned = cleaned.lstrip()
            if not cleaned:
                # The whole chunk was that leftover whitespace — wait for the first real text.
                return

        # Voice off: feed the clean delta to Unity's text lip-sync (mirrors the speech pipeline
        # feed above, which only runs when voice is on).
        if not self.voice_enabled:
            self._send_envelope(T_LIPSYNC_TEXT_APPEND, {"text": cleaned})

        speaker = (self.orchestrator.session.display_name
                   if self.orchestrator and self.orchestrator.session else "Assistant")

        if self._needs_new_assistant_entry:
            # First visible token of this round → push an assistant entry seeded with
            # the cleaned chunk. Subsequent tokens append. This round's assistant
            # message is appended to session.history when the round completes, and
            # it's the very next append, so its eventual index is the current length.
            hist_len = (len(self.orchestrator.session.history)
                        if self.orchestrator and self.orchestrator.session else -1)
            self._push_entry(HistoryEntry(
                role="assistant",
                speaker=speaker,
                text=cleaned,
                can_rollback=False,
                turn_index=-1,
                reports=[],
                todos=None,
                history_index=hist_len,
            ))
            self._needs_new_assistant_entry = False
        else:
            self._push_append_token(cleaned)

    def _on_orch_tool_activity(self, label: str | None) -> None:
        self.push_tool_activity(label)

    def push_tool_activity(self, label: str | None) -> None:
        """Push a live activity label to the renderer. Public because the Agent tool
        overrides the label mid-call to surface the sub-agent's current step (the
        orchestrator's own None-clear after the tool returns wipes it either way)."""
        self._push_envelope(T_CHAT_TOOL_ACTIVITY, {"label": label or ""})

    def _on_orch_executed_tool(self, etc) -> None:
        # Tools may have mutated session.current_outfit. Mirror to Unity so the avatar
        # actually changes outfits.
        s = self.orchestrator.session if self.orchestrator else None
        if s is not None and s.current_outfit is not None:
            self._send_envelope(T_AVATAR_APPLY_OUTFIT, {
                "outfitName": s.current_outfit.outfit_name,
                # KK outfit id — Unity's KKOutfitController switches by this (name is for logging).
                "outfitIndex": s.current_outfit.index,
            })
            # Keep the renderer's worn-outfit marker in sync when the AI changed it.
            self._push_outfit_changed(s)
        if s is not None:
            self._send_envelope(T_AVATAR_SET_STATUS, {"text": s.current_status})

        # Round just ended (LLM emitted tool_calls → tool ran). Flush any trailing
        # buffered chars from the emotion filter into the still-current assistant
        # entry, then push the tool_activity event as the next entry.
        if self._text_filter is not None:
            tail = self._text_filter.flush()
            if tail and not self._needs_new_assistant_entry:
                self._push_append_token(tail)
            self._text_filter = None

        # Push the tool_activity event. The just-executed tool call lives on the most
        # recent assistant message in session.history (the one with tool_calls). Pull
        # its arguments + name for the human-readable label and widget routing.
        if s is not None:
            self._push_tool_activity_entry(s, etc)

        # The next round (if any) starts a brand-new assistant entry.
        self._needs_new_assistant_entry = True

    def _on_orch_turn_complete(self, reply: StructuredReply) -> None:
        # Flush any trailing buffered chars from the final round's filter into the
        # active assistant entry — same reason as in _on_orch_executed_tool.
        if self._text_filter is not None:
            tail = self._text_filter.flush()
            if tail and not self._needs_new_assistant_entry:
                self._push_append_token(tail)
            self._text_filter = None

    def _push_tool_activity_entry(self, session: ChatSession, etc) -> None:
        """Build and push the tool_activity entry for the tool that just finished.
        We look at the most recent assistant message in session.history (always the
        one whose tool_calls included this call) to find the matching call and pull
        its arguments out for the label."""
        speaker = session.display_name
        # The orchestrator records ExecutedToolCall.name + .arguments (dict). We use
        # those directly rather than fishing through history.
        name = getattr(etc, "name", None) or "tool"
        args = getattr(etc, "arguments", None)
        if not isinstance(args, dict):
            args = {}
        label = _tool_activity_label(name, args)

        # Attach widgets if this tool produced them. register_report / register_todo_snapshot
        # set history_index to the latest assistant message at registration time, which is
        # the one whose tool_calls list contains the just-executed call. Multiple WriteReport
        # calls in one round each get their own row, so we pop the first matching report.
        latest_asst_idx = -1
        for i in range(len(session.history) - 1, -1, -1):
            m = session.history[i]
            if m is not None and m.role == "assistant":
                latest_asst_idx = i
                break

        reports_here: list[ReportRef] = []
        todos_here: list[TodoItemRef] | None = None
        if name == "ReportWrite":
            for r in self._reports:
                if r.history_index == latest_asst_idx and r.id not in (e.id for e in reports_here):
                    if not any(e.id == r.id for e in reports_here):
                        reports_here.append(ReportRef(id=r.id, title=r.title))
                        break  # one report per ReportWrite call
        elif name == "TodoWrite":
            # Pick the most recent snapshot for this assistant index.
            for t in reversed(self._todo_snapshots):
                if t.history_index == latest_asst_idx:
                    todos_here = [TodoItemRef(content=it.content, active_form=it.active_form,
                                              status=it.status) for it in t.items]
                    break

        self._push_entry(HistoryEntry(
            role="tool_activity",
            speaker=speaker,
            text=label,
            can_rollback=False,
            turn_index=-1,
            reports=reports_here,
            todos=todos_here,
            tool_name=name,
        ))

    def _on_orch_error(self, message: str) -> None:
        self._push_error("ai_error", message)

    async def _tts_synthesize(self, text: str) -> None:
        """Called by the speech pipeline for each sentence. Delegates to the
        app-injected coroutine that wraps tts.TtsController.synthesize_text."""
        await self._tts_synthesize_fn(text)

    def _on_text_filter_emotion(self, label: str, position: int) -> None:
        """A [LABEL] was parsed from the renderer-clean text stream. In VOICE-OFF mode, send a
        Lipsync.EmotionMarker — it's emitted right before that sentence's Lipsync.TextAppend, so Unity
        anchors it to the current text-buffer position and fires it when the mouth's char cursor
        reaches that sentence (paced to charactersPerSecond), instead of racing ahead. In voice mode
        the speech pipeline owns per-sentence emotion (Tts.EmotionMarker), so this is a no-op to avoid
        double-driving the avatar. (`position` is the cleaned-stream index; unused — Unity snapshots
        its own buffer position, which is turn-continuous across rounds.)"""
        if self.voice_enabled:
            return
        self._send_envelope(T_LIPSYNC_EMOTION_MARKER, {"label": label})
        self._push_envelope(T_CHAT_EMOTION, {"label": label})

    def _send_apply_emotion(self, label: str) -> None:
        # This is the speech pipeline's per-sentence callback (voice mode). Send a playback-synced
        # marker so Unity applies the emotion when the sentence's audio actually plays, not the instant
        # the [LABEL] is parsed (which races ahead of the slower audio). It's emitted right before the
        # sentence's TTS chunks, so Unity can anchor it to the buffer's current write position.
        if self.voice_enabled:
            self._send_envelope(T_TTS_EMOTION_MARKER, {"label": label})
        else:
            # No TTS audio clock to sync against — apply immediately.
            self._send_envelope(T_AVATAR_APPLY_EMOTION, {"label": label})
        # Also push to the renderer for any UI that wants to mirror it.
        self._push_envelope(T_CHAT_EMOTION, {"label": label})

    def _on_speaking_edge(self, active: bool) -> None:
        # Renderer's speaking indicator + button-state toggle.
        self._push_envelope(T_CHAT_SPEAKING, {"active": active})

    async def _tool_runner(self, name: str, args: dict, tool_call_id: str) -> ToolExecutionResult:
        """Dispatches the tool call through the Python ToolManager. Tools run in
        the chat asyncio loop; long-running ones (web fetches, subprocesses) yield
        properly so the LLM stream stays responsive."""
        s = self.orchestrator.session if self.orchestrator else None
        if s is None or not hasattr(self, "_tool_manager") or self._tool_manager is None:
            return ToolExecutionResult(
                result_text="Error: tool manager not initialized.",
                error="no tool manager",
            )
        result = await self._tool_manager.dispatch(name, args, s)
        return ToolExecutionResult(
            result_text=result.result_text,
            session_mutations=result.session_mutations,
            pending_attachments=result.pending_attachments,
            error=result.error,
        )

    async def _request_with_correlation(self, env_type: str, payload: dict,
                                        bucket: dict[str, asyncio.Future[dict]]) -> dict:
        """Sends an envelope to Unity, returns the resolved reply payload. Correlation
        works via `handle_unity_envelope`: when a Tool.Result / *Result envelope arrives
        with the matching `replyTo`, it pops the future from the bucket and resolves it.

        The injected `send_unity_request` returns the envelope id we put on the wire so
        we can register the future under it before Unity has a chance to reply."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict] = loop.create_future()
        req_id = self._send_request(env_type, payload)
        if not isinstance(req_id, str) or not req_id:
            print(f"[ChatManager] send_unity_request returned no id for {env_type}", file=sys.stderr)
            return {}
        bucket[req_id] = fut
        try:
            return await fut
        except asyncio.CancelledError:
            bucket.pop(req_id, None)
            raise
