"""Combined port of Assets/Scripts/Chat/ChatManager.cs + ChatUIController.cs.

ChatManager is the glue between:

  * the renderer (chat window) — incoming Chat.SubmitUserMessage / Chat.LoadSave / etc.,
    outgoing Chat.Init / Chat.UpdateBubble / Chat.History / etc.
  * the orchestrator — runs the LLM/tool loop, fires events as tokens stream
  * the avatar (Unity) — pushes Session.Begin / Avatar.ApplyEmotion / Avatar.ApplyOutfit
    so the 3d character mirrors the current chat state
  * disk — SaveLoadManager persists each turn

The C# original is 1967 lines. This Python port carries over the load-bearing parts:
session lifecycle, streaming tokens, history rebuild, save/load, voice mode, error
surfacing. A few features are deferred (noted inline): mid-stream rollback edge cases,
the more elaborate todo-snapshot dedup logic that lives inside the tool, the per-turn
report rebinding after a save/reload (the renderer can still open reports, but the
"view report" buttons re-attach on the post-resume history rebuild rather than per
inline turn). The follow-up todos in the migration list cover those.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

from ..character_store import CharacterStore, CharacterRecord, new_character_id
from ..memory_store import CharacterMemoryStore, ProjectMemoryStore
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
from .protocol import *  # noqa: F401,F403
from .view_models import (
    HistoryEntry, ReportRef, TodoItemRef,
    _tool_activity_label, _leading_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
)
from ._session import SessionMixin
from ._turn import TurnMixin
from ._library import LibraryMixin
from ._history import HistoryMixin
from ._orchestration import OrchestrationMixin
from ._tasks import BgTaskMixin


class ChatManager(SessionMixin, TurnMixin, LibraryMixin, HistoryMixin, OrchestrationMixin,
                  BgTaskMixin):
    """One instance per process. Owns the active session, orchestrator, speech pipeline,
    and the bookkeeping for reports / todos / turn snapshots so save+load works.

    The implementation is split across responsibility-focused mixins (see the imports
    above); this module holds construction, the envelope-dispatch entry points, and the
    low-level send/push helpers everything else builds on."""

    def __init__(
        self,
        config: ChatBackendConfig,
        save_manager: SaveLoadManager,
        send_to_unity: SendToUnityFn,
        push_to_renderer: PushToRendererFn,
        send_unity_request: SendUnityRequestFn,
        tts_synthesize: TtsSynthesizeFn,
        schedule: ScheduleFn,
        open_modal: OpenModalFn,
        event_loop: asyncio.AbstractEventLoop,
        llm_registry=None,
        ask_modal=None,
        tts_cancel: TtsCancelFn | None = None,
        image_processor=None,
        ocr_processor=None,
        voice_supported: bool = True,
        verbose: bool = False,
        encode_pocket_voice=None,
        voice_available=None,
        set_voice_provider=None,
        voice_provider_getter=None,
        tts_ready_getter=None,
        stt_ready_getter=None,
    ):
        self.config = config
        # Registry of named LLM configs (chat.config.LlmConfigRegistry). A chat picks one by id;
        # _apply_llm_config copies its fields into self.config in place. None → single static config.
        self._llm_registry = llm_registry
        # Id of the LLM config the ACTIVE chat uses ("" = follow the registry default). Persisted
        # per chat in ChatSaveData; chosen in the new-chat dialog / chat settings.
        self._llm_config_id: str = ""
        self.save = save_manager
        self._send = send_to_unity
        self._push = push_to_renderer
        self._send_request = send_unity_request
        self._tts_synthesize_fn = tts_synthesize
        self._tts_cancel_fn = tts_cancel
        self._schedule_fn = schedule
        self._open_modal_fn = open_modal
        # Coroutine: spawn a task window and await its reply payload dict. Used
        # by the Python tool runner for AskUserQuestion + approval gate.
        self._ask_modal_fn = ask_modal
        # Sync helpers for Read's image branch (Pillow vision encode / RapidOCR).
        # See ToolContext.image_processor / ocr_processor for the call signature.
        self._image_processor_fn = image_processor
        self._ocr_processor_fn = ocr_processor
        # The asyncio loop async methods run on. handle_unity_envelope is called from the
        # WS reader thread; we marshal future resolutions onto this loop via
        # call_soon_threadsafe so the awaiter actually wakes up (raw set_result from a
        # foreign thread is undefined behavior and silently no-ops in practice).
        self._loop = event_loop
        self.voice_supported = voice_supported
        self.verbose = verbose
        self.voice_enabled = True
        # Touch mode (avatar caress). Transient — not persisted per chat; resets off on every
        # session (re)init since a freshly loaded model is never in touch mode. Unity is the
        # source of truth and echoes changes back via Touch.ModeChanged.
        self.touch_enabled = False
        # Head/eye look-at tracking — per-chat persisted (default on). Set from the saved chat /
        # create payload, toggled live from the sidebar, sent to Unity in Session.Begin + Chat.Set*.
        self.head_tracking_enabled = True
        self.eye_tracking_enabled = True
        self.current_slot = save_manager.default_slot

        # Lazy: created when the first session begins (so we can wait until tool schemas
        # have been fetched). The same instance is reused across sessions.
        self.llm = LlmClient(config.llm)
        # Resolve externalized `saved-image:` refs to base64 only when a request is built, so
        # image bytes are read from disk at send time (not on every chat load).
        self.llm.image_resolver = self.save.resolve_saved_image_url
        self.orchestrator: ChatOrchestrator | None = None
        self.speech: SentenceSpeechPipeline | None = None

        # Per-session UI bookkeeping. The C# ChatManager tracks pending report ids per turn
        # so they attach to the right assistant bubble; this port keeps the same shape.
        self._reports: list[ReportEntry] = []
        self._todo_snapshots: list[TodoSnapshotEntry] = []
        self._turn_snapshots: list[TurnSnapshot] = []
        self._pending_report_ids_this_turn: list[str] = []
        self._pending_todo_snapshot_id_this_turn: str | None = None
        # Folders granted to the workspace sandbox via file attachments (persisted per slot so
        # tool access to attached files survives a reload). See _grant_workspace_access.
        self._attachment_roots: list[str] = []
        # Per-chat settings (chosen in the new-chat dialog, persisted in ChatSaveData):
        #   _voice_provider       — active TTS engine for this chat ("" = follow global config)
        #   _chat_workspace_roots — workspace folders for this chat ([] = follow global config)
        #   _base_workspace_roots — snapshot of the GLOBAL config roots, captured when the tool
        #                           manager is first built; the fallback when a chat has none.
        self._voice_provider: str = ""
        self._chat_workspace_roots: list[str] = []
        self._base_workspace_roots: list[str] = []
        # App-injected: switch the active TTS provider by name ("pocket"/"elevenlabs"), and
        # read the active provider's key. Used to give each chat its own voice engine.
        self._set_voice_provider_fn = set_voice_provider
        self._voice_provider_getter = voice_provider_getter
        # Chat-loading overlay readiness gates: `() -> bool`, True once TTS / STT have resolved
        # (loaded or terminally errored). None → that subsystem is treated as always ready.
        self._tts_ready_getter = tts_ready_getter
        self._stt_ready_getter = stt_ready_getter

        # Streaming state. There's no "live bubble" — assistant entries are pushed
        # to history on the first visible token of each round and grown via
        # Chat.AppendToken. `_needs_new_assistant_entry` is True at turn start AND
        # right after each tool runs, so the next visible token (a) creates a fresh
        # entry and (b) gets a fresh emotion-tag filter (so a [LABEL] split across
        # chunks doesn't leak across rounds).
        self._needs_new_assistant_entry: bool = False
        self._text_filter: EmotionStreamFilter | None = None

        # Pending Character.* RPC replies from Unity (now just Character.InspectModelEmotions).
        self._character_futures: dict[str, asyncio.Future[dict]] = {}
        # Pending Character.CaptureView (self-view screenshot) RPC replies from Unity.
        self._capture_futures: dict[str, asyncio.Future[dict]] = {}
        # Tool manager (Python). Constructed in _ensure_orchestrator on first use.
        self._tool_manager = None  # type: ignore[assignment]

        # Cached characters list (wire records; refreshed on demand).
        self._characters_cached: list[dict] | None = None

        # Cached per-character data (keyed by character id) so repeated Chat.LoadSave on the same
        # character avoids re-reading the character store.
        self._character_cache: dict[str, CharacterInfo] = {}

        # App-owned character library: one folder per character (characters/<charId>/)
        # holding character.json + the app-owned model copy (model_<hash>.<ext>, made at
        # save time so the character keeps working if the picked file moves) + the pocket
        # voice embedding (voice_<hash>.npy). Unity is just the renderer.
        self._character_store = CharacterStore()

        # Long-term memory (two scopes). Character memory is keyed by character id; project memory
        # by the active workspace root path (shared across characters). Written via the Remember /
        # Forget tools, injected into the system prompt each turn (see _memory_prompt_section).
        self._char_memory = CharacterMemoryStore()
        self._project_memory = ProjectMemoryStore()

        # Pocket-TTS voice encoding. Embeddings live in each character's folder
        # (voice_<hash>.npy). The encoder is `Callable[[clip_path, out_path], None]`;
        # None disables pocket voice encoding.
        self._encode_pocket_voice_fn = encode_pocket_voice
        # `Callable[[voices_dict], bool]` — does the character have a voice for the active
        # TTS provider. Gates voice mode: with no voice we force/keep voice-off (the
        # assistant lip-syncs to text instead of speaking). None → assume always available.
        self._voice_available_fn = voice_available

        # In-progress turn, so Chat.Stop can find and cancel it.
        self._current_turn_task: asyncio.Task | None = None

        # Touch (KK Aibu) state. `_touch_busy` blocks a second touch turn until the first finishes
        # streaming (so taps don't pile up). `_touch_active` is set once a touch turn runs and cleared
        # when the player next types — driving the one-shot "touching has ended" note prepended to that
        # typed turn so the LLM knows the caress is over.
        self._touch_busy: bool = False
        self._touch_active: bool = False

        # Background UwU helpers (see _tasks.py): live task records keyed by id, plus the
        # ids whose reports are waiting to be folded into a notification turn. Per-session,
        # not persisted — _reset_session_bookkeeping dismisses everything.
        self._bg_tasks: dict = {}
        self._bg_notify_queue: list[str] = []

        # TTS playback stream bracketing (Tts.StreamBegin/End) for assistant responses — turns AND the
        # greeting. `_tts_stream_gen` identifies the latest stream; `_tts_stream_open` tracks whether one
        # is live. A response that's superseded by a newer one (e.g. the user fires a message while the
        # previous reply/greeting is still speaking) is interrupted cleanly in _begin_tts_stream, and its
        # trailing _end_tts_stream is dropped (gen mismatch) so it can't cut the new stream's audio.
        self._tts_stream_gen = 0
        self._tts_stream_open = False

        # Chat-loading overlay: the model-ready signal from Unity (Session.Ready) for the CURRENT
        # session. Recreated per new/resume session (see _session._on_create_new/_on_load_save) and
        # gated by `_session_epoch` so a stale Session.Ready from a prior session can't satisfy it.
        self._model_ready_event: asyncio.Event | None = None
        self._session_epoch = 0

        # Per-chat Chat-mode view (camera zoom/pan/rotation + window position). Opaque dict reported by
        # Unity (Chat.ViewState); persisted with the chat and echoed back in Session.Begin to restore it.
        self._chat_view: dict = {}

    # ------------------------------------------------------------------------
    # Envelope dispatch (renderer → us via send_chat; Unity → us via the WS reader)
    # ------------------------------------------------------------------------

    def handle_renderer_chat(self, env: dict) -> None:
        """Routes a Chat.* envelope coming FROM the renderer."""
        t = env.get("type")
        payload = env.get("payload") or {}
        if t == T_CHAT_SUBMIT:
            atts = payload.get("attachments")
            atts = [str(p) for p in atts if isinstance(p, str) and p] if isinstance(atts, list) else []
            self._schedule(self._on_player_submit(str(payload.get("text") or ""), atts))
        elif t == T_CHAT_STOP:
            self._on_stop()
        elif t == T_CHAT_RESTART:
            self._schedule(self._on_restart())
        elif t == T_CHAT_ROLLBACK:
            self._schedule(self._on_rollback(int(payload.get("turnIndex", -1))))
        elif t == T_CHAT_EDIT_MESSAGE:
            self._on_edit_message(
                int(payload.get("historyIndex", -1)),
                str(payload.get("text") or ""),
                removed_attachments=[int(i) for i in (payload.get("removedAttachments") or [])
                                     if isinstance(i, (int, float))],
                turn_index=int(payload.get("turnIndex", -1)),
            )
        elif t == T_CHAT_OPEN_REPORT:
            self._on_open_report(str(payload.get("reportId") or ""))
        elif t == T_CHAT_CHANGE_OUTFIT:
            self._schedule(self._on_change_outfit(int(payload.get("outfitIndex", -1))))
        elif t == T_CHAT_DISMISS_BG_TASK:
            self._schedule(self._on_dismiss_bg_task(str(payload.get("taskId") or "")))
        elif t == T_CHAT_SET_VOICE_MODE:
            self._on_set_voice_mode(bool(payload.get("enabled", False)))
        elif t == T_CHAT_SET_TOUCH_MODE:
            self._on_set_touch_mode(bool(payload.get("enabled", False)))
        elif t == T_CHAT_SET_HEAD_TRACKING:
            self._on_set_head_tracking(bool(payload.get("enabled", True)))
        elif t == T_CHAT_SET_EYE_TRACKING:
            self._on_set_eye_tracking(bool(payload.get("enabled", True)))
        elif t == T_CHAT_LIST_SAVES:
            self.push_saves_list()
        elif t == T_CHAT_GET_CHARACTERS:
            self._schedule(self.push_characters_list())
        elif t == T_CHAT_LOAD_SAVE:
            self._schedule(self._on_load_save(str(payload.get("slot") or "")))
        elif t == T_CHAT_CREATE_NEW:
            self._schedule(self._on_create_new(
                str(payload.get("characterId") or ""),
                str(payload.get("slot") or "") or None,
                payload,
            ))
        elif t == T_CHAT_DELETE_SAVE:
            self._on_delete_save(str(payload.get("slot") or ""))
        elif t == T_CHAT_END_SESSION:
            self._on_end_session()
        elif t == T_CHAT_UPDATE_SETTINGS:
            self._schedule(self._on_update_settings(payload))
        elif t == T_CHAT_SAVE_CHARACTER:
            self._schedule(self._on_save_character(payload))
        elif t == T_CHAT_DELETE_CHARACTER:
            self._schedule(self._on_delete_character(str(payload.get("id") or "")))
        elif t == T_CHAT_IMPORT_CHARACTER:
            self._schedule(self._on_import_character(str(payload.get("path") or "")))
        elif t == T_CHAT_REBIND_SAVE:
            self._schedule(self._on_rebind_save(
                str(payload.get("slot") or ""), str(payload.get("characterId") or "")))
        elif t == T_CHAT_INSPECT_MODEL_EMOTIONS:
            self._schedule(self._on_inspect_model_emotions(str(payload.get("modelPath") or "")))
        elif t == T_CHAT_INSPECT_MODEL_COORDINATES:
            self._schedule(self._on_inspect_model_coordinates(str(payload.get("modelPath") or "")))

    def handle_unity_envelope(self, env: dict) -> None:
        """Routes an envelope coming FROM Unity over the WS that ChatManager cares about.
        Called from the WS reader thread — uses `_resolve_future` to marshal future
        resolutions back onto the chat loop (asyncio futures can only be resolved on
        their owning loop).

        After the tool migration this handler is small: Character.*Result for the
        scene-side character/outfit data + Tts playback edges. Tools live entirely
        in Python now, so no Tool.* envelopes round-trip through here anymore."""
        t = env.get("type")
        reply_to = env.get("replyTo")
        payload = env.get("payload") or {}

        if t == T_CHARACTER_INSPECT_MODEL_EMOTIONS_RESULT and reply_to:
            fut = self._character_futures.pop(reply_to, None)
            self._resolve_future(fut, payload)
            return
        if t == T_CHARACTER_CAPTURE_VIEW_RESULT and reply_to:
            fut = self._capture_futures.pop(reply_to, None)
            self._resolve_future(fut, payload)
            return
        if t == T_SESSION_READY:
            # Unity finished the Session.Begin work (model load + bind, or skip/failure). Resolve the
            # current session's model-ready gate. The epoch match + event set must happen on the chat
            # loop (where _session_epoch / _model_ready_event are mutated), so marshal a closure rather
            # than checking here on the WS reader thread — otherwise a session swap mid-check could let a
            # stale ready satisfy the new session.
            try:
                epoch = int(payload.get("sessionEpoch", -1))
            except (TypeError, ValueError):
                epoch = -1
            self._loop.call_soon_threadsafe(self._resolve_model_ready, epoch)
            return
        if t == T_CHAT_VIEW_STATE:
            # The user settled a Chat-mode framing change. Store + persist it for the active chat on the
            # chat loop (persist touches the session + does file I/O). Persisting now — not just on the
            # next turn — is what lets a chat switch with no message in between keep the framing.
            view = payload if isinstance(payload, dict) else {}
            self._loop.call_soon_threadsafe(self._on_chat_view_state, view)
            return
        if t == T_TOUCH_EVENT:
            # The player started caressing the avatar. Fold it into an invisible LLM turn (the AI reacts
            # in character; the reply may carry a rejection marker → Avatar.EndTouch). Marshal onto the
            # chat loop — we're on the WS reader thread.
            zone = str(payload.get("zone") or "")
            self._schedule(self._on_touch_event(zone))
            return
        if t == T_TOUCH_MODE_CHANGED:
            # Unity's touch mode changed on its own (keyboard toggle, or auto-off after an AI rejection).
            # Mirror it to the renderer so the header toggle stays in sync. Safe to push from this thread
            # (Tts edges above do the same). No echo back to Unity — Unity is already in this state.
            enabled = bool(payload.get("enabled", False))
            self.touch_enabled = enabled
            self._push_envelope(T_CHAT_TOUCH_MODE_CHANGED, {"enabled": enabled})
            return
        if t == T_TTS_PLAYBACK_STARTED:
            if self.speech is not None:
                self.speech.notify_playback_started()
            self._push_envelope(T_CHAT_SPEAKING, {"active": True})
            return
        if t == T_TTS_PLAYBACK_ENDED:
            if self.speech is not None:
                self.speech.notify_playback_ended()
            self._push_envelope(T_CHAT_SPEAKING, {"active": False})
            return

    def _on_chat_view_state(self, view: dict) -> None:
        """Store the latest Chat-mode framing for the active chat and persist it immediately (runs on
        the chat loop). Immediate persist means switching chats without sending a message still keeps
        the framing. No-op persist when there's no active session."""
        self._chat_view = view
        if self.orchestrator is not None and self.orchestrator.session is not None:
            self._persist()

    def _resolve_model_ready(self, epoch: int) -> None:
        """Set the model-ready gate for the CURRENT session. Runs on the chat loop (via
        call_soon_threadsafe from handle_unity_envelope), so the epoch comparison and the event are
        read consistently — a Session.Ready echoing a stale epoch is ignored."""
        if epoch == self._session_epoch and self._model_ready_event is not None:
            self._model_ready_event.set()

    def _resolve_future(self, fut: "asyncio.Future[dict] | None", payload: dict) -> None:
        """Resolve an asyncio.Future from a non-loop thread. Safe no-op when the future
        is None or already done (e.g. cancelled, or timed out before the reply arrived)."""
        if fut is None:
            return

        def _setter():
            if not fut.done():
                fut.set_result(payload)

        try:
            self._loop.call_soon_threadsafe(_setter)
        except RuntimeError as e:
            # Loop closed (shutdown) — drop quietly.
            print(f"[ChatManager] _resolve_future: loop unavailable: {e}", file=sys.stderr)

    # ------------------------------------------------------------------------
    # Low-level envelope helpers (used by every mixin)
    # ------------------------------------------------------------------------

    def _send_envelope(self, env_type: str, payload: dict | None) -> None:
        self._send({"id": "m_" + uuid.uuid4().hex, "type": env_type, "payload": payload or {}})

    # ------------------------------------------------------------------------
    # TTS playback stream bracketing (shared by turns + greeting)
    # ------------------------------------------------------------------------

    def _begin_tts_stream(self) -> int:
        """Open a fresh TTS playback stream for an assistant response and return its generation id.

        If a previous response is still speaking (the user sent a new message — or fired a turn while
        the greeting was still playing), interrupt it cleanly first: abort its in-flight synth and cut
        its audio on the Unity side, so nothing bleeds into the new stream. The superseded response's
        trailing _end_tts_stream is dropped via the generation check, so its late Tts.StreamEnd can't
        close THIS stream's buffer early. The caller pairs this with _end_tts_stream(gen)."""
        if self._tts_stream_open:
            if self._tts_cancel_fn is not None:
                try:
                    self._tts_cancel_fn("superseded by new response")
                except Exception as e:
                    print(f"[ChatManager] tts cancel (supersede) failed: {e}", file=sys.stderr)
            self._send_envelope("Tts.Cancel", {"reason": "superseded"})
        self._tts_stream_gen += 1
        self._tts_stream_open = True
        if self.speech is not None:
            self.speech.begin_session()
        self._send_envelope("Tts.StreamBegin", {})
        return self._tts_stream_gen

    def _end_tts_stream(self, gen: int) -> None:
        """Close the TTS stream opened by _begin_tts_stream — but only if it's still the latest one.
        A response that was superseded (newer gen) drops its StreamEnd so it can't cut the live
        stream; one that was hard-stopped (Chat.Stop clears _tts_stream_open) also skips it."""
        if gen == self._tts_stream_gen and self._tts_stream_open:
            self._tts_stream_open = False
            self._send_envelope("Tts.StreamEnd", {})

    def _push_envelope(self, env_type: str, payload: dict | None) -> None:
        self._push({"id": "m_" + uuid.uuid4().hex, "type": env_type, "payload": payload or {}})

    def _push_entry(self, entry: HistoryEntry) -> None:
        """Append one entry to the renderer's history list."""
        self._push_envelope(T_CHAT_PUSH_ENTRY, {"entry": entry.to_wire()})

    def _push_append_token(self, delta: str) -> None:
        """Append a text delta to the last (assistant) entry in the renderer's
        history list. Tags are already stripped on this side."""
        if not delta:
            return
        self._push_envelope(T_CHAT_APPEND_TOKEN, {"delta": delta})

    def _push_error(self, code: str, message: str) -> None:
        self._push_envelope(T_CHAT_ERROR, {"code": code, "message": message})

    # ------------------------------------------------------------------------
    # Voice availability — a character can only run in voice mode if it has a
    # voice for the active TTS provider. Otherwise voice mode is forced/kept off
    # (the assistant lip-syncs to text instead of speaking).
    # ------------------------------------------------------------------------

    def voice_available(self) -> bool:
        """Whether the active session's character has a voice for the active provider.
        True when no checker is wired (voice assumed available)."""
        if self._voice_available_fn is None:
            return True
        session = self.orchestrator.session if self.orchestrator else None
        if session is None:
            return False
        return bool(self._voice_available_fn(session.character.voices or {}))

    def _enforce_voice_availability(self) -> bool:
        """Force voice mode off when the current character has no voice. Returns the
        resulting availability so callers can include it in outbound payloads."""
        available = self.voice_available()
        if not available and self.voice_enabled:
            self.voice_enabled = False
            if self.speech is not None:
                self.speech.stop_session()
        return available

    async def refresh_voice_availability(self) -> None:
        """Re-gate voice mode and tell the renderer/Unity — used after the active TTS
        provider changes (a character may gain or lose a voice for the new provider)."""
        available = self._enforce_voice_availability()
        self._send_envelope(T_CHAT_SET_VOICE_MODE, {"enabled": self.voice_enabled})
        self._push_envelope(T_CHAT_VOICE_MODE_CHANGED,
                            {"enabled": self.voice_enabled, "available": available})

    def _schedule(self, coro) -> None:
        """Fire-and-forget for renderer-triggered work. Routes through the
        app-injected scheduler so the coroutine runs on the chat event loop
        even when the caller (WS reader thread, pywebview JS bridge) is on a
        different thread."""
        self._schedule_fn(coro)
