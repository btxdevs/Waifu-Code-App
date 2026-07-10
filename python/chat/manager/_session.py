"""Session lifecycle: bootstrap, create/load/delete chats, restart, resync, config reload, persistence."""
from __future__ import annotations

import asyncio
import json
import os
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
    _tool_activity_label, _leading_emotion_tags, strip_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
)


class SessionMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    async def bootstrap(self) -> None:
        """Called once when the renderer signals it's mounted. The renderer opens on its HOME page (a
        character + saved-chat picker), so bootstrap does NOT auto-load a session — it only primes the
        home page with the character list and the saved-chats list. A chat session is created lazily
        when the user picks a character (Chat.CreateNew) or resumes a save (Chat.LoadSave), at which
        point a real Chat.Init is pushed and the renderer switches into the chat view."""
        print("[chat] bootstrap starting — priming home page", file=sys.stderr)
        try:
            await self.push_characters_list()
            self.push_saves_list()
        except Exception as e:
            print(f"[chat] bootstrap failed: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            self._push_error("bootstrap_failed", str(e))

    async def _on_create_new(self, character_id: str, slot: str | None,
                             settings: dict | None = None) -> None:
        if not character_id:
            return
        settings = settings or {}
        self.current_slot = slot or self.save.default_slot
        character = await self._fetch_character(character_id)
        if character is None:
            self._push_error("character_missing", f"Character '{character_id}' not found.")
            return  # no Chat.Init → renderer stays on the home page
        await self._ensure_orchestrator(character)
        # Per-chat LLM config: apply BEFORE begin_session so the first turn (and greeting) use the
        # chosen model/endpoint. "" falls back to the registry default.
        self._llm_config_id = str(settings.get("llmConfigId") or "").strip()
        self._apply_llm_config(self._llm_config_id)
        # Per-chat settings from the new-chat dialog; fall back to the global config defaults.
        user_name = str(settings.get("userName") or "").strip() or self.config.user_name
        session = self.orchestrator.begin_session(character, user_name=user_name)  # type: ignore[union-attr]
        # ChangeOutfit's schema reads the outfit list off the live session, so we have
        # to rebuild schemas AFTER begin_session attaches the session to the orchestrator
        # — _ensure_orchestrator runs before that and produces an empty list otherwise.
        self._refresh_tool_schemas()
        self._reset_session_bookkeeping()  # also revokes any prior attachment grants
        # Apply this chat's voice mode / provider / workspace before the first turn + persist.
        self.voice_enabled = bool(settings.get("voiceMode", True))
        self.head_tracking_enabled = bool(settings.get("headTracking", True))
        self.eye_tracking_enabled = bool(settings.get("eyeTracking", True))
        self._voice_provider = str(settings.get("voiceProvider") or "").strip().lower()
        self._chat_workspace_roots = [str(r) for r in (settings.get("workspaceRoots") or [])
                                      if isinstance(r, str) and r]
        # New chat → no saved camera/window framing yet; Unity uses its default centered framing.
        self._chat_view = {}
        self._ensure_voice_provider(self._voice_provider)
        self._apply_workspace_roots(self._chat_workspace_roots)
        # Save immediately so the slot shows up in the saves list right away.
        self._persist()
        # The greeting is delivered as the assistant's first response once the chat is ready (it's not
        # in history yet — begin_session no longer seeds it), so Chat.Init opens to an empty chat and
        # the greeting then streams in with voice. Resumed chats don't do this (their saved history
        # already holds it).
        gmsg = self.orchestrator.make_greeting_message(session)
        greeting = gmsg.content if gmsg is not None else None
        # Block the chat behind the loading overlay until the model + TTS + STT are ready. Arm BEFORE
        # Session.Begin (so the epoch we send matches the gate), push Chat.Loading BEFORE Chat.Init (so
        # the overlay is up the instant the renderer switches to the chat view), then poll to Chat.Ready.
        self._arm_chat_loading()
        self._send_session_begin(session)
        self._push_envelope(T_CHAT_LOADING, {"model": False, "tts": False, "stt": False})
        self._push_chat_init(session)
        self._schedule(self._await_chat_ready(self._session_epoch, greeting=greeting))

    async def _on_load_save(self, slot: str) -> None:
        if not slot:
            return
        saved = self.save.load(slot)
        if saved is None:
            self._push_error("save_missing", f"Save slot '{slot}' not found.")
            return
        self.current_slot = slot
        character = await self._fetch_character(saved.character_id)
        if character is None:
            # Not a dead end: the character was deleted (e.g. a duplicate cleanup after an
            # import). Open the renderer's re-bind dialog so the chat can continue with
            # another character instead of being orphaned forever.
            self._push_envelope(T_CHAT_CHARACTER_MISSING, {
                "slot": slot,
                "characterName": saved.character_name or "",
            })
            return  # no Chat.Init → renderer stays on the home page
        await self._ensure_orchestrator(character)
        # Restore this chat's LLM config (apply before resume so the first turn uses it).
        self._llm_config_id = str(saved.llm_config_id or "").strip()
        self._apply_llm_config(self._llm_config_id)
        # Build the resume save object the orchestrator expects.
        from ..orchestrator import ChatSaveData as OrcSaveData
        resume = OrcSaveData(
            character_name=saved.character_name,
            user_name=saved.user_name,
            history=list(saved.history),
            current_outfit_name=saved.outfit_name,
            current_outfit_index=saved.outfit_index,
            current_status=saved.current_status,
            emotion_labels=list(saved.emotion_labels),
        )
        session = self.orchestrator.resume_session(character, resume)  # type: ignore[union-attr]
        # user_name is now a per-chat setting (chosen in the new-chat dialog), so honor the
        # saved value; fall back to the global config for older saves that never stored one.
        session.user_name = saved.user_name or self.config.user_name
        # Rebuild schemas now that session.character is attached (see _on_create_new).
        self._refresh_tool_schemas()
        self._reports = list(saved.reports)
        self._todo_snapshots = list(saved.todo_snapshots)
        self._turn_snapshots = list(saved.turn_snapshots)
        self._pending_report_ids_this_turn.clear()
        self._pending_todo_snapshot_id_this_turn = None
        self._touch_active = False
        # Restore this chat's voice mode / provider / workspace roots.
        self.voice_enabled = bool(saved.voice_mode)
        self.head_tracking_enabled = bool(saved.head_tracking)
        self.eye_tracking_enabled = bool(saved.eye_tracking)
        self._voice_provider = str(saved.voice_provider or "").strip().lower()
        self._chat_workspace_roots = list(saved.workspace_roots)
        # Restore this chat's saved Chat-mode framing (sent to Unity in Session.Begin below). Setting it
        # now also means a persist right after load re-saves the same value rather than blanking it.
        self._chat_view = dict(saved.chat_view)
        self._ensure_voice_provider(self._voice_provider)
        # Workspace roots: per-chat list REPLACES the global config roots. Revoke the outgoing
        # chat's attachment grants first (so they don't leak across a switch), set the per-chat
        # base, then re-apply this chat's saved attachment grants on top.
        self._revoke_attachment_roots()
        self._apply_workspace_roots(self._chat_workspace_roots)
        for root in saved.extra_workspace_roots:
            self._grant_root(root)
        self._reset_streaming_state()
        # Same loading-overlay gate as a new chat (see _on_create_new): block until model+TTS+STT ready.
        self._arm_chat_loading()
        self._send_session_begin(session)
        self._push_envelope(T_CHAT_LOADING, {"model": False, "tts": False, "stt": False})
        self._push_chat_init(session)
        self._schedule(self._await_chat_ready(self._session_epoch))

    async def _on_rebind_save(self, slot: str, character_id: str) -> None:
        """Re-bind a saved chat to another character (picked in the renderer's rebind
        dialog after the original was deleted), then resume it. The history is kept
        as-is; the new character takes over — its model, voice, emotions and system
        prompt apply from the next turn."""
        if not slot or not character_id:
            return
        character = await self._fetch_character(character_id)
        if character is None:
            self._push_error("character_missing", "That character no longer exists.")
            return
        err = self.save.rebind_slot(slot, character.id, character.character_name)
        if err:
            self._push_error("rebind_failed", err)
            return
        self.push_saves_list()  # the chat now lists under the new character's name
        await self._on_load_save(slot)

    def _on_delete_save(self, slot: str) -> None:
        if not slot:
            return
        self.save.delete_save(slot)
        # Deleting the chat that's currently open ends the live session: unload the character on
        # the Unity side and drop the orchestrator session so nothing stale lingers. (The renderer
        # independently returns to its home page.)
        active = self.orchestrator is not None and self.orchestrator.session is not None
        if slot == self.current_slot and active:
            self._end_active_session()
        self.push_saves_list()

    def _on_end_session(self) -> None:
        """Renderer asked to end the current session (sidebar "End session"). Unload the character
        on the Unity side and drop the orchestrator session, keeping the save intact. The renderer
        returns to its home page on its own."""
        active = self.orchestrator is not None and self.orchestrator.session is not None
        if active:
            self._end_active_session()
        self.push_saves_list()

    def _end_active_session(self) -> None:
        """Tear down the live session: stop speech, tell Unity to unload the character (Session.End),
        and clear the orchestrator session + current slot."""
        if self.speech is not None:
            self.speech.stop_session()
        self._send_envelope(T_SESSION_END, {})
        if self.orchestrator is not None:
            self.orchestrator.end_session()
        self._reset_session_bookkeeping()
        self.current_slot = self.save.default_slot

    # ---- per-chat settings (voice mode/provider + workspace roots) ----------

    def _effective_workspace_roots(self) -> list[str]:
        """The roots in force for the active chat: its per-chat override if set, else the
        global config baseline. (Attachment grants are layered on top separately.)"""
        return list(self._chat_workspace_roots) if self._chat_workspace_roots else list(self._base_workspace_roots)

    def _apply_workspace_roots(self, roots: list[str]) -> None:
        """Set the tool sandbox's allowed roots for the active chat. `roots` REPLACES the
        global config roots; an empty list falls back to the captured global baseline. Any
        live attachment grants (tracked in _attachment_roots) are re-layered on top so they
        survive the reset. No-op until the tool manager exists."""
        tm = getattr(self, "_tool_manager", None)
        ws = getattr(tm, "workspace", None) if tm is not None else None
        if ws is None:
            return
        base = [str(r) for r in (roots or self._base_workspace_roots) if isinstance(r, str) and r]
        ws.allowed_roots = list(base)
        # Re-layer active attachment grants (file tools must still reach attached files).
        existing = {os.path.normcase(os.path.abspath(r)) for r in ws.allowed_roots}
        for folder in self._attachment_roots:
            af = os.path.abspath(folder)
            if os.path.normcase(af) not in existing:
                ws.allowed_roots.append(af)
                existing.add(os.path.normcase(af))

    def _ensure_voice_provider(self, provider: str) -> None:
        """Switch the active TTS engine to this chat's chosen provider. No-op when the chat
        has no override ("") or no switcher is wired; the app-side switcher itself
        skips the (costly) reload when the requested provider is already active."""
        name = (provider or "").strip().lower()
        if not name or self._set_voice_provider_fn is None:
            return
        try:
            self._set_voice_provider_fn(name)
        except Exception as e:
            print(f"[ChatManager] set voice provider failed: {e}", file=sys.stderr)

    def _apply_llm_config(self, config_id: str) -> None:
        """Switch the active chat to the LLM config with this id (or the registry default when the
        id is empty/unknown). Copies the resolved config's fields into self.config IN PLACE — every
        component (self.llm.cfg, the orchestrator, each tool) holds a reference to the same
        ChatBackendConfig / LlmConfig instance, so the next LLM/tool call picks up the new values.
        No-op when no registry is wired (single static config)."""
        registry = getattr(self, "_llm_registry", None)
        if registry is None:
            return
        entry = registry.get(config_id) or registry.default()
        if entry is None:
            return
        src = entry.config
        for attr in (
            "api_url", "api_key", "model", "temperature", "request_timeout_seconds",
            "thinking", "send_system_prompt_as_user", "supports_vision", "vision_max_images",
        ):
            setattr(self.config.llm, attr, getattr(src.llm, attr))
        self.config.max_tool_call_rounds = src.max_tool_call_rounds
        self.config.vision_max_image_edge_pixels = src.vision_max_image_edge_pixels
        self.config.vision_jpeg_quality = src.vision_jpeg_quality
        # Read tools branch on these per call — keep the tool manager's mirror in sync so a vision
        # config switch takes effect on the next image read.
        if self._tool_manager is not None:
            self._tool_manager.supports_vision = bool(self.config.llm.supports_vision)
            self._tool_manager.vision_max_edge_pixels = int(self.config.vision_max_image_edge_pixels)
            self._tool_manager.vision_jpeg_quality = int(self.config.vision_jpeg_quality)

    async def _on_update_settings(self, payload: dict) -> None:
        """Apply edits from the per-chat settings dialog to the live session + persist them.
        Only the keys present in `payload` are touched (user name, voice mode, voice provider,
        workspace roots)."""
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        session = self.orchestrator.session
        payload = payload or {}
        if "userName" in payload:
            un = str(payload.get("userName") or "").strip()
            if un:
                session.user_name = un
        if "voiceMode" in payload:
            self.voice_enabled = bool(payload.get("voiceMode"))
        if "headTracking" in payload:
            self.head_tracking_enabled = bool(payload.get("headTracking"))
        if "eyeTracking" in payload:
            self.eye_tracking_enabled = bool(payload.get("eyeTracking"))
        if "voiceProvider" in payload:
            self._voice_provider = str(payload.get("voiceProvider") or "").strip().lower()
            self._ensure_voice_provider(self._voice_provider)
        if "llmConfigId" in payload:
            self._llm_config_id = str(payload.get("llmConfigId") or "").strip()
            self._apply_llm_config(self._llm_config_id)
        if "workspaceRoots" in payload:
            self._chat_workspace_roots = [str(r) for r in (payload.get("workspaceRoots") or [])
                                          if isinstance(r, str) and r]
            self._apply_workspace_roots(self._chat_workspace_roots)
        # Rebuild the system prompt so the new user_name / workspace cwd show up on the next turn.
        try:
            self.orchestrator._refresh_system_prompt()
        except Exception as e:
            print(f"[ChatManager] update_settings refresh_system_prompt failed: {e}", file=sys.stderr)
        self._persist()
        self._send_session_begin(session)
        self._push_chat_init(session)
        self.push_saves_list()

    def reload_config(self, fresh_chat_cfg: ChatBackendConfig, fresh_app: dict,
                      fresh_registry=None) -> None:
        """Apply edits made through the settings panel to live in-memory state, so
        the next LLM call / next tool call / next system-prompt rebuild picks up
        the new values without a process restart.

        We mutate the existing dataclasses in place rather than replacing them —
        every component (`self.llm.cfg`, `self._tool_manager.workspace`, each
        registered tool) holds a reference to the same instance, so an in-place
        mutation propagates everywhere without rewiring.

        fresh_registry: the re-read LLM config set. We swap it in and re-apply the ACTIVE chat's
        selected config (by id) so an edit to that config — or to the default it follows — takes
        effect, without stomping a non-default chat with the default's values.
        """
        # ----- ChatBackendConfig: user_name -----
        self.config.user_name = fresh_chat_cfg.user_name
        # ----- LLM config set: swap the registry, then re-apply the active chat's config. -----
        if fresh_registry is not None:
            self._llm_registry = fresh_registry
        self._apply_llm_config(self._llm_config_id)

        # ----- Live ChatSession: keep user_name aligned so the next outbound
        #       message / system-prompt rebuild sees the new value. -----
        if self.orchestrator is not None and self.orchestrator.session is not None:
            self.orchestrator.session.user_name = fresh_chat_cfg.user_name

        # ----- Tool manager: vision flags + workspace allowedRoots + shell allow/deny lists -----
        if self._tool_manager is not None:
            # Read tool branches on these per call — flipping supports_vision in
            # the settings panel should take effect on the very next image read.
            self._tool_manager.supports_vision = bool(self.config.llm.supports_vision)
            self._tool_manager.vision_max_edge_pixels = int(self.config.vision_max_image_edge_pixels)
            self._tool_manager.vision_jpeg_quality = int(self.config.vision_jpeg_quality)
            ws_block = fresh_app.get("workspace") if isinstance(fresh_app.get("workspace"), dict) else {}
            ws_cfg = self._tool_manager.workspace
            new_roots = ws_block.get("allowedRoots")
            if isinstance(new_roots, list):
                # The settings panel edits the GLOBAL roots — refresh the baseline, then
                # re-derive the live roots so an active chat's per-chat override still wins.
                self._base_workspace_roots = [str(r) for r in new_roots if isinstance(r, str) and r]
                self._apply_workspace_roots(self._chat_workspace_roots)
            new_allow = ws_block.get("allowedCommandPrefixes")
            if isinstance(new_allow, list):
                ws_cfg.allowed_command_prefixes = [str(s) for s in new_allow if isinstance(s, str) and s]
            new_deny = ws_block.get("deniedCommandPrefixes")
            if isinstance(new_deny, list):
                ws_cfg.denied_command_prefixes = [str(s) for s in new_deny if isinstance(s, str) and s]
            # Full-permission mode: drop the path sandbox + auto-approve every tool call.
            if isinstance(ws_block.get("fullAccess"), bool):
                ws_cfg.full_access = bool(ws_block["fullAccess"])
                self._tool_manager.approval.full_access = ws_cfg.full_access
            # Some integer caps too — leave their defaults if not set.
            for json_key, attr in (
                ("maxReadFileBytes", "max_read_file_bytes"),
                ("maxGlobResults", "max_glob_results"),
                ("maxGrepResults", "max_grep_results"),
                ("defaultExecTimeoutSeconds", "default_exec_timeout_seconds"),
            ):
                v = ws_block.get(json_key)
                if isinstance(v, int):
                    setattr(ws_cfg, attr, v)

            # WebSearch tool holds a private config dataclass; mutate it in place.
            wsearch_tool = self._tool_manager.get("WebSearch")
            wsearch_block = fresh_app.get("webSearch") if isinstance(fresh_app.get("webSearch"), dict) else {}
            if wsearch_tool is not None and hasattr(wsearch_tool, "config"):
                for json_key, attr in (
                    ("resultsPerCall", "results_per_call"),
                    ("hardCapPage", "hard_cap_page"),
                    ("requestTimeoutSeconds", "request_timeout_seconds"),
                    ("maxAttempts", "max_attempts"),
                ):
                    v = wsearch_block.get(json_key)
                    if isinstance(v, int):
                        setattr(wsearch_tool.config, attr, v)
                tb = wsearch_block.get("textBackend")
                if isinstance(tb, str) and tb.strip():
                    wsearch_tool.config.text_backend = tb.strip()

        # ----- Eagerly rebuild the system prompt so the in-memory history's
        #       leading system message reflects the new user_name + workspace
        #       root right now, rather than waiting for the next turn. The
        #       orchestrator would refresh on the next submit_message anyway,
        #       but doing it here means an immediate Chat.Stop+Resume picks up
        #       the new prompt too. -----
        if self.orchestrator is not None and self.orchestrator.session is not None:
            try:
                self.orchestrator._refresh_system_prompt()
            except Exception as e:
                print(f"[ChatManager] reload refresh_system_prompt failed: {e}", file=sys.stderr)

        # ----- Renderer: push the new user_name so any UI that shows it
        #       (current and future chat bubbles) refreshes immediately. -----
        self._push_envelope("Chat.UserNameChanged", {"userName": fresh_chat_cfg.user_name})

    async def _on_restart(self) -> None:
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        self.orchestrator.restart_session()
        # Restart clears the whole conversation, so every report goes away — delete the
        # on-disk bodies too, otherwise the report_{id}.md files are orphaned
        # (invisible to the UI but never cleaned up until the whole chat is deleted).
        for r in self._reports:
            self.save.delete_report(r.id, self.current_slot)
        self._reset_session_bookkeeping()
        self._persist()
        self._send_session_begin(self.orchestrator.session)
        self._push_chat_init(self.orchestrator.session)
        # Restart is a fresh conversation — deliver the greeting as the assistant's first response too,
        # same as a new chat (the model/TTS/STT are already loaded, so no readiness gate needed).
        gmsg = self.orchestrator.make_greeting_message(self.orchestrator.session)
        if gmsg is not None:
            await self._deliver_greeting(gmsg.content)

    def _reset_session_bookkeeping(self) -> None:
        self._reports.clear()
        self._todo_snapshots.clear()
        self._turn_snapshots.clear()
        self._pending_report_ids_this_turn.clear()
        self._pending_todo_snapshot_id_this_turn = None
        self._touch_active = False
        # Background helpers belong to the session they were summoned in — dismiss them
        # (their reports would land in the wrong conversation otherwise).
        self._cancel_all_bg_tasks()
        # Fresh/restarted chat: revoke any attachment workspace grants from the previous session.
        self._revoke_attachment_roots()
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        """Drop the per-round emotion filter + sanitizer and clear the new-entry flag.
        Called between sessions / on rollback / at the start of each user turn."""
        self._text_filter = None
        self._text_sanitizer = None
        self._needs_new_assistant_entry = False

    async def resync_unity_session(self) -> None:
        """Re-sync the live session to Unity after a (re)connect. Unity loses all avatar
        state when it restarts/reconnects, so we re-send Session.Begin — which makes
        CharacterCommandReceiver reload the character model (it guards on modelPath, so an
        already-loaded model is a no-op) and re-apply the current emotion / status.
        No-op when no session is active (e.g. the renderer is still on the home page)."""
        session = self.orchestrator.session if self.orchestrator else None
        if session is None:
            print("[chat] resync: no active session — nothing to push", file=sys.stderr)
            return
        print(f"[chat] resync: re-sending Session.Begin for '{session.character.character_name}' "
              f"(model='{session.character.model_path}')", file=sys.stderr)
        self._send_session_begin(session)

    def _send_session_begin(self, session: ChatSession, include_chat_view: bool = True) -> None:
        """Pushes Session.Begin to Unity so CharacterCommandReceiver applies the initial
        outfit / emotion / status atomically.

        include_chat_view: send the saved Chat-mode camera/window framing for Unity to restore.
        True for paths where Unity has no (or stale) camera state — initial open, resume, reconnect
        resync. False for a rollback (rewind): the camera framing is a live UI preference, not
        per-turn state, and Unity still holds the user's current framing, so re-applying the saved
        view would yank the zoom/pan unexpectedly."""
        # Keep voice mode honest: no voice for the active provider → voice-off.
        self._enforce_voice_availability()
        outfit_name = session.current_outfit.outfit_name if session.current_outfit else ""
        outfit_index = session.current_outfit.index if session.current_outfit else -1
        emotion_label = (session.current_emotions[0].label if session.current_emotions else "Neutral")
        self._send_envelope(T_SESSION_BEGIN, {
            "characterName": session.character.character_name,
            "userName": session.user_name,
            "outfitName": outfit_name,
            # KK outfit id to restore on load (KKOutfitController defaults to 0, so a resumed non-zero
            # outfit needs this to match the saved state). -1 / non-KK → Unity leaves the default.
            "outfitIndex": outfit_index,
            "emotionLabel": emotion_label,
            "status": session.current_status,
            "voiceMode": self.voice_enabled,
            "headTracking": self.head_tracking_enabled,
            "eyeTracking": self.eye_tracking_enabled,
            # App-owned characters carry the .vrm path so Unity loads + binds it on session start.
            "modelPath": session.character.model_path or "",
            # Echoed back in Session.Ready so the chat-loading gate ignores a stale ready from a prior
            # session. Bumped only by the new/resume entry points (see _arm_chat_loading); other callers
            # (rollback / restart / settings / resync) reuse the current value, which is fine.
            "sessionEpoch": self._session_epoch,
            # Saved Chat-mode camera + window framing to restore (empty {} for a new chat, or a rollback
            # where Unity keeps its current framing → Unity keeps the default framing). Opaque — Unity
            # owns the shape.
            "chatView": self._chat_view if include_chat_view else {},
        })

    def _push_chat_init(self, session: ChatSession) -> None:
        """Initial Chat.Init payload — primes the renderer with history. A new chat has no greeting in
        history yet (it's delivered afterwards as the assistant's first response, see
        _deliver_greeting); a resumed chat's saved history already contains it as a normal turn."""
        # A character with no voice for the active provider can't use voice mode — force
        # it off before reporting so the renderer disables the toggle and shows no-voice.
        available = self._enforce_voice_availability()
        # Effective voice provider for this chat: the live active engine when we can read it
        # (so a "" follow-global chat still reports the real provider), else the saved override.
        provider = ""
        if self._voice_provider_getter is not None:
            try:
                provider = self._voice_provider_getter() or ""
            except Exception:
                provider = ""
        provider = provider or self._voice_provider
        # A freshly (re)loaded model is never in touch mode — reset so the header toggle starts off
        # and matches Unity's AibuToucher (which defaults off after a model rebind).
        self.touch_enabled = False
        self._push_envelope(T_CHAT_INIT, {
            "assistantDisplayName": session.display_name,
            "assistantAvatar": self._profile_image_data_url(session.character.id),
            "userName": session.user_name,
            "initialAssistantLine": "",
            "history": [h.to_wire() for h in self._build_history_entries(session)],
            "voiceSupported": self.voice_supported,
            "voiceMode": self.voice_enabled,
            "voiceAvailable": available,
            "voiceProvider": provider,
            "touchMode": self.touch_enabled,
            "headTracking": self.head_tracking_enabled,
            "eyeTracking": self.eye_tracking_enabled,
            "llmConfigId": self._llm_config_id,
            "workspaceRoots": self._effective_workspace_roots(),
            "activeSlot": self.current_slot,
            # Worn outfit (stable KK coordinate id) so the outfit dialog can mark it. -1 for VRM.
            "currentOutfitIndex": (session.current_outfit.index if session.current_outfit else -1),
        })
        # Re-sync the background-helper pill (init resets the renderer's transient state; a
        # reconnect mid-session may still have helpers working).
        self._push_bg_tasks()

    def _push_outfit_changed(self, session: ChatSession) -> None:
        """Tell the renderer the worn outfit changed — user pick, the AI's ChangeOutfit tool, or
        a rollback restoring an older one — so the outfit dialog marks the right one."""
        o = session.current_outfit
        self._push_envelope(T_CHAT_OUTFIT_CHANGED, {
            "outfitIndex": o.index if o else -1,
            "outfitName": o.outfit_name if o else "",
        })

    # ---- chat-loading overlay (model + TTS + STT readiness gate) ------------

    def _arm_chat_loading(self) -> None:
        """Open a fresh readiness gate for a NEW/RESUMED session: bump the session epoch and create
        a new model-ready Event so a late Session.Ready from the previous session can't satisfy this
        one. MUST be called (on the chat loop) immediately before _send_session_begin so the epoch
        we send to Unity matches the one we'll wait on. Only the new/resume entry points call this —
        rollback/restart/settings/resync deliberately don't, so they never pop the overlay."""
        self._session_epoch += 1
        self._model_ready_event = asyncio.Event()

    async def _await_chat_ready(self, epoch: int, greeting: str | None = None) -> None:
        """Block the chat behind the loading overlay until the model (Unity Session.Ready), the TTS
        engine, and the STT engine have all RESOLVED (loaded or terminally errored), then push
        Chat.Ready. Bails out if a newer session armed in the meantime (epoch mismatch), so a fast
        chat switch doesn't clear the new session's overlay. Pushes Chat.Loading status updates as
        each stage flips. After clearing the overlay, delivers the greeting as a fresh spoken response
        when `greeting` is set (new chats only — resumed chats already have it in history)."""
        last = None
        while epoch == self._session_epoch:
            ev = self._model_ready_event
            model = bool(ev is not None and ev.is_set())
            tts = self._tts_ready_getter() if self._tts_ready_getter is not None else True
            stt = self._stt_ready_getter() if self._stt_ready_getter is not None else True
            status = {"model": model, "tts": tts, "stt": stt}
            # Attach first-run model-download progress (if any) so the overlay shows a real
            # bar instead of an indefinite spinner. None when nothing is downloading.
            import download_progress
            dl = download_progress.snapshot()
            if dl is not None:
                status["download"] = dl
            if status != last:
                self._push_envelope(T_CHAT_LOADING, status)
                last = status
            if model and tts and stt:
                self._push_envelope(T_CHAT_READY, {})
                # Everything's loaded — now the avatar can actually speak. Deliver the greeting only if
                # this session is still the current one (a fast switch would have bumped the epoch).
                if greeting and epoch == self._session_epoch:
                    await self._deliver_greeting(greeting)
                return
            await asyncio.sleep(0.1)

    async def _deliver_greeting(self, raw: str) -> None:
        """Deliver the character's opening line as the assistant's first response: append it to
        history (LLM context + save), push its bubble, and drive voice or text lip-sync + emotion the
        same way a real turn does. `raw` is "[Label]\\ngreeting". Because it's appended here (not
        pre-seeded), a resumed chat just replays it from saved history and never re-speaks it."""
        session = self.orchestrator.session if self.orchestrator else None
        if session is None:
            return
        clean = strip_emotion_tags(raw).strip()
        if not clean:
            return
        prefix = _leading_emotion_tags(raw)
        label = ""
        if prefix:
            m = _EMOTION_TAG_REGEX.search(prefix)
            if m:
                label = m.group(0).strip("[] \t\r\n")

        # This IS the assistant's opening turn now — append it to history so the LLM sees it on the
        # first user turn and it's saved, then show the bubble (matching _build_history_entries' shape
        # so a later reload lines up).
        session.history.append(ChatMessage(role="assistant", content=raw))
        hist_index = len(session.history) - 1
        self._push_entry(HistoryEntry(
            role="assistant",
            speaker=session.display_name,
            text=clean,
            can_rollback=False,
            turn_index=-1,
            reports=[], todos=None,
            history_index=hist_index,
        ))
        # Save now that the greeting is part of history — a resume replays it from the save as normal
        # history (no re-speak), and it survives a crash mid-playback.
        self._persist()

        if self.voice_enabled and self.speech is not None:
            # Speak it through the same sentence pipeline + stream bracketing a real turn uses, so the
            # greeting is just another assistant message — including being cleanly interrupted if the
            # user sends a message mid-greeting (see _begin_tts_stream / _end_tts_stream). The pipeline
            # parses the [Label] tag and fires per-sentence emotion synced to playback.
            tts_gen = self._begin_tts_stream()
            self.speech.feed_token(raw)
            self.speech.end_llm_stream()
            try:
                await self.speech.wait_until_done()
            except Exception as e:
                print(f"[ChatManager] greeting wait_until_done raised: {e}", file=sys.stderr)
            finally:
                self._end_tts_stream(tts_gen)
        else:
            # Voice off: drive Unity's text lip-sync so the avatar mouths the greeting, mirroring a
            # normal voice-off turn (Lipsync.EmotionMarker before the text so the face matches).
            self._send_envelope(T_LIPSYNC_TEXT_BEGIN, {})
            if label:
                self._send_envelope(T_LIPSYNC_EMOTION_MARKER, {"label": label})
                self._push_envelope(T_CHAT_EMOTION, {"label": label})
            self._send_envelope(T_LIPSYNC_TEXT_APPEND, {"text": clean})
            self._send_envelope(T_LIPSYNC_TEXT_END, {})

    def _persist(self) -> None:
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        s = self.orchestrator.session
        data = ChatSaveData(
            version=1,
            character_id=s.character.id,
            character_name=s.character.display_name or s.character.character_name,
            outfit_name=(s.current_outfit.outfit_name if s.current_outfit else ""),
            outfit_index=(s.current_outfit.index if s.current_outfit else -1),
            user_name=s.user_name,
            current_status=s.current_status,
            emotion_labels=[e.label for e in s.current_emotions],
            history=list(s.history),
            turn_snapshots=list(self._turn_snapshots),
            reports=list(self._reports),
            todo_snapshots=list(self._todo_snapshots),
            voice_mode=bool(self.voice_enabled),
            head_tracking=bool(self.head_tracking_enabled),
            eye_tracking=bool(self.eye_tracking_enabled),
            voice_provider=self._voice_provider,
            llm_config_id=self._llm_config_id,
            workspace_roots=list(self._chat_workspace_roots),
            extra_workspace_roots=list(self._attachment_roots),
            chat_view=dict(self._chat_view),
        )
        self.save.save(data, self.current_slot)
