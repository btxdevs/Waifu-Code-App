"""Turn flow: submit, stop, rollback, edit message, open report, voice-mode toggle."""
from __future__ import annotations

import asyncio
import base64
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
    HistoryEntry, ReportRef, TodoItemRef, IMAGE_EXTS,
    _tool_activity_label, _leading_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
    _ATTACHMENT_NOTE_REGEX, build_attachment_refs, strip_attachment_notes,
)


class TurnMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    async def _on_player_submit(self, text: str, attachments: list[str] | None = None) -> None:
        text = text or ""
        if self.orchestrator is None or self.orchestrator.session is None:
            self._push_error("no_session", "No active chat session. Create or load one first.")
            return

        # Fold any attachments in: images (when the LLM is vision-capable) become inline image
        # messages; everything else has its path appended to the text + its folder granted to the
        # workspace so tools can reach it. `final_text` is what's shown + sent; `image_messages`
        # are extra content-blocks user rows appended right after the turn message.
        final_text, image_messages, thumb_paths = await self._process_attachments(text, attachments or [])
        if not final_text.strip() and not image_messages:
            return

        # Take a turn snapshot BEFORE we mutate session — needed for rollback.
        s = self.orchestrator.session
        self._turn_snapshots.append(TurnSnapshot(
            user_message=text,  # original typed text (refilled on rollback)
            history_length=len(s.history),
            outfit_name=(s.current_outfit.outfit_name if s.current_outfit else ""),
            outfit_index=(s.current_outfit.index if s.current_outfit else -1),
            current_status=s.current_status,
            emotion_labels=[e.label for e in s.current_emotions],
            attachments=list(attachments or []),  # re-offered in the composer on rollback
            image_thumbs=list(thumb_paths),       # bubble thumbnails (survive reload)
        ))

        # If the player was just caressing the avatar, close that out for the LLM before this typed
        # message — a hidden note (no chat row), so the AI knows the touching has ended. Appended after
        # the snapshot so a rewind of THIS turn removes it too; it precedes the user message in history.
        if self._touch_active:
            s.history.append(ChatMessage(
                role="user",
                content_blocks=[{"type": "text", "text": self._system_message(self._TOUCH_ENDED_NOTE)}]))
            self._touch_active = False

        # Begin speech pipeline if voice mode is on. Bracket the playback with
        # Tts.StreamBegin / Tts.StreamEnd so Unity's StreamingAudioReceiver creates a
        # fresh shared buffer that all sentences feed (gapless) and tears it down on
        # end so PlaybackEnded fires when the audio thread drains.
        tts_gen: int | None = None
        if self.voice_enabled and self.speech is not None:
            # Opens the stream, cleanly interrupting any still-playing response (turn or greeting) first.
            tts_gen = self._begin_tts_stream()
        elif not self.voice_enabled:
            # Voice off: stream the clean reply text to Unity so TextLipSyncController drives
            # the mouth (no TTS audio to analyze). Bracketed by Lipsync.TextEnd in finally.
            self._send_envelope(T_LIPSYNC_TEXT_BEGIN, {})
        self._reset_streaming_state()
        self._needs_new_assistant_entry = True
        self._pending_report_ids_this_turn.clear()
        self._pending_todo_snapshot_id_this_turn = None

        # Push the user's message to the renderer immediately as a fresh history entry,
        # so the user sees their bubble during the LLM round-trip instead of staring
        # at a frozen UI. orchestrator.submit_message will mirror it into
        # session.history when it runs — the user message is the very next thing
        # appended, so its eventual index is the current history length.
        atts = build_attachment_refs(attachments or [], thumb_paths)
        self._push_entry(HistoryEntry(
            role="user",
            speaker=s.user_name,
            # The 📎 note lines stay in final_text for the LLM; the bubble shows chips instead.
            text=strip_attachment_notes(final_text) if atts else final_text,
            can_rollback=True,
            turn_index=len(self._turn_snapshots) - 1,
            reports=[],
            todos=None,
            history_index=len(s.history),
            attachments=atts,
        ))

        self._push_envelope(T_CHAT_TYPING, {"active": True})

        # Run the orchestrator turn.
        self._current_turn_task = asyncio.current_task()
        try:
            reply = await self.orchestrator.submit_message(final_text, image_messages)
        except asyncio.CancelledError:
            # Treat as a clean stop — orchestrator.cancel() also paths here.
            reply = None
        finally:
            self._current_turn_task = None
            self._push_envelope(T_CHAT_TYPING, {"active": False})
            if self.speech is not None:
                self.speech.end_llm_stream()
                # Wait for the speech pump to drain its queue + any in-flight TTS
                # synthesis BEFORE telling Unity the stream is over. The LLM almost
                # always finishes streaming long before all sentences have been TTS'd —
                # sentences are produced faster than they can be synthesized — so
                # sending StreamEnd here without waiting would close Unity's playback
                # buffer mid-utterance and cut the audio off.
                try:
                    await self.speech.wait_until_done()
                except Exception as e:
                    print(f"[ChatManager] wait_until_done raised: {e}", file=sys.stderr)
                # Close the stream we opened (no-op if a newer response superseded it, so our late
                # StreamEnd can't cut their audio).
                if tts_gen is not None:
                    self._end_tts_stream(tts_gen)
            if not self.voice_enabled:
                # Close the text lip-sync stream; Unity lets it drain at charactersPerSecond.
                self._send_envelope(T_LIPSYNC_TEXT_END, {})

        # Persist on disk. We do NOT push history here — the streaming flow has
        # already pushed every entry as it happened (user message, assistant rounds,
        # tool_activity events) via Chat.PushEntry + Chat.AppendToken. A full rebuild
        # would just be redundant churn on the renderer.
        self._persist()
        # Any background helper reports that landed while this turn ran deliver now.
        self._drain_bg_notifications()

    # Maps a KK AibuColliderKind name (sent verbatim by Unity's AibuTouchReporter) to a third-person
    # stage-direction the LLM sees as a system note. Zones not listed here are ignored (no LLM turn).
    _TOUCH_NOTES = {
        "mouth":       "The user gently brushes a fingertip across your lips.",
        "muneL":       "The user is softly caressing your left breast.",
        "muneR":       "The user is softly caressing your right breast.",
        "kokan":       "The user is touching you intimately between your legs.",
        "anal":        "The user is touching you intimately from behind.",
        "siriL":       "The user is caressing your left buttock.",
        "siriR":       "The user is caressing your right buttock.",
        "reac_head":   "The user is gently petting your head.",
    }

    # Short past-tense labels shown in the chat as a compact tool-activity event (the user's action).
    _TOUCH_LABELS = {
        "mouth":       "Touched lips",
        "muneL":       "Caressed left breast",
        "muneR":       "Caressed right breast",
        "kokan":       "Touched between the legs",
        "anal":        "Touched from behind",
        "siriL":       "Caressed left buttock",
        "siriR":       "Caressed right buttock",
        "reac_head":   "Petted head",
    }

    # Invisible one-shot note prepended to the next typed turn after touching, so the LLM knows the
    # physical interaction is over. Not shown in the chat.
    _TOUCH_ENDED_NOTE = "The user has stopped touching you."

    # The assistant emits [Reject] (anywhere in its reply) to refuse a caress. It's hidden from the
    # visible/spoken text like any [..] tag, so we read it off the RAW reply. As a registered control
    # marker (see text.CONTROL_MARKERS) it's also KEPT in stored history on a touch turn — but stripped
    # if emitted on a non-touch turn. Case-insensitive so [Reject]/[reject] both match.
    _TOUCH_REJECT_REGEX = re.compile(r"\[\s*reject\s*\]", re.IGNORECASE)

    def _touch_note(self, zone: str) -> str:
        """Stage-direction text for a touched zone, or "" for a zone we don't surface to the LLM."""
        return self._TOUCH_NOTES.get(zone or "", "")

    def _touch_label(self, zone: str) -> str:
        """Compact chat-row label for a touched zone (falls back to a generic phrasing)."""
        return self._TOUCH_LABELS.get(zone or "", "Touched the character")

    # Out-of-band environment notes are injected as a (hidden) user turn prefixed with this marker so
    # the model can tell them apart from the user's own speech — see the SYSTEM MESSAGES section of
    # system_prompt.txt. Touch uses it now; future interactions can reuse _system_message().
    _SYSTEM_MESSAGE_PREFIX = "[System Message]\n"

    def _system_message(self, body: str) -> str:
        """Wrap an environment note as a system message the model recognizes (and won't read aloud)."""
        return f"{self._SYSTEM_MESSAGE_PREFIX}{body}"

    def _reply_rejects_touch(self, reply_text: str | None) -> bool:
        return bool(reply_text) and bool(self._TOUCH_REJECT_REGEX.search(reply_text))

    async def _on_touch_event(self, zone: str) -> None:
        """The player started caressing the avatar. Inject a hidden stage-direction (a content-blocks user
        row — seen by the LLM) and run it through the normal turn pipeline so the reply streams, speaks, and
        emotes exactly like a typed message. The touch itself shows in the chat as a compact, rewindable
        tool-activity row (no user bubble). If the reply carries the rejection marker, tell Unity to play the
        dislike animation + force-end the touch."""
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        # Block a second touch turn until the first has fully finished (LLM stream + speech). Combined
        # with the _current_turn_task guard (a typed turn in flight), this is what stops tap-spam from
        # piling up touch messages. The physical reaction still plays locally in Unity regardless.
        if self._touch_busy:
            return
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return
        note = self._touch_note(zone)
        if not note:
            return
        self._touch_busy = True
        try:
            s = self.orchestrator.session
            # Snapshot BEFORE injecting so the touch turn is rewindable: a rewind truncates history to
            # here (dropping the hidden touch note + the reply). user_message="" — rewinding a touch is a
            # pure cancel, there's no composer text to restore.
            self._turn_snapshots.append(TurnSnapshot(
                user_message="",
                history_length=len(s.history),
                outfit_name=(s.current_outfit.outfit_name if s.current_outfit else ""),
                outfit_index=(s.current_outfit.index if s.current_outfit else -1),
                current_status=s.current_status,
                emotion_labels=[e.label for e in s.current_emotions],
                attachments=[],
                image_thumbs=[],
            ))
            turn_index = len(self._turn_snapshots) - 1

            # Same playback bracketing as a typed message so voice / text lip-sync / emotion all run normally.
            tts_gen: int | None = None
            if self.voice_enabled and self.speech is not None:
                tts_gen = self._begin_tts_stream()
            elif not self.voice_enabled:
                self._send_envelope(T_LIPSYNC_TEXT_BEGIN, {})
            self._reset_streaming_state()
            self._needs_new_assistant_entry = True
            self._pending_report_ids_this_turn.clear()
            self._pending_todo_snapshot_id_this_turn = None

            # Arm early-rejection detection: _on_orch_token watches the streamed reply and fires
            # Avatar.EndTouch the moment [Reject] appears (it's required at the reply's start), so the
            # dislike plays as the refusal begins rather than after the first sentence is spoken.
            self._touch_turn_zone = zone
            self._touch_reject_sent = False
            self._touch_reply_head = ""

            # Show the touch as a compact, rewindable tool-activity row. The hidden touch note is the very
            # next thing appended to history (by submit_message), so its eventual index is the current length.
            self._push_entry(HistoryEntry(
                role="user_action",
                speaker=s.display_name,
                text=self._touch_label(zone),
                can_rollback=True,
                turn_index=turn_index,
                reports=[], todos=None,
                tool_name="Touch",
                history_index=len(s.history),
            ))
            self._push_envelope(T_CHAT_TYPING, {"active": True})

            self._current_turn_task = asyncio.current_task()
            reply = None
            try:
                # hidden=True + touch_zone: the LLM sees the stage-direction as a content-blocks user row
                # (prefixed [System Message] so it's not mistaken for the user's own speech); the renderer
                # shows the tool-activity row above instead of a bubble.
                reply = await self.orchestrator.submit_message(
                    self._system_message(note), hidden=True, touch_zone=zone)
            except asyncio.CancelledError:
                reply = None
            finally:
                self._current_turn_task = None
                self._push_envelope(T_CHAT_TYPING, {"active": False})
                if self.speech is not None:
                    self.speech.end_llm_stream()
                    try:
                        await self.speech.wait_until_done()
                    except Exception as e:
                        print(f"[ChatManager] touch turn wait_until_done raised: {e}", file=sys.stderr)
                    if tts_gen is not None:
                        self._end_tts_stream(tts_gen)
                if not self.voice_enabled:
                    self._send_envelope(T_LIPSYNC_TEXT_END, {})

            # Rejection fallback: normally _on_orch_token already fired Avatar.EndTouch mid-stream
            # (early detection). This only catches the rare case where that didn't run — e.g. the
            # marker wasn't at the head — so we still force-end the touch once.
            if (reply is not None and not self._touch_reject_sent
                    and self._reply_rejects_touch(reply.reply)):
                self._send_envelope(T_AVATAR_END_TOUCH, {"reason": "reject", "zone": zone})

            # Mark the caress as ongoing so the next typed message gets a "touching ended" note.
            self._touch_active = True
            self._persist()
        finally:
            self._touch_busy = False
            self._touch_turn_zone = None   # disarm early-rejection detection for this turn
        # Any background helper reports that landed while this turn ran deliver now.
        self._drain_bg_notifications()

    @staticmethod
    def _outfit_change_label(outfit_name: str) -> str:
        """Compact chat-row label for a user-initiated outfit change."""
        return f'Changed outfit to "{outfit_name}"'

    async def _on_change_outfit(self, outfit_index: int) -> None:
        """The user picked a new outfit in the app's outfit dialog. Apply it to the avatar
        immediately, then run a hidden stage-direction turn (same pipeline as a touch) so the
        character reacts to being changed. Shows in the chat as a compact, rewindable
        user_action row; rewinding restores the previous outfit (the turn snapshot carries it,
        and _on_rollback re-applies it via Session.Begin)."""
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        # Same interaction guards as a touch: one user-action turn at a time (_touch_busy doubles
        # as that guard), and never while a typed turn is still streaming/speaking.
        if self._touch_busy:
            return
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return
        s = self.orchestrator.session
        target = s.character.get_outfit_by_index(outfit_index) if s.character else None
        if target is None:
            print(f"[ChatManager] change outfit: index {outfit_index} not found", file=sys.stderr)
            return
        if s.current_outfit is not None and s.current_outfit.index == target.index:
            return  # already wearing it
        self._touch_busy = True
        try:
            # Snapshot BEFORE the change so the turn is rewindable: a rewind truncates the hidden
            # note + the reply and restores the PREVIOUS outfit. user_message="" — pure cancel.
            self._turn_snapshots.append(TurnSnapshot(
                user_message="",
                history_length=len(s.history),
                outfit_name=(s.current_outfit.outfit_name if s.current_outfit else ""),
                outfit_index=(s.current_outfit.index if s.current_outfit else -1),
                current_status=s.current_status,
                emotion_labels=[e.label for e in s.current_emotions],
                attachments=[],
                image_thumbs=[],
            ))
            turn_index = len(self._turn_snapshots) - 1

            # Apply the outfit NOW — session state (so this turn's system prompt already describes
            # the new outfit), the Unity avatar, and the renderer's worn-outfit marker. Mirrors
            # what the AI's ChangeOutfit tool path does in _on_orch_executed_tool. The previous
            # name is captured first so the reaction note can say what was worn before (the
            # refreshed system prompt only knows the new state).
            previous_name = s.current_outfit.outfit_name if s.current_outfit else ""
            s.current_outfit = target
            self._send_envelope(T_AVATAR_APPLY_OUTFIT, {
                "outfitName": target.outfit_name,
                "outfitIndex": target.index,
            })
            self._push_outfit_changed(s)

            # Same playback bracketing as a typed message so voice / text lip-sync / emotion all run.
            tts_gen: int | None = None
            if self.voice_enabled and self.speech is not None:
                tts_gen = self._begin_tts_stream()
            elif not self.voice_enabled:
                self._send_envelope(T_LIPSYNC_TEXT_BEGIN, {})
            self._reset_streaming_state()
            self._needs_new_assistant_entry = True
            self._pending_report_ids_this_turn.clear()
            self._pending_todo_snapshot_id_this_turn = None

            # Show the change as a compact, rewindable user_action row. The hidden note is the very
            # next thing appended to history (by submit_message), so its index is the current length.
            self._push_entry(HistoryEntry(
                role="user_action",
                speaker=s.display_name,
                text=self._outfit_change_label(target.outfit_name),
                can_rollback=True,
                turn_index=turn_index,
                reports=[], todos=None,
                tool_name="ChangeOutfit",
                history_index=len(s.history),
            ))
            self._push_envelope(T_CHAT_TYPING, {"active": True})

            self._current_turn_task = asyncio.current_task()
            try:
                # Same from/to shape as the ChangeOutfit tool's result, so the model knows what
                # it was wearing before (the system prompt only describes the new state).
                if previous_name:
                    note = (f'The user has changed your outfit — you were wearing "{previous_name}" '
                            f'and are now wearing "{target.outfit_name}". React to it.')
                else:
                    note = (f'The user has changed your outfit — you are now wearing '
                            f'"{target.outfit_name}". React to it.')
                await self.orchestrator.submit_message(
                    self._system_message(note), hidden=True,
                    outfit_change=target.outfit_name)
            except asyncio.CancelledError:
                pass
            finally:
                self._current_turn_task = None
                self._push_envelope(T_CHAT_TYPING, {"active": False})
                if self.speech is not None:
                    self.speech.end_llm_stream()
                    try:
                        await self.speech.wait_until_done()
                    except Exception as e:
                        print(f"[ChatManager] outfit turn wait_until_done raised: {e}", file=sys.stderr)
                    if tts_gen is not None:
                        self._end_tts_stream(tts_gen)
                if not self.voice_enabled:
                    self._send_envelope(T_LIPSYNC_TEXT_END, {})

            self._persist()
        finally:
            self._touch_busy = False
        # Any background helper reports that landed while this turn ran deliver now.
        self._drain_bg_notifications()

    async def _process_attachments(self, text: str, attachments: list[str]) -> tuple[str, list[ChatMessage]]:
        """Fold attachments into the turn. Returns (final_text, image_messages):
          * image + vision-capable LLM → a separate content-blocks user message (seen inline by
            the LLM, hidden from the chat history view), plus a short display note in the text.
          * anything else (or images when vision is off) → the absolute path is appended to the
            text so the AI knows where it is, and the file's folder is granted to the workspace
            so file tools can reach it even outside the configured work folder.
        """
        if not attachments:
            return text.strip(), [], []
        notes: list[str] = []
        image_messages: list[ChatMessage] = []
        thumb_paths: list[str] = []  # abs paths the bubble previews (saved copy, else original)
        vision = bool(self.config.llm.supports_vision)
        for raw in attachments:
            path = os.path.abspath(raw)
            name = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTS:
                block, saved_path = await self._build_image_block(path)
                if block is not None and vision:
                    image_messages.append(ChatMessage(
                        role="user",
                        content_blocks=[
                            {"type": "text", "text": f"User attached image: {name}"},
                            block,
                        ],
                        image_source="user",
                    ))
                    notes.append(f"\U0001F4CE Attached image: {name}")
                    # Prefer the persisted, preprocessed copy for the thumbnail so it survives
                    # even if the original file later moves; fall back to the original path.
                    thumb_paths.append(saved_path or path)
                    continue
                # Vision off (or processing failed): hand over the path + grant tool access. We
                # still show a thumbnail — the saved copy if we managed to make one, else original.
                notes.append(f"\U0001F4CE Attached image: {name}")
                self._grant_workspace_access(path)
                thumb_paths.append(saved_path or path)
                continue
            notes.append(f"\U0001F4CE Attached file: {path}")
            self._grant_workspace_access(path)
        final_text = text.strip()
        if notes:
            joined = "\n".join(notes)
            final_text = (final_text + "\n\n" + joined).strip() if final_text else joined
        return final_text, image_messages, thumb_paths

    async def _build_image_block(self, path: str) -> tuple[dict | None, str]:
        """Read + downscale an image to a JPEG (white background composited for transparency, by
        the image processor), persist that JPEG as a side file for the bubble thumbnail, and
        return (image_url block, saved_abs_path). Returns (None, "") on any failure."""
        if self._image_processor_fn is None:
            return None, ""
        try:
            raw = await asyncio.to_thread(lambda: open(path, "rb").read())
        except OSError as e:
            print(f"[chat] attachment read failed '{path}': {e}", file=sys.stderr)
            return None, ""
        try:
            jpeg, _w, _h, err = await asyncio.to_thread(
                self._image_processor_fn, raw,
                int(self.config.vision_max_image_edge_pixels),
                int(self.config.vision_jpeg_quality),
            )
        except Exception as e:
            print(f"[chat] attachment image processing failed '{path}': {e}", file=sys.stderr)
            return None, ""
        if err or not jpeg:
            return None, ""
        # Persist the preprocessed JPEG now (same hash-naming as save externalization, so it
        # coincides with the eventual saved-image file rather than duplicating it).
        saved_path = self.save.store_image(jpeg, self.current_slot)
        b64 = base64.b64encode(jpeg).decode("ascii")
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}, saved_path

    def _grant_workspace_access(self, path: str) -> None:
        """Grant the attachment's containing folder to the workspace so file tools can reach it
        even outside the configured work folder. Tracked in self._attachment_roots and persisted
        per slot (see _persist / _restore_attachment_roots) so access survives a reload."""
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder:
            self._grant_root(os.path.abspath(folder))

    def _grant_root(self, folder: str) -> None:
        """Add `folder` to the live workspace allowed_roots if it isn't already covered, and
        remember it in self._attachment_roots (persisted + reapplied on load). Only folders we
        actually add are tracked, so _revoke_attachment_roots never strips a configured root."""
        if not folder:
            return
        tm = getattr(self, "_tool_manager", None)
        ws = getattr(tm, "workspace", None) if tm is not None else None
        if ws is None:
            return
        folder = os.path.abspath(folder)
        existing = {os.path.normcase(os.path.abspath(r)) for r in (ws.allowed_roots or [])}
        if os.path.normcase(folder) in existing:
            return  # already reachable (e.g. inside the configured work folder) — nothing to add
        ws.allowed_roots.append(folder)
        if folder not in self._attachment_roots:
            self._attachment_roots.append(folder)
        print(f"[chat] granted workspace access to attachment folder: {folder}", file=sys.stderr)

    def _revoke_attachment_roots(self) -> None:
        """Drop previously-granted attachment folders from the live workspace allowed_roots so
        switching/restarting a chat doesn't leak access, then clear the tracking list."""
        tm = getattr(self, "_tool_manager", None)
        ws = getattr(tm, "workspace", None) if tm is not None else None
        if ws is not None and self._attachment_roots:
            revoke = {os.path.normcase(os.path.abspath(r)) for r in self._attachment_roots}
            ws.allowed_roots = [r for r in (ws.allowed_roots or [])
                                if os.path.normcase(os.path.abspath(r)) not in revoke]
        self._attachment_roots = []

    def _on_stop(self) -> None:
        # Cancel the LLM stream first so no new tokens / sentences land while we're
        # tearing the pipeline down.
        if self.orchestrator is not None:
            self.orchestrator.cancel()
        # Stop the speech pipeline pump + drop any queued sentences.
        if self.speech is not None:
            self.speech.stop_session()
        # Abort whatever the TTS worker is currently synthesizing. Without this the
        # worker thread keeps emitting Tts.AudioChunk envelopes for the in-flight
        # sentence even after Cancel — Unity drops them once the buffer is gone, but
        # the wire chatter is wasted and any sentence that was already partially
        # buffered would still be heard.
        if self._tts_cancel_fn is not None:
            try:
                self._tts_cancel_fn("user stop")
            except Exception as e:
                print(f"[ChatManager] tts_cancel failed: {e}", file=sys.stderr)
        # Tell Unity to cut the AudioSource right now. StreamEnd waits for the
        # buffer to drain (preserves the last sentence's tail), which is the OPPOSITE
        # of what a user-initiated Stop should do — Cancel discards the DSP queue and
        # fires PlaybackEnded immediately.
        self._send_envelope("Tts.Cancel", {"reason": "user stop"})
        # The stream is now cut — mark it closed so the cancelled turn's _end_tts_stream skips its
        # (now redundant) Tts.StreamEnd.
        self._tts_stream_open = False
        if self._current_turn_task is not None and not self._current_turn_task.done():
            self._current_turn_task.cancel()

    async def _on_rollback(self, turn_index: int) -> None:
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        if turn_index < 0 or turn_index >= len(self._turn_snapshots):
            return
        snap = self._turn_snapshots[turn_index]
        s = self.orchestrator.session
        # Rewinding drops any in-progress touch context, so don't emit a stale "touching ended" note later.
        self._touch_active = False
        # Truncate history to the snapshot point.
        del s.history[snap.history_length:]
        # Restore the avatar-affecting fields.
        # Prefer the stable index (names are user-editable), then name, then default.
        new_outfit = (s.character.get_outfit_by_index(snap.outfit_index)
                      or s.character.get_outfit(snap.outfit_name)
                      or s.character.get_default_outfit())
        s.current_outfit = new_outfit
        s.current_status = snap.current_status
        s.current_emotions = [EmotionEntry(label=lbl) for lbl in snap.emotion_labels] or [EmotionEntry("Neutral")]
        # Drop any turn snapshots / pending widgets that pointed past the rollback point.
        del self._turn_snapshots[turn_index:]
        # Reports/todos attached to the truncated turns are gone. Delete the dropped reports'
        # on-disk bodies too — otherwise the report_{id}.md files are orphaned
        # (invisible to the UI but never cleaned up until the whole chat is deleted).
        dropped_reports = [r for r in self._reports if r.history_index >= snap.history_length]
        self._reports = [r for r in self._reports if r.history_index < snap.history_length]
        self._todo_snapshots = [t for t in self._todo_snapshots if t.history_index < snap.history_length]
        for r in dropped_reports:
            self.save.delete_report(r.id, self.current_slot)
        self._persist()
        # Don't restore the saved camera framing on a rewind — it's a live UI preference, not turn
        # state, and Unity still holds the user's current framing. Re-applying it here yanked the zoom.
        self._send_session_begin(s, include_chat_view=False)
        # The rewind may have restored an older outfit — update the renderer's worn marker.
        self._push_outfit_changed(s)
        # Re-emit history; the renderer drops the input field's text back to the rolled-back
        # message and re-populates its attachments so the user can re-send them.
        self._push_history(s)
        self._push_envelope(T_CHAT_PLAYER_INPUT, {
            "text": snap.user_message,
            "attachments": list(snap.attachments),
        })
        # A helper report queued mid-rollback (or parked while the rolled-back turn ran)
        # can deliver now that nothing is in flight.
        self._drain_bg_notifications()

    def _on_edit_message(self, history_index: int, text: str,
                         removed_attachments: list[int] | None = None,
                         turn_index: int = -1) -> None:
        """Edit the content of an existing user/assistant message in place. The
        renderer optimistically updates its own bubble and sends us the new text +
        the source history index; we mutate session.history and persist. No echo
        back — the renderer already shows the edited text.

        Assistant messages keep their leading [LABEL] emotion tag(s) (those drive
        the avatar and are stripped from the visible text), so we re-prepend the
        original tag prefix to the edited prose. User messages keep their trailing
        📎 attachment note lines (shown as chips, not text, so the edit box never
        contained them) — re-appended so the LLM doesn't lose the attachments.

        `removed_attachments` (user rows only) = indexes into the turn's snapshot
        attachment list the user unattached during the edit: their note lines are
        dropped, the snapshot stops re-offering them on rollback, and a removed
        IMAGE's hidden content-blocks row is deleted from history so the LLM stops
        seeing it. Deleting a row shifts indices, so this path fixes up all the
        index-keyed bookkeeping and re-pushes the full history to the renderer
        (the one edit case that DOES echo back). Workspace folder grants are left
        alone — another attachment may share the folder, and grants are per-chat.
        """
        if self.orchestrator is None or self.orchestrator.session is None:
            return
        s = self.orchestrator.session
        if history_index < 0 or history_index >= len(s.history):
            print(f"[ChatManager] edit: history index {history_index} out of range "
                  f"(len={len(s.history)})", file=sys.stderr)
            return
        m = s.history[history_index]
        if m is None or m.role not in ("user", "assistant"):
            print(f"[ChatManager] edit: refusing to edit a {m.role if m else 'None'} row",
                  file=sys.stderr)
            return
        # Image-attachment user rows have no editable plain content — skip.
        if m.content_blocks and not m.content:
            return

        new_text = text or ""
        if m.role == "assistant":
            prefix = _leading_emotion_tags(m.content or "")
            m.content = prefix + new_text
            self._persist()
            return

        notes = _ATTACHMENT_NOTE_REGEX.findall(m.content or "")
        removed_rows = False
        removed = sorted({int(i) for i in (removed_attachments or []) if int(i) >= 0})
        if removed and 0 <= turn_index < len(self._turn_snapshots):
            snap = self._turn_snapshots[turn_index]
            removed = [i for i in removed if i < len(snap.attachments)]
            notes = self._strip_removed_notes(notes, snap.attachments, removed)
            removed_rows = self._remove_attachment_rows(s, history_index, snap, removed)
            # Snapshot: stop re-offering the removed attachments on rollback, and drop their
            # thumbs (one image_thumbs entry per IMAGE attachment, in attachment order).
            removed_set = set(removed)
            image_idxs = [i for i, p in enumerate(snap.attachments)
                          if os.path.splitext(p)[1].lower() in IMAGE_EXTS]
            removed_thumb_ords = {ord_ for ord_, i in enumerate(image_idxs) if i in removed_set}
            snap.image_thumbs = [t for k, t in enumerate(snap.image_thumbs)
                                 if k not in removed_thumb_ords]
            snap.attachments = [p for k, p in enumerate(snap.attachments)
                                if k not in removed_set]

        # Legacy rows (no snapshot → no chips) keep their notes in the DISPLAYED text, so the
        # edited text may already contain them — don't re-append those (would duplicate).
        notes = [n for n in notes if n not in new_text]
        m.content = (new_text + "\n\n" + "\n".join(notes)).strip() if notes else new_text
        self._persist()
        # Deleting image rows shifted history indices — rebuild the renderer's view so its
        # per-entry historyIndex / attachment chips stay in sync.
        if removed_rows:
            self._push_history(s)

    @staticmethod
    def _strip_removed_notes(notes: list[str], attachments: list[str],
                             removed: list[int]) -> list[str]:
        """Drop the 📎 note lines of the removed attachments. Notes were appended one per
        attachment in order, so when the counts line up we drop by position; otherwise
        (legacy row / user typed a 📎 line themselves) fall back to matching each removed
        attachment's name/path against a note's tail, dropping at most one note each."""
        if len(notes) == len(attachments):
            removed_set = set(removed)
            return [n for i, n in enumerate(notes) if i not in removed_set]
        kept = list(notes)
        for i in removed:
            tail_full = attachments[i]
            tail_name = os.path.basename(tail_full) or tail_full
            hit = next((k for k, n in enumerate(kept)
                        if n.endswith(tail_full) or n.endswith(tail_name)), -1)
            if hit >= 0:
                kept.pop(hit)
        return kept

    def _remove_attachment_rows(self, s, history_index: int, snap: TurnSnapshot,
                                removed: list[int]) -> bool:
        """Delete the hidden content-blocks image rows of the removed IMAGE attachments.
        They sit in a run right after the turn's user message (submit_message appends them
        there), each leading with a "User attached image: <name>" text block. Deletion
        shifts every later history index, so the index-keyed bookkeeping (turn snapshots'
        history_length, report/todo anchors) is decremented to match. Returns whether any
        row was deleted."""
        run_end = history_index + 1
        while run_end < len(s.history):
            r = s.history[run_end]
            if r is None or r.role != "user" or not r.content_blocks or r.content:
                break
            run_end += 1

        to_delete: list[int] = []
        for i in removed:
            path = snap.attachments[i]
            if os.path.splitext(path)[1].lower() not in IMAGE_EXTS:
                continue
            marker = f"User attached image: {os.path.basename(path)}"
            for j in range(history_index + 1, run_end):
                if j in to_delete:
                    continue
                first = (s.history[j].content_blocks or [None])[0]
                if (isinstance(first, dict) and first.get("type") == "text"
                        and first.get("text") == marker):
                    to_delete.append(j)
                    break

        for j in sorted(to_delete, reverse=True):
            del s.history[j]
            for snp in self._turn_snapshots:
                if snp.history_length > j:
                    snp.history_length -= 1
            for r in self._reports:
                if r.history_index > j:
                    r.history_index -= 1
            for t in self._todo_snapshots:
                if t.history_index > j:
                    t.history_index -= 1
        return bool(to_delete)

    def _on_open_report(self, report_id: str) -> None:
        if not report_id:
            return
        body = self.save.load_report(report_id, self.current_slot) or "(report file is missing)"
        # Find the title from our local report list.
        title = next((r.title for r in self._reports if r.id == report_id), "Report")
        # Spawn the report modal locally via the app's task-window machinery.
        # We deliberately do NOT send this over the WS to Unity — nothing there
        # handles ShowReport anymore (the report body lives on the Python side now).
        env = {
            "id": "m_" + uuid.uuid4().hex,
            "type": "ShowReport",
            "payload": {"title": title, "markdown": body},
        }
        try:
            self._open_modal_fn(env)
        except Exception as e:
            print(f"[ChatManager] open_modal failed: {e}", file=sys.stderr)

    def _on_set_voice_mode(self, enabled: bool) -> None:
        available = self.voice_available()
        # Can't enable voice mode without a voice for the active provider — clamp to off.
        if enabled and not available:
            enabled = False
        self.voice_enabled = enabled
        if not enabled and self.speech is not None:
            self.speech.stop_session()
        # Voice mode is a per-chat setting — persist the header toggle so it survives a
        # reload/switch instead of reverting to the value last written for this chat.
        self._persist()
        # Tell Unity to switch which lip-sync path is live (LipSyncDirector handles it).
        self._send_envelope(T_CHAT_SET_VOICE_MODE, {"enabled": enabled})
        self._push_envelope(T_CHAT_VOICE_MODE_CHANGED, {"enabled": enabled, "available": available})

    def _on_set_head_tracking(self, enabled: bool) -> None:
        """Sidebar head-tracking toggle — persisted per chat, applied in Unity, echoed to the renderer.
        Mirrors _on_set_voice_mode (Python-authoritative; Unity is a pure consumer)."""
        self.head_tracking_enabled = enabled
        self._persist()
        self._send_envelope(T_CHAT_SET_HEAD_TRACKING, {"enabled": enabled})
        self._push_envelope(T_CHAT_HEAD_TRACKING_CHANGED, {"enabled": enabled})

    def _on_set_eye_tracking(self, enabled: bool) -> None:
        """Sidebar eye-tracking toggle — persisted per chat, applied in Unity, echoed to the renderer."""
        self.eye_tracking_enabled = enabled
        self._persist()
        self._send_envelope(T_CHAT_SET_EYE_TRACKING, {"enabled": enabled})
        self._push_envelope(T_CHAT_EYE_TRACKING_CHANGED, {"enabled": enabled})

    def _on_set_touch_mode(self, enabled: bool) -> None:
        """Header touch-mode toggle. Command Unity (AibuToucher.SetTouchMode) and optimistically
        echo to the renderer so the button responds immediately. Unity confirms (or corrects, e.g.
        if it clamps the value) via Touch.ModeChanged, handled in handle_unity_envelope."""
        self.touch_enabled = enabled
        self._send_envelope(T_CHAT_SET_TOUCH_MODE, {"enabled": enabled})
        self._push_envelope(T_CHAT_TOUCH_MODE_CHANGED, {"enabled": enabled})
