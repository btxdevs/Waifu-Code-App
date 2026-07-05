"""Envelope-type constants (mirror AppProtocol.cs) + app-injected callable types."""
from __future__ import annotations

from typing import Awaitable, Callable


# ---------- envelope type constants (mirror AppProtocol.cs) ----------

# To renderer
T_CHAT_INIT = "Chat.Init"
T_CHAT_HISTORY = "Chat.History"
T_CHAT_PUSH_ENTRY = "Chat.PushEntry"
T_CHAT_APPEND_TOKEN = "Chat.AppendToken"
T_CHAT_TYPING = "Chat.Typing"
T_CHAT_TOOL_ACTIVITY = "Chat.ToolActivity"
T_CHAT_EMOTION = "Chat.Emotion"
T_CHAT_PLAYER_INPUT = "Chat.PlayerInput"
T_CHAT_SPEAKING = "Chat.Speaking"
T_CHAT_ERROR = "Chat.Error"
T_CHAT_SAVES_LIST = "Chat.SavesList"
T_CHAT_CHARACTERS_LIST = "Chat.CharactersList"
T_CHAT_VOICE_MODE_CHANGED = "Chat.VoiceModeChanged"
# Touch mode (the avatar can be caressed). Renderer toggle → Python → Unity (Chat.SetTouchMode);
# Python → renderer state push (Chat.TouchModeChanged). Unity reports its own changes (keyboard
# toggle, AI-reject auto-off) back as Touch.ModeChanged (below) — Chat.* from Unity is dropped.
T_CHAT_TOUCH_MODE_CHANGED = "Chat.TouchModeChanged"
# Head/eye look-at tracking (avatar follows the camera). Per-chat persisted. Renderer toggle →
# Python → Unity (Chat.Set*), Python → renderer state push (Chat.*Changed). Unity is a pure consumer.
T_CHAT_HEAD_TRACKING_CHANGED = "Chat.HeadTrackingChanged"
T_CHAT_EYE_TRACKING_CHANGED = "Chat.EyeTrackingChanged"
# Chat-loading overlay: pushed when a NEW/RESUMED session begins (so the renderer blocks the chat
# behind a loading screen), then Chat.Ready once the model + TTS + STT have all resolved.
T_CHAT_LOADING = "Chat.Loading"
T_CHAT_READY = "Chat.Ready"
# Unity → Python: per-chat Chat-mode camera + window framing changed (opaque dict). Persisted per chat
# and echoed back in Session.Begin.chatView to restore the view on resume.
T_CHAT_VIEW_STATE = "Chat.ViewState"

# From renderer
T_CHAT_SUBMIT = "Chat.SubmitUserMessage"
T_CHAT_STOP = "Chat.Stop"
T_CHAT_RESTART = "Chat.Restart"
T_CHAT_ROLLBACK = "Chat.RollbackToTurn"
T_CHAT_EDIT_MESSAGE = "Chat.EditMessage"
T_CHAT_OPEN_REPORT = "Chat.OpenReport"
T_CHAT_SET_VOICE_MODE = "Chat.SetVoiceMode"
T_CHAT_SET_TOUCH_MODE = "Chat.SetTouchMode"
# User picked a new outfit in the app's outfit dialog (payload: outfitIndex). Applied to the avatar
# immediately + folded into a hidden LLM turn so the character reacts (a rewindable user_action row).
T_CHAT_CHANGE_OUTFIT = "Chat.ChangeOutfit"
# Python → renderer: the current outfit changed — user pick, the AI's ChangeOutfit tool, or a
# rollback restoring an older one (payload: outfitIndex, outfitName).
T_CHAT_OUTFIT_CHANGED = "Chat.OutfitChanged"
# Python → renderer: the background task roster changed (payload: tasks[{id,kind,label,status,
# startedAt,announced}]). Drives the status-bar chips; pushed on start / finish / announce /
# dismiss / session swap.
T_CHAT_BG_TASKS = "Chat.BgTasks"
# Renderer → Python: the user clicked the ✕ on a status-bar task chip (payload: taskId).
T_CHAT_DISMISS_BG_TASK = "Chat.DismissBgTask"
T_CHAT_SET_HEAD_TRACKING = "Chat.SetHeadTracking"
T_CHAT_SET_EYE_TRACKING = "Chat.SetEyeTracking"
T_CHAT_LIST_SAVES = "Chat.ListSaves"
T_CHAT_GET_CHARACTERS = "Chat.GetCharacters"
T_CHAT_LOAD_SAVE = "Chat.LoadSave"
T_CHAT_CREATE_NEW = "Chat.CreateNew"
T_CHAT_DELETE_SAVE = "Chat.DeleteSave"
# End the live session: unload the character in Unity (Session.End) and drop the orchestrator
# session, without deleting any save. The renderer independently returns to its home page.
T_CHAT_END_SESSION = "Chat.EndSession"
# Edit the active chat's per-chat settings (user name, voice mode, voice provider, workspace roots).
T_CHAT_UPDATE_SETTINGS = "Chat.UpdateSettings"
# Character creation page (renderer → Python)
T_CHAT_SAVE_CHARACTER = "Chat.SaveCharacter"             # persist (create or edit) a character
T_CHAT_DELETE_CHARACTER = "Chat.DeleteCharacter"         # remove a character from the app store
T_CHAT_IMPORT_CHARACTER = "Chat.ImportCharacter"         # import a .wcc character bundle (payload: path)
T_CHAT_CHARACTER_MISSING = "Chat.CharacterMissing"       # Python → renderer: resumed chat's character is gone → rebind dialog
T_CHAT_REBIND_SAVE = "Chat.RebindSave"                   # re-bind a saved chat to another character (payload: slot, characterId)
T_CHAT_INSPECT_MODEL_EMOTIONS = "Chat.InspectModelEmotions"  # ask which emotions a picked .vrm supports
T_CHAT_MODEL_EMOTIONS = "Chat.ModelEmotions"             # Python → renderer reply with the inspected emotions
T_CHAT_INSPECT_MODEL_COORDINATES = "Chat.InspectModelCoordinates"  # ask for a KK model's outfit/coordinate list
T_CHAT_MODEL_COORDINATES = "Chat.ModelCoordinates"       # Python → renderer reply with the coordinate list

# To Unity (avatar side)
T_SESSION_BEGIN = "Session.Begin"
T_SESSION_END = "Session.End"
T_AVATAR_APPLY_EMOTION = "Avatar.ApplyEmotion"
T_AVATAR_APPLY_OUTFIT = "Avatar.ApplyOutfit"
T_AVATAR_SET_STATUS = "Avatar.SetStatus"
# Reject an in-progress caress: Unity plays the matching dislike body animation and force-ends touch mode.
# Sent only when the assistant's reply carried the rejection marker. Python → Unity.
T_AVATAR_END_TOUCH = "Avatar.EndTouch"

# From Unity: the player started caressing the avatar on a KK Aibu zone. Folded into an invisible LLM turn.
T_TOUCH_EVENT = "Touch.Event"
# Unity → Python: touch mode actually changed on the Unity side (applied a Chat.SetTouchMode, the
# keyboard toggle, or auto-off after an AI rejection). Python mirrors it to the renderer as
# Chat.TouchModeChanged. A non-Chat.* type because inbound Chat.* from Unity is dropped (app.py).
T_TOUCH_MODE_CHANGED = "Touch.ModeChanged"
# Per-sentence emotion synced to TTS playback. Sent right before a sentence's audio chunks so Unity's
# StreamingAudioReceiver schedules it against the playback clock (applies it when that sentence's audio
# reaches the speakers) instead of the avatar racing through emotions as fast as Python synthesizes.
T_TTS_EMOTION_MARKER = "Tts.EmotionMarker"

# Text lip-sync stream (voice OFF). Counterpart of Tts.StreamBegin/End — forwards the clean
# reply text to Unity so TextLipSyncController can animate the mouth without TTS audio.
T_LIPSYNC_TEXT_BEGIN = "Lipsync.TextBegin"
T_LIPSYNC_TEXT_APPEND = "Lipsync.TextAppend"
T_LIPSYNC_TEXT_END = "Lipsync.TextEnd"
# Voice-OFF per-sentence emotion, paced to the text mouth's char rate (Tts.EmotionMarker analog).
T_LIPSYNC_EMOTION_MARKER = "Lipsync.EmotionMarker"

# RPC (to Unity)
T_CHARACTER_INSPECT_MODEL_EMOTIONS = "Character.InspectModelEmotions"
T_CHARACTER_INSPECT_MODEL_EMOTIONS_RESULT = "Character.InspectModelEmotionsResult"
# Capture a screenshot of the avatar's current on-screen view (for the LookAtYourself tool).
T_CHARACTER_CAPTURE_VIEW = "Character.CaptureView"
T_CHARACTER_CAPTURE_VIEW_RESULT = "Character.CaptureViewResult"

# From Unity (responses we don't request directly)
T_TTS_PLAYBACK_STARTED = "Tts.PlaybackStarted"
T_TTS_PLAYBACK_ENDED = "Tts.PlaybackEnded"
# Unity → Python: the Session.Begin work (runtime model load + bind) has settled. Gates the
# chat-loading overlay's model stage. Carries the echoed sessionEpoch so we can ignore a stale
# ready from a previous/overlapping session.
T_SESSION_READY = "Session.Ready"


# ---------- app-injected callables ----------

SendToUnityFn = Callable[[dict], None]                       # send envelope over WS to Unity
PushToRendererFn = Callable[[dict], None]                     # push envelope into the chat window
# request_from_unity: send an envelope of this type with this payload to Unity and return the
# request id we generated. The actual reply arrives over WS later and is delivered through
# `handle_unity_envelope`, which resolves the bucketed future.
SendUnityRequestFn = Callable[[str, dict], str]
# tts_synthesize: speak this text and resolve when the audio has been pushed to Unity. The
# default app wiring routes through tts.TtsController.synthesize_text under
# asyncio.to_thread; pass a no-op coroutine for headless tests.
TtsSynthesizeFn = Callable[[str], Awaitable[None]]
# tts_cancel: abort whatever the TTS worker is currently synthesizing. Routed by
# app.py to TtsController.cancel_active. Returns True if something was actually
# cancelled. Safe to call when nothing is in flight.
TtsCancelFn = Callable[[str], bool]
# open_modal: hand an envelope (e.g. {type:"ShowReport", payload:{title, markdown}}) to the
# app's task-window spawner. ChatManager uses this for report viewing — the report body
# lives on disk under CompanionApp/saves/, ChatManager loads it and asks the app to pop
# the modal directly without round-tripping through Unity (no one over there handles ShowReport).
OpenModalFn = Callable[[dict], None]
# schedule: fire-and-forget a coroutine on the chat event loop. Used by handle_renderer_chat /
# handle_unity_envelope (which run on sync threads — WS reader, pywebview JS bridge) to bridge
# into async ChatManager methods. app.py provides this via run_coroutine_threadsafe.
ScheduleFn = Callable[[Awaitable], None]
# Readiness getters for the chat-loading overlay. Each returns True once its subsystem's load has
# RESOLVED (loaded OR terminally errored — so the overlay never blocks forever). Wired by
# app.py to TtsController.is_ready / VoiceController.ready_or_errored. None → assume ready.
ReadinessGetterFn = Callable[[], bool]

