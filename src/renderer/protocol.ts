// Wire-format types shared with Unity. Mirrors Assets/Scripts/App/AppProtocol.cs —
// when one side changes, the other has to follow.

export const PROTOCOL_VERSION = 1;

// Unity → Electron
export const TYPE_HELLO = 'Hello';
export const TYPE_SHOW_REPORT = 'ShowReport';
export const TYPE_ASK_QUESTION = 'AskQuestion';
export const TYPE_REQUEST_PERMISSION = 'RequestPermission';
export const TYPE_DISMISS_MODAL = 'DismissModal';
export const TYPE_OPEN_CHAT_WINDOW = 'OpenChatWindow';

// Chat — Unity → Electron
export const TYPE_CHAT_INIT = 'Chat.Init';
/** Chat-loading overlay: pushed by the ChatManager when a new/resumed session begins (overlay up)
 *  and again as each subsystem flips. Payload: { model, tts, stt } booleans. */
export const TYPE_CHAT_LOADING = 'Chat.Loading';
/** Chat-loading overlay cleared — the model + TTS + STT have all resolved. Payload: {}. */
export const TYPE_CHAT_READY = 'Chat.Ready';
/** Full history replace. Used on session start/resume/rollback — anywhere the
 *  whole conversation needs to swap out atomically. Per-token / per-event updates
 *  go through Chat.AppendToken and Chat.PushEntry. */
export const TYPE_CHAT_HISTORY = 'Chat.History';
/** Append one new entry (user / assistant / tool_activity) to the end of history.
 *  Used by the streaming flow: when a new round starts the manager pushes an
 *  empty-but-not-quite assistant entry, then per-token deltas flow via
 *  Chat.AppendToken. */
export const TYPE_CHAT_PUSH_ENTRY = 'Chat.PushEntry';
/** Append a text delta to the LAST entry in history (always an assistant entry
 *  in practice). The streaming model — replaces the old Chat.UpdateBubble
 *  "live bubble" concept. */
export const TYPE_CHAT_APPEND_TOKEN = 'Chat.AppendToken';
export const TYPE_CHAT_TYPING = 'Chat.Typing';
export const TYPE_CHAT_TOOL_ACTIVITY = 'Chat.ToolActivity';
export const TYPE_CHAT_EMOTION = 'Chat.Emotion';
export const TYPE_CHAT_PLAYER_INPUT = 'Chat.PlayerInput';
/** Pushed by the ChatManager when settings change the user_name live. Carries
 *  the new value; the renderer updates state.userName so future bubbles use it. */
export const TYPE_CHAT_USER_NAME_CHANGED = 'Chat.UserNameChanged';

// Chat — Electron → Unity
export const TYPE_CHAT_SUBMIT_USER_MESSAGE = 'Chat.SubmitUserMessage';
export const TYPE_CHAT_OPEN_REPORT = 'Chat.OpenReport';
export const TYPE_CHAT_ROLLBACK_TO_TURN = 'Chat.RollbackToTurn';
export const TYPE_CHAT_RESTART = 'Chat.Restart';
/** Edit the content of an existing user/assistant message in place. Payload:
 *  { historyIndex, text, removedAttachments, turnIndex }. `removedAttachments` =
 *  indexes into the user row's attachment chips unattached during the edit
 *  (`turnIndex` locates the turn snapshot). The renderer optimistically updates
 *  its bubble; the app mutates session.history + persists (no echo back — EXCEPT
 *  when attachments were removed: deleting a hidden image row shifts history
 *  indices, so the app re-pushes the full history). */
export const TYPE_CHAT_EDIT_MESSAGE = 'Chat.EditMessage';
/** User picked a new outfit in the outfit dialog. Payload: { outfitIndex } (stable KK coordinate
 *  id). The app applies it to the avatar, shows a rewindable user_action row, and runs a hidden
 *  turn so the character reacts. */
export const TYPE_CHAT_CHANGE_OUTFIT = 'Chat.ChangeOutfit';
/** The worn outfit changed — user pick, the AI's ChangeOutfit tool, or a rollback restoring an
 *  older one. Payload: { outfitIndex, outfitName }. */
export const TYPE_CHAT_OUTFIT_CHANGED = 'Chat.OutfitChanged';
/** The background task roster changed (start / finish / announce / dismiss / session swap).
 *  Payload: { tasks: [{ id, kind, label, status, startedAt, announced }] }. Drives the
 *  status-bar task chips. */
export const TYPE_CHAT_BG_TASKS = 'Chat.BgTasks';
/** User clicked the ✕ on a status-bar task chip. Payload: { taskId }. */
export const TYPE_CHAT_DISMISS_BG_TASK = 'Chat.DismissBgTask';
/** Stop the current AI turn. Unity is the single decision point: if a reply is
 *  still streaming, cancel + commit the partial as assistant history; if the
 *  stream has finished but TTS is still speaking, just stop the speech pipeline.
 *  Renderer fires this without distinguishing the two cases. */
export const TYPE_CHAT_STOP = 'Chat.Stop';

// Chat — Unity → Electron (continued). Speaking edges fired by SentenceSpeechPipeline:
// active=true once the first audible TTS sample lands, active=false once playback drains.
export const TYPE_CHAT_SPEAKING = 'Chat.Speaking';

// NOTE: voice/hands-free/wake-word state used to live here as Chat.* envelopes.
// That was a misuse of the protocol — those messages never traverse the WS
// (renderer ↔ app only). Commands are now direct ChatApi methods exposed
// via the pywebview JS bridge (window.app.startRecording(), etc.) and
// state pushes flow through appEvents.ts.

// Chat — TTS mute toggle (speaker icon in the header)
export const TYPE_CHAT_SET_VOICE_MODE = 'Chat.SetVoiceMode';         // Electron → Unity
export const TYPE_CHAT_VOICE_MODE_CHANGED = 'Chat.VoiceModeChanged'; // Unity → Electron

// Chat — touch mode toggle (avatar caress). The toggle drives Unity's AibuToucher via Python;
// the changed-event reflects Unity's actual state (also flips on keyboard toggle / AI-reject auto-off).
export const TYPE_CHAT_SET_TOUCH_MODE = 'Chat.SetTouchMode';         // renderer → Python → Unity
export const TYPE_CHAT_TOUCH_MODE_CHANGED = 'Chat.TouchModeChanged'; // Python → renderer

// Chat — head/eye look-at tracking toggles (avatar follows the camera). Per-chat persisted;
// Python-authoritative (Unity is a pure consumer), echoed back to the renderer on change.
export const TYPE_CHAT_SET_HEAD_TRACKING = 'Chat.SetHeadTracking';         // renderer → Python → Unity
export const TYPE_CHAT_HEAD_TRACKING_CHANGED = 'Chat.HeadTrackingChanged'; // Python → renderer
export const TYPE_CHAT_SET_EYE_TRACKING = 'Chat.SetEyeTracking';           // renderer → Python → Unity
export const TYPE_CHAT_EYE_TRACKING_CHANGED = 'Chat.EyeTrackingChanged';   // Python → renderer

// TTS — barge-in. Sent by app when STT detects user speech onset during
// active TTS playback so Unity can drain its StreamingAudioBuffer.
export const TYPE_TTS_CANCEL = 'Tts.Cancel';                         // app → Unity

// Chat — Errors (Unity → Electron, surfaced as a dismissible banner)
export const TYPE_CHAT_ERROR = 'Chat.Error';

// Chat Save Slot management
export const TYPE_CHAT_LIST_SAVES = 'Chat.ListSaves';
export const TYPE_CHAT_SAVES_LIST = 'Chat.SavesList';
export const TYPE_CHAT_GET_CHARACTERS = 'Chat.GetCharacters';
export const TYPE_CHAT_CHARACTERS_LIST = 'Chat.CharactersList';
export const TYPE_CHAT_LOAD_SAVE = 'Chat.LoadSave';
export const TYPE_CHAT_CREATE_NEW = 'Chat.CreateNew';
export const TYPE_CHAT_DELETE_SAVE = 'Chat.DeleteSave';
/** End the live session: unloads the character in Unity (Session.End) and drops the
 *  orchestrator session without deleting the save. Renderer returns to the home page. */
export const TYPE_CHAT_END_SESSION = 'Chat.EndSession';
/** Edit the active chat's per-chat settings (user name, voice mode, voice provider, LLM config,
 *  workspace roots). Payload: { userName?, voiceMode?, voiceProvider?, llmConfigId?, workspaceRoots? }. */
export const TYPE_CHAT_UPDATE_SETTINGS = 'Chat.UpdateSettings';
// Character creation page
export const TYPE_CHAT_SAVE_CHARACTER = 'Chat.SaveCharacter';                 // renderer → Python (create or edit)
export const TYPE_CHAT_DELETE_CHARACTER = 'Chat.DeleteCharacter';             // renderer → Python (remove)
export const TYPE_CHAT_INSPECT_MODEL_EMOTIONS = 'Chat.InspectModelEmotions';  // renderer → Python (ask Unity)
export const TYPE_CHAT_MODEL_EMOTIONS = 'Chat.ModelEmotions';                 // Python → renderer (inspected emotions)
export const TYPE_CHAT_INSPECT_MODEL_COORDINATES = 'Chat.InspectModelCoordinates'; // renderer → Python (KK outfits)
export const TYPE_CHAT_MODEL_COORDINATES = 'Chat.ModelCoordinates';          // Python → renderer (the outfit list)
export const TYPE_CHAT_IMPORT_CHARACTER = 'Chat.ImportCharacter';            // renderer → Python (import a .wcc bundle)
/** Python → renderer: a resumed chat's character no longer exists. Opens the rebind
 *  dialog. Payload: { slot, characterName }. */
export const TYPE_CHAT_CHARACTER_MISSING = 'Chat.CharacterMissing';
/** renderer → Python: re-bind a saved chat to another character and resume it.
 *  Payload: { slot, characterId }. */
export const TYPE_CHAT_REBIND_SAVE = 'Chat.RebindSave';

/** Python → renderer: files were dropped onto the window, with their FULL disk paths
 *  (only the Python-side pywebview DOM handler can see those). The renderer decides what
 *  the drop means for its active view — home imports .wcc character bundles; the chat
 *  view can treat drops as attachments. Payload: { paths: string[] }. */
export const TYPE_APP_FILES_DROPPED = 'App.FilesDropped';


// Electron → Unity
export const TYPE_CLIENT_READY = 'ClientReady';
export const TYPE_QUESTION_ANSWER = 'QuestionAnswer';
export const TYPE_PERMISSION_DECISION = 'PermissionDecision';
export const TYPE_REPORT_CLOSED = 'ReportClosed';
export const TYPE_ERROR = 'Error';

// Migration: chat orchestrator moves to Python. These envelopes are the new boundary
// between the Python-side orchestrator and the Unity-side avatar runtime. The renderer
// itself doesn't use them directly — they travel Python ↔ Unity over the WS — but
// the constants live here so the wire format is documented in one place.

// Python → Unity (session lifecycle)
export const TYPE_SESSION_BEGIN = 'Session.Begin';
export const TYPE_SESSION_END   = 'Session.End';

// Python → Unity (avatar commands)
export const TYPE_AVATAR_APPLY_EMOTION = 'Avatar.ApplyEmotion';
export const TYPE_AVATAR_APPLY_OUTFIT  = 'Avatar.ApplyOutfit';
export const TYPE_AVATAR_RUN_ACTION    = 'Avatar.RunAction';
export const TYPE_AVATAR_SET_STATUS    = 'Avatar.SetStatus';

// Characters are owned by the app now (Python store); Character.GetData / Character.List are
// retired. The only remaining Character.* request:
// Inspect a .vrm's supported emotions from the file (no scene load) — used by character creation.
export const TYPE_CHARACTER_INSPECT_MODEL_EMOTIONS        = 'Character.InspectModelEmotions';
export const TYPE_CHARACTER_INSPECT_MODEL_EMOTIONS_RESULT = 'Character.InspectModelEmotionsResult';

// Python ↔ Unity (tools still live in Unity; Python RPCs them)
export const TYPE_TOOL_EXECUTE = 'Tool.Execute';
export const TYPE_TOOL_RESULT  = 'Tool.Result';

export interface Envelope<P = unknown> {
  id: string;
  type: string;
  replyTo?: string;
  payload?: P;
}

// Payload / DTO interfaces live in ./payloads, grouped by message family — import them
// directly from there (e.g. `import type { ChatInitPayload } from '../payloads/chat'`).
// This file holds the message-type constants, the Envelope frame, and newId().

export function newId(): string {
  // crypto.randomUUID() is available in modern Chromium / Electron renderers.
  return 'm_' + crypto.randomUUID().replaceAll('-', '');
}
