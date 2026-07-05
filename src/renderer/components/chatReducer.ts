/** Chat window state shape + reducer, extracted from ChatView. The reducer is the single
 *  writer of the chat scroll-back and all the voice/connection flags; ChatView wires envelopes
 *  to dispatch() and the rest of the UI reads the resulting state. */
import type {
  ChatAppendTokenPayload, ChatBgTaskEntry, ChatBgTasksPayload, ChatErrorPayload,
  ChatHistoryEntry, ChatHistoryPayload, ChatInitPayload,
  ChatOutfitChangedPayload, ChatPushEntryPayload, ChatSpeakingPayload, ChatToolActivityPayload,
  ChatTypingPayload, ChatUserNameChangedPayload, ChatVoiceModeChangedPayload,
  ChatTouchModeChangedPayload, ChatHeadTrackingChangedPayload, ChatEyeTrackingChangedPayload,
  ChatSaveMetadata,
} from '../payloads/chat';
import type { CharacterRecordWire } from '../payloads/character';
import type {
  AutoSubmitChangedEvent, HandsFreeChangedEvent, VoiceBusyEvent,
  VoicePartialEvent, VoiceRecordingEvent, WakeWordArmedEvent, WakeWordChangedEvent,
} from '../payloads/voice';
import type { UnityConnectionEvent } from '../payloads/connection';


export interface ChatState {
  ready: boolean;
  assistantName: string;
  /** The character's profile picture as a data URL ('' when none) — shown in the chat header. */
  assistantAvatar: string;
  userName: string;
  /** Single source of truth for the chat scroll-back. Updated by:
   *    'history'     — full replace (session init / resume / rollback)
   *    'pushEntry'   — append one new entry (user msg, assistant round start,
   *                    tool_activity event)
   *    'appendToken' — append a text delta to the LAST entry (streaming).
   *  No separate "live bubble" — the in-progress assistant row IS the last
   *  history entry. */
  history: ChatHistoryEntry[];
  typing: boolean;
  /** True while the assistant is producing audible TTS. Combines with `typing`
   * to drive the Send→Stop button swap: button is "Stop" whenever the assistant
   * is doing anything (generating a reply or speaking it out), "Send" otherwise. */
  speaking: boolean;
  toolActivity: string;
  voiceSupported: boolean;
  voiceRecording: boolean;
  voiceBusy: boolean;
  voiceBusyReason: 'loading' | 'transcribing' | null;
  voiceMode: boolean;
  /** Whether the active character has a voice for the active TTS provider. When false the
   *  assistant-voice toggle is disabled and voice mode stays off (text lip-sync). */
  voiceAvailable: boolean;
  /** Whether avatar caress (touch) mode is on. Mirrors Unity's AibuToucher — toggled from the
   *  header, but Unity can also flip it (keyboard toggle, AI-reject auto-off). */
  touchMode: boolean;
  /** Per-chat head/eye look-at tracking (avatar follows the camera). Toggled from the sidebar. */
  headTracking: boolean;
  eyeTracking: boolean;
  /** Worn outfit (stable KK coordinate id) — marks the current card in the outfit dialog.
   *  -1 for VRM models / unknown. Updated by Chat.Init and Chat.OutfitChanged. */
  currentOutfitIndex: number;
  /** Background UwU helpers (sub-agent tasks) — drives the "helpers working" pill.
   *  Replaced wholesale by every Chat.BgTasks push. */
  bgTasks: ChatBgTaskEntry[];
  /** Active TTS provider for the current chat ("pocket"/"elevenlabs"). Prefills the per-chat
   *  settings dialog. */
  voiceProvider: string;
  /** Id of the LLM config the current chat uses ("" = follow the default). Prefills the
   *  per-chat settings dialog. */
  llmConfigId: string;
  /** Workspace folders in force for the current chat. Prefills the settings dialog. */
  workspaceRoots: string[];
  /** Hands-free conversation mode: mic stays open, endpoint detector segments
   * utterances, and (if autoSubmit is true) each utterance is sent automatically. */
  handsFree: boolean;
  /** Only meaningful while handsFree is true. False = endpoint transcripts just
   * fill the draft like PTT dictation does. */
  autoSubmit: boolean;
  /** Live partial-transcript text from the streaming recognizer. Cleared on
   * recording end / explicit toggle off. */
  voicePartial: string;
  /** Wake-word gating ("okay google" style). When enabled, hands-free
   * auto-submit only fires for utterances spoken within `wakeWordArmExpiresAt`
   * ms of clock time. `wakeWordPhrase` is the configured phrase (for tooltips
   * + status). */
  wakeWordEnabled: boolean;
  wakeWordPhrase: string;
  wakeWordArmExpiresAt: number;
  error: { code: string; message: string } | null;
  saves: ChatSaveMetadata[];
  characters: CharacterRecordWire[];
  activeSlot: string;
  /** Whether the Unity backend WS is connected. Defaults true so a fast normal
   * startup doesn't flash the overlay; a disconnect/failed-connect flips it false. */
  unityConnected: boolean;
  /** Chat-loading overlay: true from the moment a chat is started/resumed until the model + TTS +
   *  STT have all resolved (Chat.Ready). Set optimistically on the start/resume click so there's no
   *  flash of interactive chat, and authoritatively by Chat.Loading/Chat.Ready from the backend. */
  chatLoading: boolean;
  /** Per-subsystem readiness shown in the loading overlay's checklist. */
  chatLoadStatus: { model: boolean; tts: boolean; stt: boolean };
  /** First-run model-download progress shown in the loading overlay, or null when nothing is
   *  downloading. `percent` is 0..1; `total`/`completed` are bytes. */
  chatDownload: { completed: number; total: number; percent: number; label: string } | null;
}

export const initialState: ChatState = {
  ready: false,
  assistantName: 'Assistant',
  assistantAvatar: '',
  userName: 'User',
  history: [],
  typing: false,
  speaking: false,
  toolActivity: '',
  voiceSupported: false,
  voiceRecording: false,
  voiceBusy: false,
  voiceBusyReason: null,
  voiceMode: false,
  voiceAvailable: false,
  touchMode: false,
  headTracking: true,
  eyeTracking: true,
  currentOutfitIndex: -1,
  bgTasks: [],
  voiceProvider: 'pocket',
  llmConfigId: '',
  workspaceRoots: [],
  handsFree: false,
  autoSubmit: true,
  voicePartial: '',
  wakeWordEnabled: false,
  wakeWordPhrase: '',
  wakeWordArmExpiresAt: 0,
  error: null,
  saves: [],
  characters: [],
  activeSlot: 'default',
  unityConnected: true,
  chatLoading: false,
  chatLoadStatus: { model: false, tts: false, stt: false },
  chatDownload: null,
};

export type Action =
  | { kind: 'init'; payload: ChatInitPayload }
  | { kind: 'history'; payload: ChatHistoryPayload }
  | { kind: 'pushEntry'; payload: ChatPushEntryPayload }
  | { kind: 'appendToken'; payload: ChatAppendTokenPayload }
  | { kind: 'typing'; payload: ChatTypingPayload }
  | { kind: 'speaking'; payload: ChatSpeakingPayload }
  | { kind: 'tool'; payload: ChatToolActivityPayload }
  | { kind: 'voiceRecording'; payload: VoiceRecordingEvent }
  | { kind: 'voiceBusy'; payload: VoiceBusyEvent }
  | { kind: 'voiceMode'; payload: ChatVoiceModeChangedPayload }
  | { kind: 'touchMode'; payload: ChatTouchModeChangedPayload }
  | { kind: 'outfitChanged'; payload: ChatOutfitChangedPayload }
  | { kind: 'bgTasks'; payload: ChatBgTasksPayload }
  | { kind: 'headTracking'; payload: ChatHeadTrackingChangedPayload }
  | { kind: 'eyeTracking'; payload: ChatEyeTrackingChangedPayload }
  | { kind: 'userName'; payload: ChatUserNameChangedPayload }
  | { kind: 'editEntry'; payload: { index: number; text: string; removedAttachments?: number[] } }
  | { kind: 'voicePartial'; payload: VoicePartialEvent }
  | { kind: 'handsFree'; payload: HandsFreeChangedEvent }
  | { kind: 'autoSubmit'; payload: AutoSubmitChangedEvent }
  | { kind: 'wakeWord'; payload: WakeWordChangedEvent }
  | { kind: 'wakeWordArmed'; payload: WakeWordArmedEvent }
  | { kind: 'error'; payload: ChatErrorPayload }
  | { kind: 'dismissError' }
  | { kind: 'saves'; payload: ChatSaveMetadata[] }
  | { kind: 'characters'; payload: CharacterRecordWire[] }
  | { kind: 'upsertCharacter'; payload: CharacterRecordWire }
  | { kind: 'clearSession' }
  | { kind: 'unityConnection'; payload: UnityConnectionEvent }
  | { kind: 'chatLoadingStart' }
  | { kind: 'chatLoadingCancel' }
  | { kind: 'chatLoading'; payload: { model: boolean; tts: boolean; stt: boolean; download?: { completed: number; total: number; percent: number; label: string } } }
  | { kind: 'chatReady' };

export function reducer(state: ChatState, action: Action): ChatState {
  switch (action.kind) {
    case 'init':
      return {
        ...state,
        ready: true,
        assistantName: action.payload.assistantDisplayName || 'Assistant',
        assistantAvatar: action.payload.assistantAvatar || '',
        userName: action.payload.userName || 'User',
        history: action.payload.history ?? [],
        voiceSupported: !!action.payload.voiceSupported,
        voiceMode: !!action.payload.voiceMode,
        voiceAvailable: !!action.payload.voiceAvailable,
        touchMode: !!action.payload.touchMode,
        // Head/eye tracking default ON when the field is absent from an older save.
        headTracking: action.payload.headTracking ?? true,
        eyeTracking: action.payload.eyeTracking ?? true,
        voiceProvider: action.payload.voiceProvider || 'pocket',
        llmConfigId: action.payload.llmConfigId ?? '',
        workspaceRoots: action.payload.workspaceRoots ?? [],
        activeSlot: action.payload.activeSlot || 'default',
        currentOutfitIndex: action.payload.currentOutfitIndex ?? -1,
        // Fresh session view; the app re-pushes the live roster right after Chat.Init.
        bgTasks: [],
      };

    case 'history': {
      // Full replace — session start / resume / rollback. The streaming flow uses
      // pushEntry + appendToken instead so we never hit the duplicate-bubble class
      // of bugs that an interim live-bubble would create.
      return { ...state, history: action.payload.entries ?? [] };
    }

    case 'pushEntry': {
      const entry = action.payload.entry;
      if (!entry) return state;
      return { ...state, history: [...state.history, entry] };
    }

    case 'appendToken': {
      // Append the delta to the LAST entry's text. In the normal flow the last
      // entry is always an assistant row (the Python side pushes a fresh assistant
      // entry right before any tokens flow). Defensive: if it's not, do nothing
      // rather than mutating a user/tool row.
      const delta = action.payload.delta;
      const n = state.history.length;
      if (!delta || n === 0) return state;
      const last = state.history[n - 1];
      if (!last || last.role !== 'assistant') return state;
      const updated: ChatHistoryEntry = { ...last, text: (last.text || '') + delta };
      return { ...state, history: [...state.history.slice(0, n - 1), updated] };
    }

    case 'typing':
      return { ...state, typing: !!action.payload.active };

    case 'speaking':
      return { ...state, speaking: !!action.payload.active };

    case 'tool':
      return { ...state, toolActivity: action.payload.label ?? '' };

    case 'voiceRecording':
      // Clear any lingering partial caption when recording ends — the final
      // transcript (if any) has already been folded into the draft or auto-sent.
      return {
        ...state,
        voiceRecording: !!action.payload.active,
        voicePartial: action.payload.active ? state.voicePartial : '',
      };

    case 'voicePartial':
      return { ...state, voicePartial: action.payload.text ?? '' };

    case 'handsFree':
      // Mic-recording active reflects recorder state, which the app will
      // update via its own envelope; we only mirror the toggle flag here.
      return { ...state, handsFree: !!action.payload.enabled };

    case 'autoSubmit':
      return { ...state, autoSubmit: !!action.payload.enabled };

    case 'wakeWord':
      return {
        ...state,
        wakeWordEnabled: !!action.payload.enabled,
        wakeWordPhrase: action.payload.phrase ?? '',
        // Disabling clears any lingering armed indicator.
        wakeWordArmExpiresAt: action.payload.enabled ? state.wakeWordArmExpiresAt : 0,
      };

    case 'wakeWordArmed':
      return {
        ...state,
        wakeWordArmExpiresAt: action.payload.armed
          ? Date.now() + Math.max(0, action.payload.expiresInSeconds) * 1000
          : 0,
      };

    case 'voiceBusy':
      return {
        ...state,
        voiceBusy: !!action.payload.busy,
        voiceBusyReason: action.payload.busy ? (action.payload.reason ?? null) : null,
      };

    case 'voiceMode':
      return {
        ...state,
        voiceMode: !!action.payload.enabled,
        // `available` is optional on the wire; preserve the current flag when absent.
        voiceAvailable: action.payload.available ?? state.voiceAvailable,
      };

    case 'touchMode':
      return { ...state, touchMode: !!action.payload.enabled };

    case 'headTracking':
      return { ...state, headTracking: !!action.payload.enabled };

    case 'eyeTracking':
      return { ...state, eyeTracking: !!action.payload.enabled };

    case 'userName':
      return { ...state, userName: action.payload.userName || 'User' };

    case 'outfitChanged':
      return { ...state, currentOutfitIndex: action.payload.outfitIndex ?? -1 };

    case 'bgTasks':
      return { ...state, bgTasks: action.payload.tasks ?? [] };

    case 'editEntry': {
      const { index, text, removedAttachments } = action.payload;
      if (index < 0 || index >= state.history.length) return state;
      const history = state.history.slice();
      const prev = history[index];
      const attachments = removedAttachments?.length && prev.attachments
        ? prev.attachments.filter((_, i) => !removedAttachments.includes(i))
        : prev.attachments;
      history[index] = { ...prev, text, attachments };
      return { ...state, history };
    }

    case 'error':
      return {
        ...state,
        error: {
          code: action.payload.code ?? '',
          message: action.payload.message ?? '',
        },
        // A failure during session start (e.g. character_missing) returns to home — never leave the
        // loading overlay stranded. Safe in-chat too: the overlay blocks input, so no turn-level
        // error can fire while it's up.
        chatLoading: false,
      };

    case 'dismissError':
      return state.error ? { ...state, error: null } : state;

    case 'saves':
      return { ...state, saves: action.payload };

    case 'characters':
      return { ...state, characters: action.payload };

    case 'upsertCharacter': {
      // Optimistic local merge after an edit, so reopening the editor reflects the change before the
      // backend's refreshed list (delayed by the async voice encode) arrives. Merge by id; the real
      // Chat.CharactersList replaces this wholesale once it lands.
      const rec = action.payload;
      if (!rec.id) return state;
      const found = state.characters.some(c => c.id === rec.id);
      const characters = found
        ? state.characters.map(c => (c.id === rec.id ? { ...c, ...rec } : c))
        : [...state.characters, rec];
      return { ...state, characters };
    }

    case 'clearSession':
      // The active chat went away (e.g. user deleted it). Drop everything tied to the
      // live session so the home page stops offering "Resume current chat" on stale data.
      return {
        ...state,
        ready: false,
        history: [],
        typing: false,
        speaking: false,
        toolActivity: '',
        bgTasks: [],
        activeSlot: 'default',
        chatLoading: false,
        chatLoadStatus: { model: false, tts: false, stt: false },
      };

    case 'unityConnection':
      return { ...state, unityConnected: !!action.payload.connected };

    case 'chatLoadingStart':
      // Optimistic: the user clicked start/resume. Block the chat before any backend round-trip so
      // there's no flash of interactive chat once Chat.Init flips the view.
      return { ...state, chatLoading: true, chatLoadStatus: { model: false, tts: false, stt: false }, chatDownload: null };

    case 'chatLoadingCancel':
      // The load didn't proceed but it's not an error either (e.g. the rebind dialog opened
      // instead) — just drop the optimistic loading overlay.
      return { ...state, chatLoading: false };

    case 'chatLoading':
      return {
        ...state,
        chatLoading: true,
        chatLoadStatus: {
          model: !!action.payload.model,
          tts: !!action.payload.tts,
          stt: !!action.payload.stt,
        },
        chatDownload: action.payload.download ?? null,
      };

    case 'chatReady':
      return {
        ...state,
        chatLoading: false,
        chatLoadStatus: { model: true, tts: true, stt: true },
        chatDownload: null,
      };

    default:
      return state;
  }
}
