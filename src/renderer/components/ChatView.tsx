import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState } from 'react';
import { ArrowLeft, ArrowUp, Ear, Eraser, Eye, FolderOpen, Hand, Headphones, LoaderCircle, LogOut, Menu, Mic, Paperclip, Pencil, Plus, RotateCcw, ScanFace, Send, Shirt, SlidersHorizontal, Square, Trash2, Volume2, VolumeX, X } from 'lucide-react';
import { ConfigView } from './ConfigView';
import { NewChatDialog } from './NewChatDialog';
import { OutfitPickerDialog } from './OutfitPickerDialog';
import { RebindDialog } from './RebindDialog';
import { Titlebar } from './Titlebar';
import { CharacterEditor } from './CharacterEditor';
import { reducer, initialState } from './chatReducer';
import { HomeScreen } from './HomeScreen';
import { StatusBar } from './StatusBar';
import { ConnectionOverlay } from './ConnectionOverlay';
import { ChatLoadingOverlay } from './ChatLoadingOverlay';
import { MessageBubble } from './MessageBubble';
import { AttachmentChips, attachmentRefForPath } from './AttachmentChips';
import {
  AppEvent,
  EVT_AUTO_SUBMIT_CHANGED,
  EVT_HANDS_FREE_CHANGED,
  EVT_VOICE_BUSY,
  EVT_VOICE_PARTIAL,
  EVT_VOICE_RECORDING,
  EVT_VOICE_TRANSCRIPT,
  EVT_UNITY_CONNECTION,
  EVT_WAKE_WORD_ARMED,
  EVT_WAKE_WORD_CHANGED,
} from '../appEvents';
import type {
  AutoSubmitChangedEvent,
  HandsFreeChangedEvent,
  VoiceBusyEvent,
  VoicePartialEvent,
  VoiceRecordingEvent,
  VoiceTranscriptEvent,
  WakeWordArmedEvent,
  WakeWordChangedEvent,
} from '../payloads/voice';
import type { UnityConnectionEvent } from '../payloads/connection';
import type {
  ChatSettings,
  ChatAppendTokenPayload,
  ChatErrorPayload,
  ChatHistoryPayload,
  ChatInitPayload,
  ChatPlayerInputPayload,
  ChatPushEntryPayload,
  ChatSpeakingPayload,
  ChatToolActivityPayload,
  ChatTypingPayload,
  ChatUserNameChangedPayload,
  ChatVoiceModeChangedPayload,
  ChatTouchModeChangedPayload,
  ChatHeadTrackingChangedPayload,
  ChatEyeTrackingChangedPayload,
  ChatOutfitChangedPayload,
  ChatBgTasksPayload,
  ChatSaveMetadata,
} from '../payloads/chat';
import type { CharacterRecordWire } from '../payloads/character';
import {
  Envelope,
  TYPE_APP_FILES_DROPPED,
  TYPE_CHAT_APPEND_TOKEN,
  TYPE_CHAT_CHARACTER_MISSING,
  TYPE_CHAT_ERROR,
  TYPE_CHAT_IMPORT_CHARACTER,
  TYPE_CHAT_REBIND_SAVE,
  TYPE_CHAT_HISTORY,
  TYPE_CHAT_INIT,
  TYPE_CHAT_LOADING,
  TYPE_CHAT_READY,
  TYPE_CHAT_OPEN_REPORT,
  TYPE_CHAT_PLAYER_INPUT,
  TYPE_CHAT_PUSH_ENTRY,
  TYPE_CHAT_RESTART,
  TYPE_CHAT_ROLLBACK_TO_TURN,
  TYPE_CHAT_EDIT_MESSAGE,
  TYPE_CHAT_SET_VOICE_MODE,
  TYPE_CHAT_SET_TOUCH_MODE,
  TYPE_CHAT_TOUCH_MODE_CHANGED,
  TYPE_CHAT_CHANGE_OUTFIT,
  TYPE_CHAT_OUTFIT_CHANGED,
  TYPE_CHAT_BG_TASKS,
  TYPE_CHAT_DISMISS_BG_TASK,
  TYPE_CHAT_SET_HEAD_TRACKING,
  TYPE_CHAT_HEAD_TRACKING_CHANGED,
  TYPE_CHAT_SET_EYE_TRACKING,
  TYPE_CHAT_EYE_TRACKING_CHANGED,
  TYPE_CHAT_SPEAKING,
  TYPE_CHAT_STOP,
  TYPE_CHAT_SUBMIT_USER_MESSAGE,
  TYPE_CHAT_TOOL_ACTIVITY,
  TYPE_CHAT_TYPING,
  TYPE_CHAT_USER_NAME_CHANGED,
  TYPE_CHAT_VOICE_MODE_CHANGED,
  TYPE_CHAT_LIST_SAVES,
  TYPE_CHAT_SAVES_LIST,
  TYPE_CHAT_GET_CHARACTERS,
  TYPE_CHAT_CHARACTERS_LIST,
  TYPE_CHAT_LOAD_SAVE,
  TYPE_CHAT_CREATE_NEW,
  TYPE_CHAT_DELETE_SAVE,
  TYPE_CHAT_DELETE_CHARACTER,
  TYPE_CHAT_UPDATE_SETTINGS,
  TYPE_CHAT_END_SESSION,
} from '../protocol';

export function ChatView() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [draft, setDraft] = useState('');
  // The window opens on the HOME page (character + saved-chat picker). It switches to
  // 'chat' only once a real Chat.Init arrives (i.e. the user picked a character or
  // resumed a save). The Home button in the chat header returns here — the Python-side
  // session stays alive, so resuming/switching from home just re-enters it.
  const [view, setView] = useState<'home' | 'chat'>('home');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  // Always-on-top (pin) state for the whole window. Held here — not in Titlebar — so the
  // home and chat views share one source of truth and their pin icons stay in sync.
  const [alwaysOnTop, setAlwaysOnTop] = useState(false);
  const toggleAlwaysOnTop = () => {
    const next = !alwaysOnTop;
    window.app.setAlwaysOnTop(next);
    setAlwaysOnTop(next);
  };
  // Seed the pin icon from the persisted preference. The window is already created with the
  // saved on_top state (app.py install_chat_window), so this only syncs the icon — it
  // doesn't re-apply, avoiding a redundant setAlwaysOnTop round-trip.
  useEffect(() => {
    let cancelled = false;
    window.app.getConfig()
      .then(c => { if (!cancelled) setAlwaysOnTop(!!c?.ui?.alwaysOnTop); })
      .catch(() => { /* defaults to false */ });
    return () => { cancelled = true; };
  }, []);
  const [showCharCreation, setShowCharCreation] = useState(false);
  // When set, the new-chat dialog is open for this character id (chosen settings → Chat.CreateNew).
  const [newChatFor, setNewChatFor] = useState<string | null>(null);
  // Set when a resumed chat's character no longer exists (Chat.CharacterMissing) — opens
  // the rebind dialog so the chat can continue with another character.
  const [rebindFor, setRebindFor] = useState<{ slot: string; characterName: string } | null>(null);
  // When true, the same dialog is open in EDIT mode for the active chat (→ Chat.UpdateSettings).
  const [showChatSettings, setShowChatSettings] = useState(false);
  // Outfit picker modal (KK models with 2+ outfits only). Confirming is a USER action —
  // the app applies the outfit and the character reacts to it in a hidden turn.
  const [showOutfitPicker, setShowOutfitPicker] = useState(false);
  // Absolute paths of files attached to the next message. Images go inline; other files are
  // handed to the AI as a path (+ workspace access). Cleared on send.
  const [attachments, setAttachments] = useState<string[]>([]);
  // When set, the creation modal opens in EDIT mode prefilled from this record; null = create new.
  const [editingCharacter, setEditingCharacter] = useState<CharacterRecordWire | null>(null);
  // Snapshot of the draft taken when recording starts; the transcript is appended onto this
  // so the user can keep typed text + dictate more on top of it.
  const preDictationRef = useRef('');
  const scrollRef = useRef<HTMLDivElement>(null);
  // Whether to keep the view pinned to the bottom as new content streams in. True while the user is
  // reading at (or near) the bottom; flips false once they scroll up into history, so incoming AI
  // responses don't yank them back down. Re-armed when they scroll back near the bottom.
  const stickToBottomRef = useRef(true);
  // Distinguishes (re)entering the chat view — which always jumps to the latest — from content
  // updates while already in chat, which only follow the bottom when stickToBottomRef is set.
  const prevViewRef = useRef(view);
  // How close to the bottom (px) still counts as "at the bottom" for auto-follow.
  const BOTTOM_STICK_THRESHOLD_PX = 80;
  const onChatScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom <= BOTTOM_STICK_THRESHOLD_PX;
  }, []);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Wall-clock of the last sign of progress (turn start, streamed token, or new entry). Drives
  // the "still thinking" indicator: if a turn is in flight but no new text has arrived for a
  // moment and no tool is running, we show the dots so the chat doesn't look frozen — e.g. while
  // the model generates a tool call whose arguments stream invisibly.
  const lastProgressAtRef = useRef(0);

  useEffect(() => {
    const off = window.app.onChatEvent((env: Envelope) => {
      switch (env.type) {
        case TYPE_CHAT_INIT:
          // A real session was created/resumed → enter the chat view.
          dispatch({ kind: 'init', payload: env.payload as ChatInitPayload });
          setDraft('');
          setView('chat');
          window.app.sendChat(TYPE_CHAT_LIST_SAVES, {});
          window.app.sendChat(TYPE_CHAT_GET_CHARACTERS, {});
          break;
        case TYPE_CHAT_LOADING:
          dispatch({ kind: 'chatLoading', payload: env.payload as { model: boolean; tts: boolean; stt: boolean; download?: { completed: number; total: number; percent: number; label: string } } });
          break;
        case TYPE_CHAT_READY:
          dispatch({ kind: 'chatReady' });
          break;
        case TYPE_CHAT_SAVES_LIST: {
          const payload = env.payload as { slots: ChatSaveMetadata[] };
          dispatch({ kind: 'saves', payload: payload.slots ?? [] });
          break;
        }
        case TYPE_CHAT_CHARACTERS_LIST: {
          const payload = env.payload as { characters: CharacterRecordWire[] };
          dispatch({ kind: 'characters', payload: payload.characters ?? [] });
          break;
        }
        case TYPE_CHAT_HISTORY:
          dispatch({ kind: 'history', payload: env.payload as ChatHistoryPayload });
          break;
        case TYPE_CHAT_PUSH_ENTRY:
          lastProgressAtRef.current = Date.now();
          dispatch({ kind: 'pushEntry', payload: env.payload as ChatPushEntryPayload });
          break;
        case TYPE_CHAT_APPEND_TOKEN:
          lastProgressAtRef.current = Date.now();
          dispatch({ kind: 'appendToken', payload: env.payload as ChatAppendTokenPayload });
          break;
        case TYPE_CHAT_TYPING:
          // A turn starting counts as fresh progress — reset the stall clock.
          if ((env.payload as ChatTypingPayload)?.active) lastProgressAtRef.current = Date.now();
          dispatch({ kind: 'typing', payload: env.payload as ChatTypingPayload });
          break;
        case TYPE_CHAT_SPEAKING:
          dispatch({ kind: 'speaking', payload: env.payload as ChatSpeakingPayload });
          break;
        case TYPE_CHAT_TOOL_ACTIVITY:
          dispatch({ kind: 'tool', payload: env.payload as ChatToolActivityPayload });
          break;
        case EVT_VOICE_RECORDING:
          dispatch({ kind: 'voiceRecording', payload: env.payload as VoiceRecordingEvent });
          break;
        case EVT_VOICE_BUSY:
          dispatch({ kind: 'voiceBusy', payload: env.payload as VoiceBusyEvent });
          break;
        case TYPE_CHAT_VOICE_MODE_CHANGED:
          dispatch({ kind: 'voiceMode', payload: env.payload as ChatVoiceModeChangedPayload });
          break;
        case TYPE_CHAT_TOUCH_MODE_CHANGED:
          dispatch({ kind: 'touchMode', payload: env.payload as ChatTouchModeChangedPayload });
          break;
        case TYPE_CHAT_OUTFIT_CHANGED:
          dispatch({ kind: 'outfitChanged', payload: env.payload as ChatOutfitChangedPayload });
          break;
        case TYPE_CHAT_BG_TASKS:
          dispatch({ kind: 'bgTasks', payload: env.payload as ChatBgTasksPayload });
          break;
        case TYPE_CHAT_HEAD_TRACKING_CHANGED:
          dispatch({ kind: 'headTracking', payload: env.payload as ChatHeadTrackingChangedPayload });
          break;
        case TYPE_CHAT_EYE_TRACKING_CHANGED:
          dispatch({ kind: 'eyeTracking', payload: env.payload as ChatEyeTrackingChangedPayload });
          break;
        case TYPE_CHAT_USER_NAME_CHANGED:
          dispatch({ kind: 'userName', payload: env.payload as ChatUserNameChangedPayload });
          break;
        case EVT_HANDS_FREE_CHANGED:
          dispatch({ kind: 'handsFree', payload: env.payload as HandsFreeChangedEvent });
          break;
        case EVT_AUTO_SUBMIT_CHANGED:
          dispatch({ kind: 'autoSubmit', payload: env.payload as AutoSubmitChangedEvent });
          break;
        case EVT_WAKE_WORD_CHANGED:
          dispatch({ kind: 'wakeWord', payload: env.payload as WakeWordChangedEvent });
          break;
        case EVT_WAKE_WORD_ARMED:
          dispatch({ kind: 'wakeWordArmed', payload: env.payload as WakeWordArmedEvent });
          break;
        case EVT_VOICE_PARTIAL:
          dispatch({ kind: 'voicePartial', payload: env.payload as VoicePartialEvent });
          break;
        case EVT_UNITY_CONNECTION:
          dispatch({ kind: 'unityConnection', payload: env.payload as UnityConnectionEvent });
          break;
        case EVT_VOICE_TRANSCRIPT: {
          const payload = env.payload as VoiceTranscriptEvent;
          const t = payload?.text ?? '';
          if (!t) break;
          // If the app already sent this as a Chat.SubmitUserMessage
          // (hands-free + autoSubmit), clear the dictation prefix so the next
          // utterance starts fresh. Don't touch the draft — it might be a typed
          // message in progress and we shouldn't clobber it.
          if (payload?.submitted) {
            preDictationRef.current = '';
            break;
          }
          // Append to whatever was already in the draft when recording started, mirroring the
          // original Unity flow. Add a single trailing space so the user can keep typing.
          const prefix = preDictationRef.current;
          const joiner = prefix && !prefix.endsWith(' ') ? ' ' : '';
          const next = prefix + joiner + t;
          preDictationRef.current = next.endsWith(' ') ? next : next + ' ';
          setDraft(preDictationRef.current);
          break;
        }
        case TYPE_CHAT_PLAYER_INPUT: {
          // Fired after a rollback — carries the rolled-back user message + its attachments so
          // the composer is pre-filled, ready to edit and re-send.
          const payload = env.payload as ChatPlayerInputPayload;
          setDraft(payload?.text ?? '');
          setAttachments(payload?.attachments ?? []);
          // Defer focus to the next tick — disabled state may flip during the same render pass.
          setTimeout(() => inputRef.current?.focus(), 0);
          break;
        }
        case TYPE_CHAT_CHARACTER_MISSING: {
          // The chat the user tried to resume points at a deleted character. Swap the
          // optimistic loading overlay for the rebind dialog.
          const p = env.payload as { slot?: string; characterName?: string };
          dispatch({ kind: 'chatLoadingCancel' });
          setRebindFor({ slot: p.slot || '', characterName: p.characterName || '' });
          break;
        }
        case TYPE_CHAT_ERROR:
          dispatch({ kind: 'error', payload: env.payload as ChatErrorPayload });
          break;
        // Emotion is dropped on the floor for now.
        default:
          break;
      }
    });

    window.app.notifyChatReady();
    // The home page needs these independently of any Chat.Init (we no longer auto-open a chat),
    // so pull them as soon as we mount. Idempotent with the bootstrap-side push.
    window.app.sendChat(TYPE_CHAT_GET_CHARACTERS, {});
    window.app.sendChat(TYPE_CHAT_LIST_SAVES, {});
    return () => { off(); };
  }, []);

  // Tick once a second while the wake-word arm window is open so the visual
  // "armed" state expires on its own without waiting for a app-side
  // envelope. The app sends an explicit `armed: false` after submission,
  // but if the user goes silent and the arm window times out we need this
  // local tick to clear the badge.
  const [, setNowTick] = useState(0);
  useEffect(() => {
    if (state.wakeWordArmExpiresAt <= Date.now()) return;
    const id = window.setInterval(() => setNowTick(t => t + 1), 500);
    return () => window.clearInterval(id);
  }, [state.wakeWordArmExpiresAt]);

  // While a turn is in flight, re-render periodically so the "still thinking" indicator can
  // appear once text streaming has stalled (see the chat-status block below). The interval only
  // runs during a turn, so it's idle the rest of the time.
  useEffect(() => {
    if (!state.typing) return;
    const id = window.setInterval(() => setNowTick(t => t + 1), 400);
    return () => window.clearInterval(id);
  }, [state.typing]);

  // Keep the view pinned to the bottom as new content streams in — but only while the user is already
  // reading near the bottom. If they've scrolled up into history, leave their position alone so new AI
  // responses don't yank them down. (Re)entering the chat view always jumps to the latest message:
  // `view` is in the deps because the chat-scroll element unmounts on the home page; resuming a chat
  // (Resume current chat → setView('chat')) doesn't change history, so without it the freshly remounted
  // scroll container would stay at the top instead of the latest message.
  useLayoutEffect(() => {
    if (view !== 'chat') { prevViewRef.current = view; return; }
    const el = scrollRef.current;
    if (!el) return;
    const enteringChat = prevViewRef.current !== 'chat';
    prevViewRef.current = view;
    if (enteringChat || stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
      stickToBottomRef.current = true;
    }
  }, [state.history, state.typing, state.toolActivity, view]);

  const submit = useCallback(() => {
    const text = draft.trim();
    // A message is always required — including when sending attachments.
    if (!text) return;
    window.app.sendChat(TYPE_CHAT_SUBMIT_USER_MESSAGE, { text, attachments });
    setDraft('');
    setAttachments([]);
    inputRef.current?.focus();
  }, [draft, attachments]);

  const addAttachments = useCallback(async () => {
    const paths = await window.app.pickAttachments();
    if (paths && paths.length) {
      setAttachments(prev => Array.from(new Set([...prev, ...paths])));
    }
  }, []);

  // Paste-to-attach. Copied files (Explorer) attach by their real path, read from the OS
  // clipboard (the JS File objects don't expose paths). Raw image DATA (a screenshot, an
  // image copied from a browser) has no path at all — those bytes go to the backend,
  // which writes a temp file whose path then rides the normal attachment flow.
  const pasteAttachmentsFromClipboard = useCallback(async (imageFiles: File[] = []) => {
    const paths = await window.app.pasteAttachments();
    if (paths && paths.length) {
      setAttachments(prev => Array.from(new Set([...prev, ...paths])));
      return;
    }
    for (const f of imageFiles) {
      const b64 = await new Promise<string>((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result).split(',', 2)[1] || '');
        r.onerror = () => resolve('');
        r.readAsDataURL(f);
      });
      if (!b64) continue;
      const ext = (f.type.split('/')[1] || 'png').toLowerCase().replace('jpeg', 'jpg');
      const p = await window.app.storeClipboardImage(b64, ext);
      if (p) setAttachments(prev => Array.from(new Set([...prev, p])));
    }
  }, []);

  const removeAttachment = useCallback((path: string) => {
    setAttachments(prev => prev.filter(p => p !== path));
  }, []);

  const stopAi = useCallback(() => {
    // Single signal — Unity decides whether to cancel an in-flight stream (and
    // commit the partial as assistant history) or just stop the TTS pipeline.
    window.app.sendChat(TYPE_CHAT_STOP, {});
  }, []);

  const openReport = useCallback((reportId: string) => {
    if (!reportId) return;
    window.app.sendChat(TYPE_CHAT_OPEN_REPORT, { reportId });
  }, []);

  const rollbackTo = useCallback((turnIndex: number) => {
    if (turnIndex < 0) return;
    window.app.sendChat(TYPE_CHAT_ROLLBACK_TO_TURN, { turnIndex });
  }, []);

  const dismissBgTask = useCallback((taskId: string) => {
    // ✕ on a status-bar task chip: cancel the background task (helper or command).
    // The app pushes the updated roster back via Chat.BgTasks — no optimistic update.
    if (!taskId) return;
    window.app.sendChat(TYPE_CHAT_DISMISS_BG_TASK, { taskId });
  }, []);

  const editMessage = useCallback((arrayIndex: number, historyIndex: number, text: string,
                                   removedAttachments: number[], turnIndex: number) => {
    if (historyIndex < 0) return;
    // Optimistic local update — the app persists but doesn't echo back (except when the edit
    // removed attachments: deleting a hidden image row shifts history indices, so the app
    // re-pushes the full history and this optimistic entry gets replaced by the rebuild).
    dispatch({ kind: 'editEntry', payload: { index: arrayIndex, text, removedAttachments } });
    window.app.sendChat(TYPE_CHAT_EDIT_MESSAGE, { historyIndex, text, removedAttachments, turnIndex });
  }, []);

  const toggleMic = useCallback(() => {
    if (!state.voiceSupported || state.voiceBusy) return;
    // Hands-free repurposes this button: the mic is already open continuously, so
    // start/stop don't apply. Instead the button cancels an in-flight utterance —
    // clears whatever the recognizer has accumulated locally + clears the draft
    // before the endpoint fires and (with auto-submit) sends it to Unity.
    if (state.handsFree) {
      window.app.clearVoiceTranscript();
      setDraft('');
      preDictationRef.current = '';
      return;
    }
    if (state.voiceRecording) {
      window.app.stopRecording();
    } else {
      // Capture the current draft so the eventual transcript appends, not replaces.
      preDictationRef.current = draft;
      window.app.startRecording();
    }
  }, [state.voiceSupported, state.voiceBusy, state.voiceRecording, state.handsFree, draft]);

  const toggleVoiceMode = useCallback(() => {
    if (!state.ready) return;
    window.app.sendChat(TYPE_CHAT_SET_VOICE_MODE, { enabled: !state.voiceMode });
  }, [state.ready, state.voiceMode]);

  const toggleTouchMode = useCallback(() => {
    if (!state.ready) return;
    // Unity (AibuToucher) is the source of truth; it confirms/corrects via Chat.TouchModeChanged.
    window.app.sendChat(TYPE_CHAT_SET_TOUCH_MODE, { enabled: !state.touchMode });
  }, [state.ready, state.touchMode]);

  const toggleHeadTracking = useCallback(() => {
    if (!state.ready) return;
    // Python persists + echoes Chat.HeadTrackingChanged back; the toggle updates from that.
    window.app.sendChat(TYPE_CHAT_SET_HEAD_TRACKING, { enabled: !state.headTracking });
  }, [state.ready, state.headTracking]);

  const toggleEyeTracking = useCallback(() => {
    if (!state.ready) return;
    window.app.sendChat(TYPE_CHAT_SET_EYE_TRACKING, { enabled: !state.eyeTracking });
  }, [state.ready, state.eyeTracking]);

  const toggleHandsFree = useCallback(() => {
    if (!state.voiceSupported || state.voiceBusy) return;
    // Reset the dictation prefix when turning on; in hands-free we don't want
    // stale typed text getting concatenated onto an auto-submitted utterance.
    if (!state.handsFree) preDictationRef.current = '';
    window.app.setHandsFree(!state.handsFree);
  }, [state.voiceSupported, state.voiceBusy, state.handsFree]);

  const toggleAutoSubmit = useCallback(() => {
    if (!state.voiceSupported) return;
    window.app.setAutoSubmit(!state.autoSubmit);
  }, [state.voiceSupported, state.autoSubmit]);

  const toggleWakeWord = useCallback(() => {
    if (!state.voiceSupported || !state.wakeWordPhrase) return;
    window.app.setWakeWord(!state.wakeWordEnabled);
  }, [state.voiceSupported, state.wakeWordPhrase, state.wakeWordEnabled]);

  const startNewChat = useCallback((characterId: string) => {
    // Reachable from the home page where state.ready is still false (no session yet),
    // so this only guards against firing mid-reply. Opens the new-chat settings dialog;
    // the actual Chat.CreateNew fires on confirm (see confirmNewChat).
    if (state.typing || !characterId) return;
    setNewChatFor(characterId);
  }, [state.typing]);

  const confirmNewChat = useCallback((settings: ChatSettings) => {
    const characterId = newChatFor;
    setNewChatFor(null);
    setSidebarOpen(false);
    if (!characterId) return;
    const newSlot = `chat_${Date.now()}`;
    // Block the chat behind the loading overlay immediately, before Chat.Init flips the view — no
    // flash of interactive chat. The backend's Chat.Loading/Chat.Ready then drive it authoritatively.
    dispatch({ kind: 'chatLoadingStart' });
    window.app.sendChat(TYPE_CHAT_CREATE_NEW, { characterId, slot: newSlot, ...settings });
  }, [newChatFor]);

  const saveChatSettings = useCallback((settings: ChatSettings) => {
    setShowChatSettings(false);
    window.app.sendChat(TYPE_CHAT_UPDATE_SETTINGS, settings);
  }, []);

  // Settings and the character editor are both full-area `config-panel` overlays with the same
  // z-index, so opening one while the other is up would stack them (the later DOM node wins and the
  // new panel looks like it never opened). Opening Settings closes any open editor/dialog first.
  const openSettings = useCallback(() => {
    setShowCharCreation(false);
    setEditingCharacter(null);
    setNewChatFor(null);
    setShowChatSettings(false);
    setShowConfig(true);
  }, []);

  const editCharacter = useCallback((record: CharacterRecordWire) => {
    setEditingCharacter(record);
    setShowCharCreation(true);
  }, []);

  const deleteCharacter = useCallback(async (record: CharacterRecordWire, e?: React.MouseEvent) => {
    e?.stopPropagation();
    const label = record.displayName || record.name;
    const ok = await window.app.confirm({
      message: `Delete character “${label}”?`,
      detail: 'This removes the character. Existing chats with it are kept but can no longer be resumed.',
      confirmLabel: 'Delete',
      cancelLabel: 'Cancel',
    });
    if (!ok) return;
    window.app.sendChat(TYPE_CHAT_DELETE_CHARACTER, { id: record.id });
  }, []);

  const importCharacter = useCallback(() => {
    // Errors surface via the Chat.Error banner; success pushes a refreshed character list.
    void window.app.importCharacter();
  }, []);

  const confirmRebind = useCallback((characterId: string) => {
    if (!rebindFor) return;
    const slot = rebindFor.slot;
    setRebindFor(null);
    // Rebind ends in a resume, so put the loading overlay back up right away.
    dispatch({ kind: 'chatLoadingStart' });
    window.app.sendChat(TYPE_CHAT_REBIND_SAVE, { slot, characterId });
  }, [rebindFor]);

  // Files dropped onto the window (full paths, forwarded by the Python-side DOM drop
  // handler as App.FilesDropped). What a drop means depends on the active view: the home
  // page imports .wcc character bundles; the chat view attaches the files to the composer,
  // same as the attach button. Re-subscribed on view change so the closure stays current.
  useEffect(() => {
    const off = window.app.onChatEvent((env) => {
      if (env.type !== TYPE_APP_FILES_DROPPED) return;
      const paths = ((env as Envelope).payload as { paths?: string[] })?.paths ?? [];
      if (!paths.length) return;
      if (view === 'home') {
        for (const p of paths) {
          if (p.toLowerCase().endsWith('.wcc')) {
            window.app.sendChat(TYPE_CHAT_IMPORT_CHARACTER, { path: p });
          }
        }
      } else {
        setAttachments(prev => Array.from(new Set([...prev, ...paths])));
      }
    });
    return () => off();
  }, [view]);

  // "Drop to attach" overlay while a file is dragged over the chat view (the home page
  // has its own inside HomeScreen). Visual only — the drop lands on the Python handler.
  const [chatDragging, setChatDragging] = useState(false);
  useEffect(() => {
    if (view !== 'chat') {
      setChatDragging(false);
      return;
    }
    let depth = 0;
    const isFileDrag = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
    const enter = (e: DragEvent) => { if (!isFileDrag(e)) return; depth++; setChatDragging(true); };
    const leave = (e: DragEvent) => { if (!isFileDrag(e)) return; depth = Math.max(0, depth - 1); if (depth === 0) setChatDragging(false); };
    const end = () => { depth = 0; setChatDragging(false); };
    document.addEventListener('dragenter', enter);
    document.addEventListener('dragleave', leave);
    // Capture phase — pywebview's body drop handler stops propagation (see HomeScreen).
    document.addEventListener('drop', end, true);
    window.addEventListener('dragend', end);
    return () => {
      document.removeEventListener('dragenter', enter);
      document.removeEventListener('dragleave', leave);
      document.removeEventListener('drop', end, true);
      window.removeEventListener('dragend', end);
    };
  }, [view]);

  const exportCharacter = useCallback(async (record: CharacterRecordWire) => {
    const res = await window.app.exportCharacter(record.id, record.displayName || record.name);
    if (!res?.ok && res?.error) {
      dispatch({ kind: 'error', payload: { code: 'character_export', message: res.error } });
    }
  }, []);

  const loadChatSlot = useCallback((slot: string) => {
    if (state.typing) return;
    // Same optimistic gate as confirmNewChat — overlay up before Chat.Init switches to the chat view.
    dispatch({ kind: 'chatLoadingStart' });
    window.app.sendChat(TYPE_CHAT_LOAD_SAVE, { slot });
    setSidebarOpen(false);
  }, [state.typing]);

  const deleteChatSlot = useCallback(async (slot: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (state.typing) return;
    const ok = await window.app.confirm({
      message: 'Delete this chat session?',
      detail: 'This will delete the message history and all associated reports.',
      confirmLabel: 'Delete',
      cancelLabel: 'Cancel',
    });
    if (!ok) return;
    window.app.sendChat(TYPE_CHAT_DELETE_SAVE, { slot });
    // Deleting the chat you're currently in leaves no session to show — drop back
    // to the home page and clear the live-session state so "Resume current chat"
    // doesn't linger pointing at the now-deleted conversation.
    if (slot === state.activeSlot) {
      setSidebarOpen(false);
      setView('home');
      dispatch({ kind: 'clearSession' });
    }
  }, [state.typing, state.activeSlot]);

  const restartChat = useCallback(async () => {
    if (!state.ready || state.typing) return;
    const ok = await window.app.confirm({
      message: 'Restart the chat?',
      detail: 'The current conversation will be cleared.',
      confirmLabel: 'Restart',
      cancelLabel: 'Cancel',
    });
    if (!ok) return;
    window.app.sendChat(TYPE_CHAT_RESTART, {});
  }, [state.ready, state.typing]);

  const endSession = useCallback(() => {
    if (!state.ready || state.typing) return;
    // Tell the app to unload the model (Session.End) + drop the live session, then return
    // home and clear local session state so "Resume current chat" doesn't point at a dead session.
    // No confirm — the chat is saved and resumable, so this is non-destructive.
    window.app.sendChat(TYPE_CHAT_END_SESSION, {});
    setSidebarOpen(false);
    setView('home');
    dispatch({ kind: 'clearSession' });
  }, [state.ready, state.typing]);

  const openWorkspaceFolder = useCallback(async () => {
    const res = await window.app.openWorkspaceFolder();
    if (res && !res.ok) {
      dispatch({ kind: 'error', payload: { code: 'open_folder', message: res.error || 'Could not open the workspace folder.' } });
    }
  }, []);

  const placeholderText = !state.ready
    ? 'Waiting for Unity…'
    : state.history.length === 0
    ? 'No messages yet.'
    : '';

  // Disable input while a reply is in flight — the Unity-side OnPlayerSubmit would silently
  // drop it anyway.
  const inputDisabled = !state.ready || state.typing;

  // The sidebar lists only chats belonging to the character of the active session. We derive
  // the current character from the active slot's save metadata; if it can't be resolved yet
  // (e.g. a brand-new chat not yet persisted), fall back to showing all saves.
  const currentCharacterId = state.saves.find(s => s.slot === state.activeSlot)?.characterId;
  const currentCharacter = currentCharacterId
    ? state.characters.find(c => c.id === currentCharacterId)
    : undefined;
  // Outfit changing is only offered for KK models with something to switch between —
  // VRM models have no coordinates, and a single outfit leaves nothing to change into.
  const outfitChoices = currentCharacter?.coordinates ?? [];
  const canChangeOutfit = outfitChoices.length >= 2;
  const changeOutfit = useCallback((outfitIndex: number) => {
    setShowOutfitPicker(false);
    window.app.sendChat(TYPE_CHAT_CHANGE_OUTFIT, { outfitIndex });
  }, []);
  // Providers a character has a voice configured for — only these are offered in the dialog
  // (pocket needs an encoded embedding, elevenlabs needs a voice id). Empty → voice can't be on.
  const providersFor = (record?: CharacterRecordWire): string[] => {
    const v = record?.voices;
    if (!v) return [];
    const out: string[] = [];
    if (v.pocket?.embeddingFile) out.push('pocket');
    if (v.elevenlabs?.voiceId) out.push('elevenlabs');
    return out;
  };
  const visibleSaves = currentCharacterId
    ? state.saves.filter(s => s.characterId === currentCharacterId)
    : state.saves;

  const formatTime = (isoString?: string) => {
    if (!isoString) return '';
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  if (view === 'home') {
    return (
      <div className="app-shell">
        <Titlebar onOpenSettings={showConfig ? undefined : openSettings} settingsDisabled={!state.unityConnected} alwaysOnTop={alwaysOnTop} onToggleAlwaysOnTop={toggleAlwaysOnTop} />
        <div className="app-content">
          <HomeScreen
            characters={state.characters}
            saves={state.saves}
            activeSlot={state.activeSlot}
            chatReady={state.ready}
            error={state.error}
            onDismissError={() => dispatch({ kind: 'dismissError' })}
            onStartChat={startNewChat}
            onEditCharacter={editCharacter}
            onDeleteCharacter={deleteCharacter}
            onResume={loadChatSlot}
            onDelete={deleteChatSlot}
            onCreateCharacter={() => { setEditingCharacter(null); setShowCharCreation(true); }}
            onImportCharacter={importCharacter}
            onExportCharacter={exportCharacter}
            onResumeCurrent={state.ready ? () => setView('chat') : undefined}
            formatTime={formatTime}
          />
          {showConfig && <ConfigView onClose={() => setShowConfig(false)} />}
          {showCharCreation && (
            <CharacterEditor
              character={editingCharacter}
              onClose={() => { setShowCharCreation(false); setEditingCharacter(null); }}
              onSaved={(rec) => { dispatch({ kind: 'upsertCharacter', payload: rec }); setShowCharCreation(false); setEditingCharacter(null); }}
            />
          )}
        </div>
        {newChatFor && (
          <NewChatDialog
            mode="create"
            availableProviders={providersFor(state.characters.find(c => c.id === newChatFor))}
            onCancel={() => setNewChatFor(null)}
            onConfirm={confirmNewChat}
          />
        )}
        {rebindFor && (
          <RebindDialog
            characterName={rebindFor.characterName}
            characters={state.characters}
            onCancel={() => setRebindFor(null)}
            onConfirm={confirmRebind}
          />
        )}
        {!state.unityConnected && <ConnectionOverlay />}
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Titlebar onOpenSettings={showConfig ? undefined : openSettings} settingsDisabled={!state.unityConnected} alwaysOnTop={alwaysOnTop} onToggleAlwaysOnTop={toggleAlwaysOnTop} />
      {chatDragging && (
        <div className="home-drop-overlay">
          <div className="home-drop-box">
            <Paperclip size={28} />
            <span>Drop files to attach them to your message</span>
          </div>
        </div>
      )}
      <div className="app-content">
      <div className="chat-app">
      <div className="chat-main-layout" {...(showConfig ? { inert: '' } : {})}>
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <span className="sidebar-title">Saved Chats</span>
          <button
            className="new-chat-btn"
            onClick={() => currentCharacterId && startNewChat(currentCharacterId)}
            disabled={!state.ready || !currentCharacterId}
            title="Start a new chat with this character"
          >
            <Plus size={14} /> New Chat
          </button>
        </div>

        {/* Saved chats — caps at ~3 rows then scrolls (see .saves-list max-height). */}
        <div className="saves-list">
          {visibleSaves.map(save => {
            const isActive = state.activeSlot === save.slot;
            return (
              <div
                key={save.slot}
                className={`chat-item ${isActive ? 'active' : ''}`}
                onClick={() => !isActive && loadChatSlot(save.slot)}
              >
                <div className="chat-item-info">
                  <div className="chat-item-header">
                    <span className="chat-item-name">{save.characterName}</span>
                    <span className="chat-item-time">{formatTime(save.savedAtUtc)}</span>
                  </div>
                  <div className="chat-item-snippet">
                    {save.lastMessageText || 'New conversation'}
                  </div>
                </div>
                <button
                  className="delete-chat-btn"
                  onClick={(e) => deleteChatSlot(save.slot, e)}
                  title="Delete this chat"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })}
          {visibleSaves.length === 0 && (
            <div className="sidebar-empty">No saved chats.</div>
          )}
        </div>

        <div className="sidebar-separator" />

        {/* Per-chat tracking toggles (avatar looks at the camera). Saved with this chat. */}
        <div className="sidebar-toggles">
          <button
            className={`sidebar-toggle ${state.headTracking ? 'on' : 'off'}`}
            onClick={toggleHeadTracking}
            disabled={!state.ready}
            aria-pressed={state.headTracking}
            title="Head tracking — the character turns her head to follow the camera"
          >
            <ScanFace size={15} />
            <span className="sidebar-toggle-label">Head tracking</span>
            <span className="sidebar-toggle-state">{state.headTracking ? 'On' : 'Off'}</span>
          </button>
          <button
            className={`sidebar-toggle ${state.eyeTracking ? 'on' : 'off'}`}
            onClick={toggleEyeTracking}
            disabled={!state.ready}
            aria-pressed={state.eyeTracking}
            title="Eye tracking — the character's eyes follow the camera"
          >
            <Eye size={15} />
            <span className="sidebar-toggle-label">Eye tracking</span>
            <span className="sidebar-toggle-state">{state.eyeTracking ? 'On' : 'Off'}</span>
          </button>
        </div>

        <div className="sidebar-separator" />

        {/* Action grid — 3 per row. More options will be added here later. */}
        <div className="sidebar-actions">
          <button
            className="sidebar-action-btn"
            onClick={restartChat}
            disabled={!state.ready || state.typing}
            title="Restart chat — clears the current conversation"
          >
            <RotateCcw size={16} /><span>Restart</span>
          </button>
          <button
            className="sidebar-action-btn"
            onClick={openWorkspaceFolder}
            title="Open the workspace folder in your file explorer"
          >
            <FolderOpen size={16} /><span>Workspace</span>
          </button>
          <button
            className="sidebar-action-btn"
            onClick={() => { setShowChatSettings(true); setSidebarOpen(false); }}
            disabled={!state.ready}
            title="This chat's settings — name, voice, workspace folders"
          >
            <SlidersHorizontal size={16} /><span>Chat config</span>
          </button>
          <button
            className="sidebar-action-btn"
            onClick={() => { if (currentCharacter) { editCharacter(currentCharacter); setSidebarOpen(false); } }}
            disabled={!currentCharacter}
            title="Edit this character"
          >
            <Pencil size={16} /><span>Character</span>
          </button>
          {canChangeOutfit && (
            <button
              className="sidebar-action-btn"
              onClick={() => { setShowOutfitPicker(true); setSidebarOpen(false); }}
              disabled={!state.ready || state.typing}
              title="Change outfit — the character reacts to it"
            >
              <Shirt size={16} /><span>Outfit</span>
            </button>
          )}
          <button
            className="sidebar-action-btn"
            onClick={endSession}
            disabled={!state.ready || state.typing}
            title="End session — unloads the character and returns to home (the chat is saved)"
          >
            <LogOut size={16} /><span>End session</span>
          </button>
        </div>
      </div>
      <div className="chat-header">
        <div className="chat-header-left">
          <button
            className="menu-btn"
            onClick={() => setView('home')}
            title="Back to home"
            aria-label="Back to home"
          >
            <ArrowLeft size={18} />
          </button>
          {state.assistantAvatar && <img className="chat-header-avatar" src={state.assistantAvatar} alt="" />}
          <span className="chat-header-title">{state.assistantName}</span>
        </div>
        <div className="chat-header-right">
          <button
            className={`icon-toggle ${state.voiceMode ? 'on' : 'off'}`}
            onClick={toggleVoiceMode}
            disabled={!state.ready || !state.voiceAvailable}
            title={!state.voiceAvailable
              ? 'No voice set for this character — add one in the character editor'
              : (state.voiceMode ? 'Mute assistant voice' : 'Unmute assistant voice')}
            aria-label={state.voiceMode ? 'Mute assistant voice' : 'Unmute assistant voice'}
            aria-pressed={state.voiceMode}
          >
            {state.voiceMode ? <Volume2 size={14} /> : <VolumeX size={14} />}
          </button>
          <button
            className={`icon-toggle ${state.touchMode ? 'on' : 'off'}`}
            onClick={toggleTouchMode}
            disabled={!state.ready}
            title={state.touchMode ? 'Touch mode: on — click or middle mouse button to disable' : 'Touch mode: off — click or middle mouse button to caress the avatar'}
            aria-label={state.touchMode ? 'Disable touch mode' : 'Enable touch mode'}
            aria-pressed={state.touchMode}
          >
            <Hand size={14} />
          </button>
          {state.voiceSupported && (
            <>
              <button
                className={`icon-toggle ${state.handsFree ? 'on' : 'off'}`}
                onClick={toggleHandsFree}
                disabled={!state.ready || state.voiceBusy}
                title={state.handsFree ? 'Turn off hands-free voice' : 'Turn on hands-free voice (mic stays open)'}
                aria-label={state.handsFree ? 'Disable hands-free' : 'Enable hands-free'}
                aria-pressed={state.handsFree}
              >
                <Headphones size={14} />
              </button>
              {state.handsFree && (
                <button
                  className={`icon-toggle ${state.autoSubmit ? 'on' : 'off'}`}
                  onClick={toggleAutoSubmit}
                  disabled={!state.ready}
                  title={state.autoSubmit ? 'Auto-send each utterance: on' : 'Auto-send each utterance: off (fills draft only)'}
                  aria-label={state.autoSubmit ? 'Disable auto-send' : 'Enable auto-send'}
                  aria-pressed={state.autoSubmit}
                >
                  <Send size={14} />
                </button>
              )}
              {state.handsFree && state.autoSubmit && state.wakeWordPhrase && (
                <button
                  className={`icon-toggle ${state.wakeWordEnabled ? 'on' : 'off'} ${state.wakeWordArmExpiresAt > Date.now() ? 'armed' : ''}`}
                  onClick={toggleWakeWord}
                  disabled={!state.ready}
                  title={
                    state.wakeWordEnabled
                      ? `Wake-word required: "${state.wakeWordPhrase}" — click to disable`
                      : `Click to require wake phrase "${state.wakeWordPhrase}" before each command`
                  }
                  aria-label={state.wakeWordEnabled ? 'Disable wake word' : 'Enable wake word'}
                  aria-pressed={state.wakeWordEnabled}
                >
                  <Ear size={14} />
                </button>
              )}
            </>
          )}
          <button
            className="icon-btn"
            onClick={() => setSidebarOpen(true)}
            disabled={!state.ready}
            title="Previous chats"
            aria-label="Previous chats"
          >
            <Menu size={14} />
          </button>
        </div>
      </div>

      <div className="chat-scroll" ref={scrollRef} onScroll={onChatScroll}>
        {placeholderText && <div className="chat-empty">{placeholderText}</div>}

        {state.history.map((entry, idx) => (
          <MessageBubble
            key={`h-${idx}`}
            entry={entry}
            arrayIndex={idx}
            onOpenReport={openReport}
            onRollback={rollbackTo}
            onEdit={editMessage}
            editable={!state.typing}
          />
        ))}

        {(() => {
          // typing=true spans the whole turn (used to flip the Send button to Stop). The dots
          // show whenever a turn is in flight, no tool is running, and there's no fresh text:
          // either nothing has streamed yet, or streaming has stalled for STALL_MS (e.g. the
          // model is generating a tool call whose arguments stream invisibly — without this the
          // chat would just look frozen). A tool-activity label overrides the dots.
          const STALL_MS = 1000;
          const last = state.history[state.history.length - 1];
          const hasLiveContent = !!(last && last.role === 'assistant' && last.text);
          const stalled = Date.now() - lastProgressAtRef.current > STALL_MS;
          const showDots = state.typing && !state.toolActivity && (!hasLiveContent || stalled);
          if (!showDots && !state.toolActivity) return null;
          return (
            <div className="chat-status">
              {showDots && <span className="typing-dots"><span /><span /><span /></span>}
              {state.toolActivity && <span className="tool-activity">{state.toolActivity}</span>}
            </div>
          );
        })()}
      </div>

      {state.error && (
        <div className="error-banner">
          <span className="error-text">{state.error.message || 'Something went wrong.'}</span>
          <button
            className="error-dismiss"
            onClick={() => dispatch({ kind: 'dismissError' })}
            title="Dismiss"
            aria-label="Dismiss error"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <div className="composer">
        {state.voiceRecording && state.voicePartial && (
          /* Live caption while streaming. Mirrors what the recognizer thinks the user
           * is currently saying — useful in hands-free mode to spot mis-hears before
           * the utterance endpoints and (maybe) auto-submits. */
          <div className="voice-partial" aria-live="polite">{state.voicePartial}</div>
        )}
        {attachments.length > 0 && (
          <div className="composer-attachments">
            <AttachmentChips
              attachments={attachments.map(attachmentRefForPath)}
              onRemove={removeAttachment}
            />
          </div>
        )}
        <div className="composer-input-wrap">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder={inputDisabled && state.typing ? 'Waiting for reply…' : 'Type a message… (Shift+Enter for newline)'}
            value={draft}
            rows={1}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => {
              // Enter submits, Shift+Enter inserts a newline (textarea default).
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            onPaste={e => {
              // If the clipboard carries copied files or image data, attach instead of
              // pasting text. Image Files must be grabbed SYNCHRONOUSLY — clipboardData
              // is neutered once the handler returns.
              const cd = e.clipboardData;
              const imageFiles = Array.from(cd?.items ?? [])
                .filter(i => i.kind === 'file' && i.type.startsWith('image/'))
                .map(i => i.getAsFile())
                .filter((f): f is File => !!f);
              const hasFiles = !!cd && (cd.files?.length > 0 || Array.from(cd.types || []).includes('Files'));
              if (hasFiles || imageFiles.length) {
                e.preventDefault();
                void pasteAttachmentsFromClipboard(imageFiles);
              }
            }}
            disabled={inputDisabled}
          />
          {state.voiceSupported && (() => {
            // Hands-free: button is "cancel utterance" (clear draft + reset
            // recognizer). Disabled when there's nothing to clear.
            // Push-to-talk: button is start/stop, original behavior.
            const handsFreeNothingToClear = state.handsFree
              && !draft.trim()
              && !state.voicePartial.trim();
            const handsFreeMode = state.handsFree && !state.voiceBusy;
            return (
              <button
                className={`mic-btn ${state.voiceRecording ? 'recording' : ''} ${state.voiceBusy ? 'busy' : ''}`}
                onClick={toggleMic}
                disabled={inputDisabled || state.voiceBusy || (handsFreeMode && handsFreeNothingToClear)}
                title={
                  state.voiceBusy
                    ? (state.voiceBusyReason === 'loading'
                        ? 'Loading speech model…'
                        : 'Transcribing…')
                    : handsFreeMode ? 'Clear what was transcribed (cancel before send)'
                    : state.voiceRecording ? 'Stop recording'
                    : 'Start recording'
                }
                aria-label={
                  handsFreeMode ? 'Clear transcript'
                  : state.voiceRecording ? 'Stop recording'
                  : 'Start recording'
                }
              >
                {state.voiceBusy
                  ? <LoaderCircle size={14} className="spin" />
                  : handsFreeMode
                    ? <Eraser size={14} />
                    : state.voiceRecording
                      ? <Square size={12} fill="currentColor" />
                      : <Mic size={14} />}
              </button>
            );
          })()}
        </div>

        <div className="composer-controls">
          <div className="composer-controls-left">
            <button
              className="attach-btn"
              onClick={addAttachments}
              disabled={inputDisabled}
              title="Attach files — images are sent inline, other files are shared by path"
              aria-label="Attach files"
            >
              <Paperclip size={16} />
            </button>
          </div>
          {state.typing || state.speaking ? (
            // The assistant is doing something (generating or speaking) — let
            // the user interrupt. Unity decides which mode it's in and acts
            // accordingly. Button stays enabled while ready so the user can
            // always abort.
            <button
              className="send-btn stop"
              onClick={stopAi}
              disabled={!state.ready}
              title={state.typing ? 'Stop generation (keeps partial reply)' : 'Stop speaking'}
              aria-label="Stop"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={submit}
              disabled={inputDisabled || !draft.trim()}
              title={attachments.length > 0 && !draft.trim() ? 'Type a message to send the attachment' : 'Send (Enter)'}
              aria-label="Send"
            >
              <ArrowUp size={16} strokeWidth={2.5} />
            </button>
          )}
        </div>
      </div>

      <StatusBar
        voiceRecording={state.voiceRecording}
        voiceBusy={state.voiceBusy}
        voiceBusyReason={state.voiceBusyReason}
        bgTasks={state.bgTasks}
        onDismissTask={dismissBgTask}
      />
      </div>
      {showConfig && <ConfigView onClose={() => setShowConfig(false)} />}
      {newChatFor && (
        <NewChatDialog
          mode="create"
          availableProviders={providersFor(state.characters.find(c => c.id === newChatFor))}
          onCancel={() => setNewChatFor(null)}
          onConfirm={confirmNewChat}
        />
      )}
      {rebindFor && (
        <RebindDialog
          characterName={rebindFor.characterName}
          characters={state.characters}
          onCancel={() => setRebindFor(null)}
          onConfirm={confirmRebind}
        />
      )}
      {showOutfitPicker && currentCharacter && (
        <OutfitPickerDialog
          coordinates={outfitChoices}
          currentIndex={state.currentOutfitIndex}
          modelPath={currentCharacter.modelPath}
          onCancel={() => setShowOutfitPicker(false)}
          onConfirm={changeOutfit}
        />
      )}
      {showChatSettings && (
        <NewChatDialog
          mode="edit"
          availableProviders={providersFor(currentCharacter)}
          initial={{
            userName: state.userName,
            voiceMode: state.voiceMode,
            voiceProvider: state.voiceProvider,
            llmConfigId: state.llmConfigId,
            workspaceRoots: state.workspaceRoots,
          }}
          onCancel={() => setShowChatSettings(false)}
          onConfirm={saveChatSettings}
        />
      )}
      {showCharCreation && (
        <CharacterEditor
          character={editingCharacter}
          onClose={() => { setShowCharCreation(false); setEditingCharacter(null); }}
          onSaved={() => { setShowCharCreation(false); setEditingCharacter(null); }}
        />
      )}
      {/* Disconnected wins: if Unity is down, Session.Ready can't arrive anyway, so the
          reconnect overlay is the actionable one. Otherwise show the chat-loading overlay
          until the model + TTS + STT have all resolved. */}
      {!state.unityConnected
        ? <ConnectionOverlay />
        : state.chatLoading && <ChatLoadingOverlay status={state.chatLoadStatus} download={state.chatDownload} />}
      </div>
      </div>
    </div>
  );
}

/** Full-window overlay shown whenever the Unity backend WS is down. The app
 *  auto-reconnects with backoff (see app.py `_ws_loop`), so this is purely
 *  informational — it clears itself the moment the connection is re-established. */
