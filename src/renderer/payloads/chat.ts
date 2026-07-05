// Chat-window payloads: history entries + the streaming/init/saves/character-page envelopes the chat UI exchanges with Python.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

import type { CharacterRecordWire, KkCoordinate } from './character';

export interface ChatReportRef {
  id: string;
  title: string;
}

export type ChatTodoStatus = 'pending' | 'in_progress' | 'completed';

/** One attachment on a user message, rendered as a compact chip. For images `path` is the
 *  preprocessed saved copy (falls back to the original file); the chip shows a tiny thumbnail. */
export interface ChatAttachmentRef {
  name: string;
  path: string;
  kind: 'image' | 'file';
}

export interface ChatTodoEntry {
  content: string;
  activeForm: string;
  status: ChatTodoStatus;
}

/** `tool_activity` is a compact inline event between speech bubbles — e.g.
 *  "Wrote a report: Title", "Edited file: src/foo.cs". Emitted by the Python
 *  manager for every tool call the assistant made. `toolName` carries the
 *  underlying tool identifier so the renderer can pick an icon. The widgets
 *  (`reports`, `todos`) attach to the tool_activity entry whose tool produced
 *  them — e.g. the ReportWrite entry carries the "view report" button. */
export interface ChatHistoryEntry {
  /** `user_action` = something the user DID to the avatar (e.g. a touch/caress) — its own
   *  category, rendered on the right and kept separate from the assistant's `tool_activity` rows. */
  role: 'user' | 'assistant' | 'tool_activity' | 'user_action';
  speaker: string;
  text: string;
  canRollback: boolean;
  turnIndex: number;
  reports: ChatReportRef[];
  /** Snapshot of the todo list as of this turn (TodoWrite tool_activity rows). */
  todos?: ChatTodoEntry[] | null;
  /** Set only on `tool_activity` rows — the tool name (e.g. "ReportWrite", "Edit"). */
  toolName?: string;
  /** Attachments on this (user) message, rendered as one compact chip row. */
  attachments?: ChatAttachmentRef[];
  /** Index into the app's session.history this row was built from. Used to
   *  map an inline edit back to its source message. -1 for non-editable rows. */
  historyIndex?: number;
}

export interface ChatInitPayload {
  assistantDisplayName: string;
  /** The character's profile picture as a data URL ('' when none). */
  assistantAvatar?: string;
  userName: string;
  initialAssistantLine: string;
  history: ChatHistoryEntry[];
  voiceSupported: boolean;
  voiceMode: boolean;
  /** Whether the active character has a voice for the active TTS provider. When false,
   *  voice mode is forced off and the toggle is disabled (assistant lip-syncs to text). */
  voiceAvailable?: boolean;
  /** Active TTS provider for this chat ("pocket"/"elevenlabs") — used to prefill the
   *  per-chat settings dialog. */
  voiceProvider?: string;
  /** Whether the avatar caress (touch) mode is on. Resets off on each session (re)init. */
  touchMode?: boolean;
  /** Per-chat head/eye look-at tracking (avatar follows the camera). Default on. */
  headTracking?: boolean;
  eyeTracking?: boolean;
  /** Id of the LLM config this chat uses ("" = follow the default) — prefills the settings dialog. */
  llmConfigId?: string;
  /** Workspace folders in force for this chat (per-chat override, else the global roots). */
  workspaceRoots?: string[];
  activeSlot?: string;
  /** Worn outfit (stable KK coordinate id) so the outfit dialog can mark it. -1 for VRM. */
  currentOutfitIndex?: number;
}

/** Per-chat settings — chosen in the new-chat dialog, editable later from the sidebar.
 *  Sent on create (with Chat.CreateNew) and on edit (Chat.UpdateSettings). */
export interface ChatSettings {
  userName: string;
  voiceMode: boolean;
  voiceProvider: string;
  /** Id of the LLM config to use for this chat ("" = follow the default). */
  llmConfigId: string;
  workspaceRoots: string[];
}

export interface ChatHistoryPayload {
  entries: ChatHistoryEntry[];
}

export interface ChatPushEntryPayload {
  entry: ChatHistoryEntry;
}

export interface ChatAppendTokenPayload {
  /** Delta to append to the last history entry's text. Tags/emotion markup are
   *  already stripped on the producer side. Carrying just the delta (not the
   *  cumulative text) keeps streaming envelopes small. */
  delta: string;
}

export interface ChatTypingPayload {
  active: boolean;
}

export interface ChatSpeakingPayload {
  active: boolean;
}

export interface ChatToolActivityPayload {
  label: string;
}

export interface ChatEmotionPayload {
  label: string;
}

export interface ChatPlayerInputPayload {
  text: string;
  /** Absolute paths of files that were attached to the rolled-back message; re-offered in the composer. */
  attachments?: string[];
}

export interface ChatSubmitUserMessagePayload {
  text: string;
}

export interface ChatOpenReportPayload {
  reportId: string;
}

export interface ChatRollbackToTurnPayload {
  turnIndex: number;
}

export interface ChatErrorPayload {
  /** Short machine tag — e.g. "ai_error", "tool_error". */
  code: string;
  /** Human-readable detail. Shown to the user. */
  message: string;
}

export interface ChatSetVoiceModePayload {
  enabled: boolean;
}

export interface ChatSetTouchModePayload {
  enabled: boolean;
}

export interface ChatTouchModeChangedPayload {
  enabled: boolean;
}

/** The worn outfit changed — user pick, the AI's ChangeOutfit tool, or a rollback. */
export interface ChatOutfitChangedPayload {
  outfitIndex: number;
  outfitName: string;
}

/** One background task (UwU helper sub-agent, or a background shell command). */
export interface ChatBgTaskEntry {
  id: string;
  kind: string;    // agent_type ("researcher" / "explorer" / "general") or "shell"
  label: string;   // short user-facing description
  status: 'running' | 'completed' | 'failed' | 'dismissed';
  /** Wall-clock start (epoch ms) — the status-bar chip ticks elapsed time off this. */
  startedAt?: number;
  /** True once the task's report has been folded into a chat reaction turn. */
  announced?: boolean;
}

/** The background helper roster changed (summon / finish / dismiss / session swap). */
export interface ChatBgTasksPayload {
  tasks: ChatBgTaskEntry[];
}

export interface ChatSetHeadTrackingPayload {
  enabled: boolean;
}

export interface ChatHeadTrackingChangedPayload {
  enabled: boolean;
}

export interface ChatSetEyeTrackingPayload {
  enabled: boolean;
}

export interface ChatEyeTrackingChangedPayload {
  enabled: boolean;
}

export interface ChatVoiceModeChangedPayload {
  enabled: boolean;
  /** Whether the active character has a voice for the active provider (gates the toggle). */
  available?: boolean;
}

export interface ChatUserNameChangedPayload {
  userName: string;
}

export interface ChatSaveMetadata {
  slot: string;
  /** Stable id of the character this chat belongs to. */
  characterId: string;
  /** Display-name snapshot (labels the chat even if the character was deleted). */
  characterName: string;
  userName: string;
  savedAtUtc: string;
  lastMessageText: string;
  /** Per-chat workspace folders (file-tool roots for this chat). Empty = follows the global
   *  config roots. Shown in the home-page saves list and searchable. */
  workspaceRoots?: string[];
}

export interface ChatLoadSavePayload {
  slot: string;
}

export interface ChatCreateNewPayload {
  characterId: string;
  slot?: string;
  /** Per-chat settings from the new-chat dialog (defaults filled from the global config). */
  userName?: string;
  voiceMode?: boolean;
  voiceProvider?: string;
  llmConfigId?: string;
  workspaceRoots?: string[];
}

export interface ChatDeleteCharacterPayload {
  id: string;
}

export interface ChatDeleteSavePayload {
  slot: string;
}

export interface ChatSavesListPayload {
  slots: ChatSaveMetadata[];
}

export interface ChatCharactersListPayload {
  characters: CharacterRecordWire[];
}

export interface ChatSaveCharacterPayload {
  /** Present when editing an existing character; omit/empty to create a new one. */
  id?: string;
  name: string;
  displayName?: string;
  characterDefinition?: string;
  /** Processed profile picture as a data URL; '' removes the stored picture. */
  profileImage?: string;
  initialScenario?: string;
  initialAssistantMessage?: string;
  systemPromptTemplate?: string;
  initialEmotionLabel?: string;
  modelPath?: string;
  availableEmotions?: string[];
  coordinates?: KkCoordinate[];
  defaultOutfitIndex?: number;
}

export interface ChatInspectModelEmotionsPayload {
  modelPath: string;
}

export interface ChatModelEmotionsPayload {
  modelPath: string;
  emotions: string[];
  error?: string;
}

export interface ChatInspectModelCoordinatesPayload {
  modelPath: string;
}

export interface ChatModelCoordinatesPayload {
  modelPath: string;
  coordinates: KkCoordinate[];
  error?: string;
}
