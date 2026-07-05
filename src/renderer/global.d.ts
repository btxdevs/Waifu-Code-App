export {};

import type { AppEvent } from './appEvents';
import type { Envelope } from './protocol';
import type { AppConfig } from './payloads/config';

declare global {
  interface Window {
    app: {
      /** Reply to a Unity task-window envelope (Chat-window's parent task). WS. */
      reply: (type: string, payload: unknown) => void;
      /** Send a Chat.* envelope to Unity over the WebSocket. Only use for types
       *  that genuinely round-trip to Unity (Chat.SubmitUserMessage,
       *  Chat.RollbackToTurn, Chat.SetVoiceMode, etc.) — anything UI ↔ Python
       *  local has its own dedicated method below. */
      sendChat: (type: string, payload: unknown) => void;
      /** Subscribe to events from the app. The stream multiplexes (a)
       *  Chat.* envelopes arriving from Unity over the WebSocket and (b)
       *  renderer-internal app events (`appEvents.ts`). Listeners
       *  receive both kinds with a string `type` discriminator. */
      onChatEvent: (listener: (env: Envelope | AppEvent) => void) => () => void;

      // ----- Renderer ↔ app-local methods (no Unity round-trip). -----
      notifyChatReady: () => void;
      startRecording: () => void;
      stopRecording: () => void;
      setHandsFree: (enabled: boolean) => void;
      setAutoSubmit: (enabled: boolean) => void;
      setWakeWord: (enabled: boolean, phrase?: string) => void;
      /** Drops whatever the streaming recognizer has accumulated for the
       *  current utterance + clears wake-word arm. */
      clearVoiceTranscript: () => void;

      // ----- Settings panel API. Reads/writes the two config files and lets the
      //       user pick a workspace folder via the OS dialog. -----
      getConfig: () => Promise<AppConfig>;
      saveConfig: (data: AppConfig) => Promise<{ ok: boolean; error?: string }>;
      pickDirectory: (current?: string) => Promise<string | null>;
      /** Open the OS file picker for a .vrm model. Returns the chosen path or null. */
      pickVrmFile: (current?: string) => Promise<string | null>;
      /** Open the OS file picker for a voice reference clip. Returns the chosen path or null. */
      pickAudioFile: (current?: string) => Promise<string | null>;
      /** Pick a character profile picture. Returns the PROCESSED image (square-cropped,
       *  ≤250px, white-flattened JPEG q90) as a data URL, or null on cancel/failure. */
      pickProfileImage: (current?: string) => Promise<string | null>;
      /** Export a character as a .wcc bundle via a save dialog. ok=false with no error = cancelled. */
      exportCharacter: (id: string, suggestedName?: string) => Promise<{ ok: boolean; path?: string; error?: string }>;
      /** Pick a .wcc bundle to import; the backend imports it and refreshes the character
       *  list (errors surface via the Chat.Error banner). Returns the path or null on cancel. */
      importCharacter: () => Promise<string | null>;
      /** Open the OS file picker (multi-select, any type) for message attachments. Returns absolute paths. */
      pickAttachments: () => Promise<string[]>;
      /** Read file paths currently on the OS clipboard (files copied in Explorer). For paste-to-attach. */
      pasteAttachments: () => Promise<string[]>;
      /** Persist pasted clipboard image DATA (screenshot / browser image — no file on disk)
       *  to a temp file; returns its path for the normal attachment flow, or null. */
      storeClipboardImage: (base64: string, ext?: string) => Promise<string | null>;
      /** Read a local image file as a base64 data URL (original bytes, no resize) for inline thumbnails. */
      readImageDataUrl: (path: string) => Promise<string | null>;
      /** Read an outfit screenshot from inside a .kkm model archive (filename from
       *  KK_Coordinates.json screenshots) as a base64 data URL. Null when missing. */
      readModelScreenshot: (modelPath: string, filename: string) => Promise<string | null>;
      /** Open the configured workspace folder in the OS file explorer. */
      openWorkspaceFolder: () => Promise<{ ok: boolean; error?: string }>;
      /** Launch the Unity player (disconnected-overlay button). No-op if already running.
       *  `launched` is false when it was already up. */
      openPlayer: () => Promise<{ ok: boolean; launched?: boolean; error?: string }>;

      // ----- Frameless-window controls (the renderer draws its own titlebar). -----
      /** Minimize the current window. */
      minimizeWindow: () => void;
      /** Maximize the current window (report windows). */
      maximizeWindow: () => void;
      /** Restore the current window from maximized. */
      restoreWindow: () => void;
      /** Toggle whether the current window stays above all others (chat window pin). */
      setAlwaysOnTop: (on: boolean) => void;
      /** Close the current window. Chat window → quits the app; task windows → close + default reply. */
      closeWindow: () => void;

      confirm: (opts: {
        message: string;
        detail?: string;
        confirmLabel?: string;
        cancelLabel?: string;
      }) => Promise<boolean>;
    };
  }
}
