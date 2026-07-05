// Voice / STT event payloads — renderer-internal events the app pushes over the
// pywebview bridge (window.__chatPush → onChatEvent). NOT WebSocket envelopes; they never
// reach Unity. Mirrored in CompanionApp/python/stt.py (EVT_* constants).

export interface VoiceRecordingEvent {
  active: boolean;
}

export interface VoiceBusyEvent {
  busy: boolean;
  /** Only set while busy=true. Tells the status bar what the system is actually doing
   * — model download/init vs running inference on captured audio. */
  reason?: 'loading' | 'transcribing';
}

export interface VoiceTranscriptEvent {
  text: string;
  /** True iff the app already sent this transcript on as Chat.SubmitUserMessage
   * (hands-free + autoSubmit). Renderer uses it to skip appending to the draft. */
  submitted?: boolean;
}

export interface VoicePartialEvent {
  text: string;
}

export interface HandsFreeChangedEvent {
  enabled: boolean;
}

export interface AutoSubmitChangedEvent {
  enabled: boolean;
}

export interface WakeWordChangedEvent {
  enabled: boolean;
  phrase: string;
  armSeconds: number;
}

export interface WakeWordArmedEvent {
  armed: boolean;
  /** Seconds until the arm window expires, when armed=true; 0 otherwise. */
  expiresInSeconds: number;
}
