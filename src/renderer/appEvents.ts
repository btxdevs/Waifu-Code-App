// Renderer-internal events pushed from Python over the pywebview JS bridge
// (window.__chatPush → window.app.onChatEvent). These are NOT WebSocket
// envelopes — they never leave the app process — so they live here
// instead of protocol.ts. Type strings are mirrored in CompanionApp/python/stt.py
// (EVT_* constants); when one side changes the other has to follow.
//
// Why not just call typed methods on `window.app`? The renderer needs
// fan-out: many React components subscribe to "what did the voice subsystem
// just do". A single push channel with discriminated-union events is cheaper
// than maintaining per-event listener registries on both sides.

export const EVT_VOICE_RECORDING = 'Voice.Recording';
export const EVT_VOICE_BUSY = 'Voice.Busy';
export const EVT_VOICE_TRANSCRIPT = 'Voice.Transcript';
export const EVT_VOICE_PARTIAL = 'Voice.Partial';
export const EVT_HANDS_FREE_CHANGED = 'Voice.HandsFreeChanged';
export const EVT_AUTO_SUBMIT_CHANGED = 'Voice.AutoSubmitChanged';
export const EVT_WAKE_WORD_CHANGED = 'Voice.WakeWordChanged';
export const EVT_WAKE_WORD_ARMED = 'Voice.WakeWordArmed';

/** Pushed whenever the Unity WS connection comes up or drops. Mirrored in
 *  CompanionApp/python/app.py (EVT_UNITY_CONNECTION). Drives the
 *  "reconnecting…" overlay. */
export const EVT_UNITY_CONNECTION = 'Unity.Connection';

// Event payload interfaces live in ./payloads/{voice,connection}.ts — import them directly
// from there. This file holds the EVT_* constants and the AppEvent frame.

/** Wire shape: same envelope-style {id, type, payload} as protocol.ts, only
 *  because the JS bridge transport reuses the chatPush JSON format. Discriminate
 *  on `type` to extract the right payload interface. */
export interface AppEvent<P = unknown> {
  id: string;
  type: string;
  payload?: P;
}
