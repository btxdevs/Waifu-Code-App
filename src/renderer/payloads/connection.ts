// Unity WS connection-status event — pushed by the app to its own chat window so it
// can show the "reconnecting…" overlay. Mirrored in CompanionApp/python/app.py
// (EVT_UNITY_CONNECTION). Renderer-internal; never hits the WebSocket.

export interface UnityConnectionEvent {
  connected: boolean;
}
