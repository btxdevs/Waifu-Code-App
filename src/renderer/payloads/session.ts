// Session-lifecycle payloads (Python → Unity): begin/end of a chat session, carrying the avatar's initial state + model path.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

export interface SessionBeginPayload {
  characterName: string;
  userName: string;
  outfitName: string;
  emotionLabel: string;
  status: string;
  voiceMode: boolean;
  /** Absolute path to the character's .vrm model; Unity loads + binds it at session start. */
  modelPath?: string;
}

export interface SessionEndPayload {}
