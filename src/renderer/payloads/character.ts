// Character payloads: the full character record the picker/editor uses + the .vrm emotion-inspection request/result.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

/** ElevenLabs voice entry: just the opaque voice id the server identifies the voice by. */
export interface ElevenLabsVoiceEntry {
  voiceId: string;
}

/** Pocket-TTS voice entry. The editor sends `clipPath` (a freshly picked reference clip);
 *  the backend encodes it into an embedding and persists `clipHash` + `embeddingFile`
 *  (+ `clipName` for display). On a record loaded for editing, `clipPath` is absent and the
 *  stored fields describe the existing embedding. */
export interface PocketVoiceEntry {
  clipPath?: string;
  clipHash?: string;
  embeddingFile?: string;
  clipName?: string;
}

/** Per-provider TTS voices for a character — at most one entry per provider. The active TTS
 *  provider (global setting) decides which is used at synth time. Mirrors CharacterRecord.voices. */
export interface CharacterVoices {
  elevenlabs?: ElevenLabsVoiceEntry;
  pocket?: PocketVoiceEntry;
}

/** A KK character's outfit/coordinate, loaded from the KK_Coordinates.json inside a .kkm
 *  model. `name`/`description` are user-editable labels; `index` and `screenshots` (front/back
 *  filenames) are preserved as-is. Empty for VRM models. Mirrors CharacterRecord.coordinates. */
export interface KkCoordinate {
  index: number;
  name: string;
  description: string;
  screenshots?: { front?: string; back?: string };
}

/** Full character record — the picker lists these (id + display name) and the edit form
 *  prefills from the same object, so no extra fetch is needed. Mirrors CharacterRecord.to_wire. */
export interface CharacterRecordWire {
  id: string;
  name: string;
  displayName: string;
  characterDefinition: string;
  initialScenario: string;
  initialAssistantMessage: string;
  systemPromptTemplate: string;
  initialEmotionLabel: string;
  /** Profile picture as a data URL of the processed JPEG (square, ≤250px). '' = none.
   *  Stored as profile.jpg in the character's folder. */
  profileImage?: string;
  modelPath: string;
  /** The originally picked model file's name (e.g. "Tomoko.kkm") — what the editor shows;
   *  modelPath itself points at the app-owned copy. Mirrors the pocket voice's clipName. */
  modelName?: string;
  availableEmotions: string[];
  /** KK outfit/coordinate labels (empty for VRM models). */
  coordinates?: KkCoordinate[];
  /** Stable coordinate index of the outfit a NEW chat starts in (-1 = first outfit). */
  defaultOutfitIndex?: number;
  voices?: CharacterVoices;
  createdAtUtc?: string;
  updatedAtUtc?: string;
}

export interface CharacterInspectModelEmotionsPayload {
  /** Absolute path to the .vrm file to inspect. */
  modelPath: string;
}

export interface CharacterInspectModelEmotionsResultPayload {
  modelPath: string;
  /** Emotion labels the model supports (granular PerfectSync or coarse VRoid set). */
  emotions: string[];
  error?: string;
}
