// Settings-panel config blocks: the subset of llm.config.json + app.config.json the UI reads/writes.
// Mirrors the matching payloads in Assets/Scripts/App/AppProtocol.cs.

export interface LlmConfigBlock {
  /** Stable id referenced per-chat (in ChatSettings.llmConfigId). */
  id: string;
  /** Human label shown in the settings list and the chat pickers. */
  name: string;
  api_url: string;
  api_key: string;
  model: string;
  temperature: number;
  request_timeout_seconds: number;
  /** "unset" | "disabled" | "enabled" — DeepSeek reasoner-style chain-of-thought toggle. */
  thinking: string;
  send_system_prompt_as_user: boolean;
  supports_vision: boolean;
  /** 0 = uncapped. */
  max_tool_call_rounds: number;
  /** Most-recent image-bearing messages kept per origin (user vs tool) when building the LLM
   *  request. 0 = send no images. Only matters when supports_vision is on. */
  vision_max_images: number;
}

/** The set of named LLM configs plus which one is the default. A chat references a config by id;
 *  an empty/unknown id falls back to `defaultId`. */
export interface LlmConfigSet {
  configs: LlmConfigBlock[];
  defaultId: string;
}

export interface WorkspaceConfigBlock {
  /** Absolute paths the assistant is allowed to read/write under. First entry
   *  doubles as the cwd for relative paths. */
  allowedRoots: string[];
  /** Bash/PowerShell prefixes that bypass the approval modal. */
  allowedCommandPrefixes: string[];
  /** Bash/PowerShell prefixes that are refused outright. */
  deniedCommandPrefixes: string[];
  /** Full-permission mode: skip every approval prompt and let file tools reach paths
   *  outside the configured roots. Off by default. */
  fullAccess: boolean;
}

/** ElevenLabs-compatible HTTP endpoint settings (e.g. the chatterbox-tts-server).
 *  Only consumed when `tts.provider === 'elevenlabs'`. */
export interface ElevenLabsConfigBlock {
  baseUrl: string;
  apiKey: string;
  model: string;
  stability: number;
  similarityBoost: number;
  useSpeakerBoost: boolean;
  speed: number;
  requestTimeoutSeconds: number;
}

/** Local pocket-tts-onnx engine settings. Only consumed when `tts.provider === 'pocket'`. */
export interface PocketTtsConfigBlock {
  /** "int8" (quantized, faster) | "fp32" (full precision, best quality). */
  precision: string;
  /** Flow refinement steps per audio frame (1–4, default 3). More steps = cleaner speech,
   *  more CPU — past 4 the gain is negligible for a model this small. */
  lsdSteps: number;
}

export interface TtsConfigBlock {
  /** "pocket" (local pocket-tts-onnx, default) | "elevenlabs". */
  provider: string;
  pocket: PocketTtsConfigBlock;
  elevenlabs: ElevenLabsConfigBlock;
}

/** App/window UI preferences not tied to a specific subsystem. */
export interface UiConfigBlock {
  /** Whether the chat window floats above all others (the titlebar pin toggle). */
  alwaysOnTop: boolean;
}

export interface AppConfig {
  llm: LlmConfigSet;
  user_name: string;
  workspace: WorkspaceConfigBlock;
  tts: TtsConfigBlock;
  ui: UiConfigBlock;
}
