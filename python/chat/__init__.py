"""Python port of the Unity-side chat orchestrator.

The C# implementation in Assets/Scripts/Chat/* is being moved here piece by piece. Each
module mirrors the file it replaces:

  models       — ChatMessage / ToolCall / ToolSchema / EmotionEntry (DTOs)
  llm_client   — Streaming HTTP client for OpenAI-compatible chat completions
  text         — EmotionStreamFilter + SentenceSplitter (pure text processing)
  speech       — SentenceSpeechPipeline (LLM stream -> sentences -> TTS)
  orchestrator — ChatOrchestrator (system prompt, history, tool loop)
  manager      — ChatManager + ChatUIController combined (renderer-facing glue)
  save_load    — ChatSaveData + SaveLoadManager (JSON on disk)
  config       — Parsing of llm.config.json (replaces BackendConfig SO)

Unity owns the avatar runtime; this package talks to it over the existing app WS
using the Avatar.* / Character.* / Session.* / Tool.* envelopes defined in
Assets/Scripts/App/AppProtocol.cs.
"""
