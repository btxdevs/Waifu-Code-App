<p align="center">
  <img src="docs/waifu_code_logo.png" alt="Waifu Code" width="640">
</p>

<p align="center">
  A desktop AI companion with a 3D avatar — chat, voice, and a permission-gated coding agent, all running on your own machine.
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%2010%2F11-0078d6">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="React" src="https://img.shields.io/badge/react-18-61dafb">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

**Waifu Code** is the "brain + chat window" of a two-part desktop companion. It pairs with the **Waifu Code Player** — a separate Unity application that renders an animated 3D character (VRM or Koikatsu models) on your desktop. This app hosts the chat UI, talks to any OpenAI-compatible LLM, synthesizes and recognizes speech locally, remembers things long-term, and can act as a sandboxed coding assistant with per-action approval.

> The Unity player lives in its own project and is **not** part of this repository. This app connects to it over a local WebSocket (port `8770` by default) and can auto-launch it at startup.

## Features

**Chat & characters**
- Create and edit characters: persona definition, initial scenario, greeting, profile image, per-character voice.
- Multiple saved chats per character, with rollback to any turn, message editing, and restart.
- Markdown message rendering with streaming tokens, live tool-activity indicators, and image/file attachments (drag-drop or clipboard paste).
- Characters export/import as portable `.wcc` bundles (model, voice, and profile included).
- Emotion tags embedded in replies (`[Joy]`, `[Sadness]`, …) drive the avatar's facial expressions and animations in the Unity player.

**Voice**
- **Speech-to-text** — local streaming recognition via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (default model: NVIDIA Nemotron Streaming 0.6B). Push-to-talk or hands-free mode with VAD utterance segmentation, optional wake word, auto-submit, and TTS echo suppression.
- **Text-to-speech** — local synthesis via [pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx) with per-character voice cloning from a reference clip, or any ElevenLabs-compatible HTTP endpoint (e.g. a local chatterbox-tts-server).
- All speech models download automatically on first run — no manual setup.

**Avatar interaction** (through the Unity player)
- Touch mode — click/caress the avatar and the character reacts in-turn.
- Head and eye tracking toggles, outfit changes for multi-outfit characters, TTS lip-sync.

**Agent capabilities**
- The character can work as a coding/desktop assistant with a full tool belt: `Read`, `Write`, `Edit`, `Glob`, `Grep` (vendored ripgrep), `PowerShell`, `Open`, `TodoWrite`, `WebSearch` (keyless, via [ddgs](https://github.com/deedy5/ddgs)), `WebFetch` (local HTML→markdown), `Screenshot`, `LookAtYourself` (sees its own avatar; vision models), OCR fallback for non-vision models, and background sub-agents (researcher / explorer / general).
- Long/structured answers go to a **report modal** instead of being "spoken", keeping replies conversational.
- **Long-term memory** with two scopes — per-character and per-project (workspace folder) — using a two-tier recall system (an always-on index plus relevance-ranked full entries).

**Safety & sandboxing**
- Filesystem access is restricted to configured workspace roots; every write and shell command outside the allow-list pops an approval modal (*Allow once / Allow this session / Deny*).
- Dangerous command patterns (`rm -rf /`, `curl | sh`, `Invoke-Expression`, …) are always blocked.
- Everything is local: chats, characters, memories, and voice samples never leave your machine except for calls to the LLM endpoint you configure.

## Architecture

```
┌────────────────────────────┐         WebSocket          ┌──────────────────────┐
│  Waifu Code App (this repo)│  ws://127.0.0.1:8770       │  Waifu Code Player   │
│                            │ ◄────────────────────────► │  (separate Unity app)│
│  python/app.py (pywebview) │   chat / TTS audio / OCR   │  3D avatar: VRM & KK │
│   ├─ React UI (Vite build) │   emotions / touch / etc.  │  models, animations, │
│   ├─ LLM orchestrator       │                            │  lip-sync, touch     │
│   ├─ STT / TTS (ONNX)      │                            └──────────────────────┘
│   ├─ tools + approval gate │            HTTPS
│   └─ memory + characters   │ ◄────────────────────────► any OpenAI-compatible
└────────────────────────────┘     /v1/chat/completions       LLM endpoint
```

- The desktop host is **Python + [pywebview](https://pywebview.flowrl.com/)** (WebView2); the UI is **React 18 + TypeScript + Vite**. There is no Electron.
- The app and the Unity player find each other on a fixed loopback port — no tokens or pairing. Either can start first; the app shows a "Disconnected" overlay until the player is up, and can auto-start it (`unity.autostart` + `unity.exePath`).

## Getting started

### Prerequisites

- Windows 10/11 (the app uses WebView2, PowerShell, and Win32 APIs)
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/) (only needed to build the UI from source)
- An OpenAI-compatible LLM endpoint (OpenAI, DeepSeek, OpenRouter, a local server, …)
- A Waifu Code Player build for the 3D avatar (the app runs without it, but you'll be chatting with a disconnected overlay)

### Setup

```bat
git clone https://github.com/btxdevs/Waifu-Code-App.git
cd Waifu-Code-App

:: 1. Build the UI
npm install
npm run build

:: 2. Create the Python environment
cd python
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd ..
```

### Configure the LLM

Create `llm.config.json` next to `package.json`:

```json
{
  "configs": [
    {
      "id": "default",
      "name": "Default",
      "api_url": "https://api.deepseek.com/chat/completions",
      "api_key": "sk-...",
      "model": "deepseek-chat",
      "temperature": 1.0,
      "supports_vision": false
    }
  ],
  "default_id": "default"
}
```

Multiple named configs are supported and selectable per-chat from the UI. Optional per-config fields include `thinking` (`"enabled"` / `"disabled"`), `request_timeout_seconds`, `max_tool_call_rounds`, and vision limits — see [`python/chat/config.py`](python/chat/config.py) for the full schema.

### Run

```bat
run-app.bat
```

App settings live in `app.config.json` (created automatically when you change settings in the UI). To pre-configure it by hand, copy [`app.config.example.json`](app.config.example.json) — every key is documented inline. Highlights:

| Section | What it controls |
|---|---|
| `unity` | WebSocket port, auto-starting the player, player exe path |
| `stt` | STT model choice, hands-free mode, wake word, echo suppression |
| `tts` | `pocket` (local) vs `elevenlabs` provider and their settings |
| `workspace` | Allowed filesystem roots, shell allow/deny lists, sandbox caps |
| `webSearch` | Search engines, result counts, timeouts |

On first launch the speech models (~1 GB total for the default STT + TTS bundles) download into `models/`.

## Development

```bat
npm run dev
```

Then point the app at the Vite dev server instead of the built `dist/`:

```bat
set APP_DEV_URL=http://localhost:5173
run-app.bat
```

Renderer code is in [`src/renderer/`](src/renderer/), the Python backend in [`python/`](python/) (`app.py` is the entry point; the chat engine lives in [`python/chat/`](python/chat/)).

## Packaging a standalone build

```bat
build-app.bat
```

This builds the UI and runs PyInstaller (onedir), producing `build\pyi-dist\WaifuCodeApp\Waifu Code App.exe` with all resources staged beside it. Speech models are not bundled — they download on first run. Your `app.config.json` / `llm.config.json` are deliberately never copied into a build.

## Data & privacy

Everything is stored locally, next to the app:

| Folder | Contents |
|---|---|
| `characters/` | Character definitions, model copies, voice embeddings, profile images |
| `saves/` | Chat histories and their image attachments |
| `memory/` | Long-term memory (per-character and per-project JSON) |
| `models/` | Auto-downloaded STT/TTS/OCR models |
| `Logs/` | `app.log` |

The only network traffic is: your configured LLM endpoint, web search/fetch when the character uses those tools, and Hugging Face model downloads on first run.

> **Note:** character behavior is entirely user-defined. The default prompt template ([`system_prompt.txt`](system_prompt.txt)) is written for unrestricted adult roleplay — this project is intended for adults.

## License

[MIT](LICENSE) © btxdevs

Bundled third-party binaries keep their own licenses: [`python/vendor/ripgrep/`](python/vendor/ripgrep/) (ripgrep, MIT / Unlicense). Speech models downloaded at runtime are subject to their respective upstream licenses (sherpa-onnx model zoo, pocket-tts-onnx, RapidOCR).
