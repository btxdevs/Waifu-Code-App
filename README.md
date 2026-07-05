<p align="center">
  <img src="docs/waifu_code_logo.png" alt="Waifu Code" width="640">
</p>

<p align="center">
  <b>Your AI companion, living on your desktop.</b><br>
  She chats, she codes, she reads manga with you, she remembers your birthday.<br>
  Speech, memory, and all your data stay on your machine — bring your own LLM (cloud or local). ♡
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows%2010%2F11-0078d6">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="React" src="https://img.shields.io/badge/react-18-61dafb">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Kawaii" src="https://img.shields.io/badge/kawaii-100%25-ff69b4">
</p>

---

**Waifu Code** puts an animated 3D character on your desktop — a VRM or Koikatsu model rendered by its companion Unity player — and gives her a brain, a voice, ears, eyes, and hands. She's not a chatbot in a box: she perks up when you talk, reacts when you poke her, blushes (or complains) when you pet her head, and can grab the keyboard and actually *do things* on your PC — with your permission, every time.

## ✨ What can you do together?

### 💬 Just hang out
Talk to her like a person. With **hands-free voice mode** the mic stays open — she hears you, answers out loud in her own voice, and lip-syncs while she says it. Set a **wake word** if you want her to only listen when called. Or go push-to-talk, or just type. Her replies carry emotion tags that drive her expressions and animations, so a `[Joy]` really *looks* like joy.

### 🎭 Roleplay anything
Every character is fully yours to define: persona, backstory, opening scenario, first message, voice. She stays in character — teasing, tsundere, soft-spoken, whatever you wrote. She reacts in-turn when you touch her (and can genuinely refuse if her character would), you can change her outfit mid-chat, and she'll keep her personality across every saved conversation. Rewind any chat to an earlier turn if the story took a wrong exit.

### 📖 Read manga & browse together
Give her a vision-capable model and she can **see your screen** — share the window your manga reader is in and talk about the chapter as you go, get raw pages OCR'd, or ask her opinion on the fight choreography. She can also search and read the web herself (no API keys needed) — "find out when the next volume drops" is a thing she can just… go do.

### 💻 Get actual work done
She's a real, permission-gated coding agent. She can read and edit files, search your codebase (bundled ripgrep!), run PowerShell commands, look things up online, and delegate research to background helper agents — all inside a filesystem sandbox where **every write and command needs your approval** (allow once / allow this session / deny). Long answers land in a tidy markdown report window instead of being read out loud for five minutes. Pair-programming with someone who never gets tired and occasionally calls you senpai.

### 🧠 She remembers you
Long-term memory with two scopes: things about *you and her* (your preferences, running jokes, that thing you told her last week) and things about *each project you work on together*. Relevant memories surface automatically in conversation — she brings them up herself when they matter.

### 🎨 Make her yours
Create characters from any **VRM** or **Koikatsu** model. Clone a voice for her from a short reference clip (fully local) or plug in an ElevenLabs-compatible endpoint. Multi-outfit Koikatsu characters get an outfit picker with screenshots. Export a finished character as a single `.wcc` file — model, voice, and portrait included — and share her with a friend.

### 🖥️ She lives *on* your desktop
The Unity player renders her in a transparent, always-on-top window — she stands beside your work, tracks you with her head and eyes, and reacts to being touched. The chat window is her other half; the two find each other automatically.

## 🛡️ Your data stays home

- Chats, characters, memories, and voice samples never leave your machine. The only network traffic is your chosen LLM endpoint, the web when *you* ask her to search it, and one-time model downloads.
- The coding agent is sandboxed to folders you allow, dangerous commands (`rm -rf /`, `curl | sh`, …) are hard-blocked, and everything else asks first.
- Bring any OpenAI-compatible LLM: OpenAI, DeepSeek, OpenRouter, or a local server like LM Studio / Ollama.

> 🔞 **A heads-up:** character behavior is entirely user-defined, and the default prompt template ([`system_prompt.txt`](system_prompt.txt)) is written for unrestricted adult roleplay. This project is intended for adults.

---

# 🔧 The technical stuff

**Waifu Code** is the "brain + chat window" half of a two-part system. The **Waifu Code Player** (a separate Unity project, not in this repository) renders the 3D avatar; this app hosts the chat UI, LLM orchestration, speech, memory, and agent tools, and talks to the player over a local WebSocket.

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

- Desktop host: **Python + [pywebview](https://pywebview.flowrl.com/)** (WebView2); UI: **React 18 + TypeScript + Vite**. No Electron.
- **STT**: local streaming recognition via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (default: NVIDIA Nemotron Streaming 0.6B) with Silero VAD for hands-free segmentation, wake word, and TTS echo suppression.
- **TTS**: local [pocket-tts-onnx](https://huggingface.co/KevinAHM/pocket-tts-onnx) with per-character voice cloning, or any ElevenLabs-compatible HTTP endpoint. Speech models auto-download on first run.
- **Agent tools**: `Read` / `Write` / `Edit` / `Glob` / `Grep` (vendored ripgrep) / `PowerShell` / `Open` / `TodoWrite` / `WebSearch` ([ddgs](https://github.com/deedy5/ddgs), keyless) / `WebFetch` (local HTML→markdown) / `Screenshot` / `LookAtYourself` / OCR fallback for non-vision models / background sub-agents (researcher, explorer, general).
- **Memory**: two-tier recall (always-on index + relevance-ranked entries), per-character and per-project scopes.
- The app and player pair over a fixed loopback port (`8770`) — no tokens. Either side can start first; the app can auto-launch the player (`unity.autostart` + `unity.exePath`).

## Getting started

**Prerequisites:** Windows 10/11 · [Python 3.10+](https://www.python.org/downloads/) · [Node.js 18+](https://nodejs.org/) (UI build only) · an OpenAI-compatible LLM endpoint · a Waifu Code Player build (the app runs without it, but she won't have a body).

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

**Configure the LLM** — create `llm.config.json` next to `package.json`:

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

Multiple named configs are supported and selectable per-chat. Optional fields include `thinking`, `request_timeout_seconds`, `max_tool_call_rounds`, and vision limits — see [`python/chat/config.py`](python/chat/config.py) for the full schema.

**Run:**

```bat
run-app.bat
```

App settings live in `app.config.json` (written automatically when you change settings in the UI; copy [`app.config.example.json`](app.config.example.json) to pre-configure by hand — every key is documented inline):

| Section | What it controls |
|---|---|
| `unity` | WebSocket port, auto-starting the player, player exe path |
| `stt` | STT model choice, hands-free mode, wake word, echo suppression |
| `tts` | `pocket` (local) vs `elevenlabs` provider and their settings |
| `workspace` | Allowed filesystem roots, shell allow/deny lists, sandbox caps |
| `webSearch` | Search engines, result counts, timeouts |

On first launch the speech models (~1 GB for the default STT + TTS bundles) download into `models/`.

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

Builds the UI and runs PyInstaller (onedir) → `build\pyi-dist\WaifuCodeApp\Waifu Code App.exe` with all resources staged beside it. Speech models are not bundled (they download on first run), and your `app.config.json` / `llm.config.json` are deliberately never copied into a build.

## Where your data lives

| Folder | Contents |
|---|---|
| `characters/` | Character definitions, model copies, voice embeddings, portraits |
| `saves/` | Chat histories and their image attachments |
| `memory/` | Long-term memory (per-character and per-project JSON) |
| `models/` | Auto-downloaded STT/TTS/OCR models |
| `Logs/` | `app.log` |

## License

[MIT](LICENSE) © btxdevs

Bundled third-party binaries keep their own licenses: [`python/vendor/ripgrep/`](python/vendor/ripgrep/) (ripgrep, MIT / Unlicense). Speech models downloaded at runtime are subject to their respective upstream licenses (sherpa-onnx model zoo, pocket-tts-onnx, RapidOCR).
