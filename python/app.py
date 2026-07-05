"""
Python app for the Unity chat character. Replaces the old Electron host
(see git history / src/main/ before removal) with a pywebview + websocket-client setup.

Architecture:
  - Unity's AppBridge listens on a fixed loopback WS port (app.config.json →
    unity.port, default 8770; mirrored by AppBridge.port in Unity). We dial
    ws://127.0.0.1:<port>/ and retry until it's up. Optionally we autostart the Unity build
    first (unity.autostart + unity.exePath). No tokens / connection files — local port only.
  - One pywebview window per task envelope (ShowReport / AskQuestion / RequestPermission).
  - One persistent chat window opened by the OpenChatWindow envelope; survives close
    (intercept → minimize) until the process exits.
  - The React renderer is unchanged. We inject a small JS shim at page load that
    rebuilds the `window.app.*` surface the renderer expects, forwarding through
    pywebview's `window.pywebview.api.*` bridge.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import webview
import websocket  # websocket-client

import image_proc  # local — Pillow-based image transcode/resize for read_image, see image_proc.py
import ocr  # local — RapidOCR for the read_image_ocr tool, see ocr.py
import stt  # local — speech-to-text + mic capture, see stt.py
import tts  # local — text-to-speech (pocket-tts / ElevenLabs), see tts.py
from chat import config as chat_config
from chat.app_paths import APP_ROOT  # base dir for resources + user data (frozen-aware)
from chat.manager import ChatManager
from chat.save_load import SaveLoadManager


# ----------- logging: redirect stdout/stderr to a file -----------
# Unity launches us with CreateNoWindow=true, so there's no console attached. Without this
# redirect, every print() and traceback would vanish. The log lives in APP_ROOT/Logs/ so it
# sits alongside the Unity player's Player.log (same Logs/ folder in a packaged release).
_LOG_PATH = APP_ROOT / 'Logs' / 'app.log'

class _TimestampedStream:
    """Tee writes through a per-line timestamp prefix. Flushes after each line so a crash
    doesn't lose the last few messages."""
    def __init__(self, sink, tag: str):
        self._sink = sink
        self._tag = tag
        self._buf = ''
    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            self._sink.write(f'{ts} [{self._tag}] {line}\n')
            self._sink.flush()
        return len(s)
    def flush(self) -> None:
        if self._buf:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            self._sink.write(f'{ts} [{self._tag}] {self._buf}')
            self._buf = ''
        self._sink.flush()

def _setup_logging() -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)        # ensure Logs/ exists
        fh = open(_LOG_PATH, 'w', encoding='utf-8', buffering=1)  # line-buffered
    except OSError as e:
        # If we can't open the log file, give up on redirection silently — the process
        # still runs, we just lose diagnostics.
        sys.__stderr__.write(f'[CompanionApp] Failed to open log {_LOG_PATH}: {e}\n')
        return
    sys.stdout = _TimestampedStream(fh, 'out')
    sys.stderr = _TimestampedStream(fh, 'err')
    fh.write(f'=== app.py started {datetime.now().isoformat()} pid={os.getpid()} ===\n')
    fh.flush()

_setup_logging()

# ----------- taskbar identity -----------
# Share one explicit AppUserModelID with the Unity player (which tags its own window with the
# SAME id). Windows then groups both taskbar buttons under a single app identity instead of
# showing the chat window and the avatar window as two unrelated programs. Must run before the
# pywebview window is created. The string MUST match AUMID in WC_Player WindowsAppUserModelId.cs.
APP_USER_MODEL_ID = 'WaifuCode'

def _set_app_user_model_id() -> None:
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception as e:
        sys.stderr.write(f'[CompanionApp] SetAppUserModelID failed: {e}\n')

_set_app_user_model_id()

# ----------- paths + HF cache redirect -----------
# This block has to run BEFORE huggingface_hub (used by stt.py to fetch sherpa-onnx
# model bundles) is touched, because HF_HOME / HF_HUB_CACHE are read at import time.
# Otherwise model files land in ~/.cache/huggingface/ instead of CompanionApp/models/.

# Vite build output. `npm run build` writes dist/index.html alongside hashed assets;
# the packaged build copies dist/ next to the .exe (APP_ROOT).
DIST_INDEX = APP_ROOT / 'dist' / 'index.html'
# Dev: point at the vite dev server. `npm run dev` listens on 5173 by default.
DEV_URL = os.environ.get('APP_DEV_URL')

# Local model cache so the sherpa-onnx STT bundle + silero_vad.onnx live next to the app instead of
# in the user's home directory — easier to inspect, ship, or wipe. Uses `setdefault` so
# callers can still override by setting HF_HOME / HF_HUB_CACHE in the env before launch.
MODELS_DIR = APP_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(MODELS_DIR))
os.environ.setdefault('HF_HUB_CACHE', str(MODELS_DIR))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', 'true')

# TTS reference voices. Each character carries its own per-provider voice (see
# CharacterRecord.voices); pocket voices are encoded from a reference clip into the
# character's folder (`characters/<charId>/voice_<hash>.npy`) at save time.
VOICES_DIR = APP_ROOT / 'voices'
VOICES_DIR.mkdir(parents=True, exist_ok=True)

# ----------- protocol constants (mirrors C# AppProtocol) -----------

PROTOCOL_VERSION = 1

TYPE_HELLO = 'Hello'
TYPE_SHOW_REPORT = 'ShowReport'
TYPE_ASK_QUESTION = 'AskQuestion'
TYPE_REQUEST_PERMISSION = 'RequestPermission'
TYPE_OPEN_CHAT_WINDOW = 'OpenChatWindow'

TYPE_CLIENT_READY = 'ClientReady'
TYPE_QUESTION_ANSWER = 'QuestionAnswer'
TYPE_PERMISSION_DECISION = 'PermissionDecision'
TYPE_REPORT_CLOSED = 'ReportClosed'

INTERACTIVE_TYPES = {TYPE_SHOW_REPORT, TYPE_ASK_QUESTION, TYPE_REQUEST_PERMISSION}
CHAT_TYPE_PREFIX = 'Chat.'

# Renderer-internal event (pushed over the JS bridge, never hits the WS): tells the chat
# window whether the Unity backend is currently connected, so it can overlay a
# "reconnecting…" message. Mirrored in src/renderer/appEvents.ts (EVT_UNITY_CONNECTION).
EVT_UNITY_CONNECTION = 'Unity.Connection'

# pywebview 5.x replaced the module-level dialog constants (OPEN_DIALOG / FOLDER_DIALOG) with
# a FileDialog enum; the old constants are deprecated. Prefer the enum, fall back for older
# pywebview so the venv version doesn't matter.
try:
    FD_OPEN = webview.FileDialog.OPEN
    FD_FOLDER = webview.FileDialog.FOLDER
    FD_SAVE = webview.FileDialog.SAVE
except AttributeError:  # pragma: no cover - older pywebview
    FD_OPEN = webview.OPEN_DIALOG
    FD_FOLDER = webview.FOLDER_DIALOG
    FD_SAVE = webview.SAVE_DIALOG

# ----------- app.config.json -----------
# Settings live in CompanionApp/app.config.json so both Unity and Python can read
# a single source of truth. Currently only `stt.*` keys are consumed (by stt.py); the
# loader is here so future non-STT settings can share it.

_CONFIG_PATH = APP_ROOT / 'app.config.json'


def _load_app_config() -> dict:
    """Read app.config.json. Returns an empty dict on any failure — settings then
    fall through to per-consumer defaults. Never raises; misconfigured JSON shouldn't
    prevent the chat window from opening."""
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f'[config] {_CONFIG_PATH} root must be an object, got {type(data).__name__}', file=sys.stderr)
            return {}
        return data
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f'[config] failed to read {_CONFIG_PATH}: {e}', file=sys.stderr)
        return {}


_CONFIG = _load_app_config()


def _persist_ui_always_on_top(value: bool) -> None:
    """Persist `ui.alwaysOnTop` to app.config.json and the in-memory _CONFIG so the
    chat window restores its pin state on next launch (install_chat_window reads it). Re-reads
    the file first so we don't clobber keys the settings panel owns. Best-effort — never raises;
    a failed write just means the preference isn't remembered."""
    try:
        existing: dict = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                existing = raw
        ui = existing.get('ui') if isinstance(existing.get('ui'), dict) else {}
        ui['alwaysOnTop'] = bool(value)
        existing['ui'] = ui
        with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        # Keep the live config in sync (read at window creation).
        live_ui = _CONFIG.get('ui') if isinstance(_CONFIG.get('ui'), dict) else {}
        live_ui['alwaysOnTop'] = bool(value)
        _CONFIG['ui'] = live_ui
    except (OSError, json.JSONDecodeError) as e:
        print(f'[config] persist ui.alwaysOnTop failed: {e}', file=sys.stderr)


# Loopback WS port Unity's AppBridge listens on. Overridable via app.config.json
# (unity.port); must match AppBridge.port on the Unity side.
_DEFAULT_UNITY_PORT = 8770

# The Unity player process WE launched — via autostart at boot or the disconnected-overlay
# "Open Player" button. Resolved exe path + port are cached so the button can relaunch. We only
# ever track a player WE spawned; a player the user started independently (port already in use)
# is never tracked or killed. Whatever we launch is closed when the app closes, regardless of
# the autostart setting — promptly via close_player() on a graceful exit, and as a hard backstop
# via the kill-on-close Job Object (`_unity_job`) for crashes / force-kills.
_unity_proc = None              # type: ignore[assignment]  # subprocess.Popen | None
_unity_exe_path = ''
_unity_port = _DEFAULT_UNITY_PORT
_unity_job = None               # Win32 Job Object handle (kill-on-close); see _ensure_unity_job

# App-side LLM config (DeepSeek / OpenAI-compatible). Lives alongside
# app.config.json so the API key has its own file with separate ownership.
_LLM_CONFIG_PATH = APP_ROOT / 'llm.config.json'


def _read_clipboard_file_paths() -> list:
    """Return absolute paths of files currently on the clipboard (Windows CF_HDROP — i.e.
    files copied in Explorer). Returns [] on non-Windows or when no files are present.
    Used so pasting copied files into the composer attaches them by path."""
    if not sys.platform.startswith('win'):
        return []
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []
    CF_HDROP = 15
    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32
    # restype matters on 64-bit: the default c_int truncates the HANDLE pointer.
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    paths: list[str] = []
    if not user32.OpenClipboard(0):
        return []
    try:
        if not user32.IsClipboardFormatAvailable(CF_HDROP):
            return []
        handle = user32.GetClipboardData(CF_HDROP)
        if not handle:
            return []
        count = shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
        for i in range(count):
            need = shell32.DragQueryFileW(handle, i, None, 0)
            buf = ctypes.create_unicode_buffer(need + 1)
            shell32.DragQueryFileW(handle, i, buf, need + 1)
            if buf.value:
                paths.append(buf.value)
    finally:
        user32.CloseClipboard()
    return [p.replace('\\', '/') for p in paths]


# Keys (besides id/name) the settings panel reads/writes per LLM config. Kept as a single
# list so _read_settings (defaulting) and _write_settings (merge) stay in lockstep.
_LLM_WIRE_KEYS = (
    'api_url', 'api_key', 'model', 'temperature', 'request_timeout_seconds',
    'thinking', 'send_system_prompt_as_user', 'supports_vision', 'max_tool_call_rounds',
    'vision_max_images',
)


def _llm_entry_to_wire(entry) -> dict:
    """Serialize one chat_config.LlmConfigEntry to the per-config wire shape the UI edits
    (id + name + the editable LLM fields)."""
    cfg = entry.config
    return {
        'id': entry.id,
        'name': entry.name,
        'api_url': str(cfg.llm.api_url or ''),
        'api_key': str(cfg.llm.api_key or ''),
        'model': str(cfg.llm.model or ''),
        'temperature': cfg.llm.temperature if isinstance(cfg.llm.temperature, (int, float)) else 1,
        'request_timeout_seconds': cfg.llm.request_timeout_seconds
            if isinstance(cfg.llm.request_timeout_seconds, int) else 30,
        'thinking': str(cfg.llm.thinking or 'unset'),
        'send_system_prompt_as_user': bool(cfg.llm.send_system_prompt_as_user),
        'supports_vision': bool(cfg.llm.supports_vision),
        'max_tool_call_rounds': cfg.max_tool_call_rounds if isinstance(cfg.max_tool_call_rounds, int) else 0,
        'vision_max_images': cfg.llm.vision_max_images if isinstance(cfg.llm.vision_max_images, int) else 10,
    }


def _read_settings() -> dict:
    """Snapshot the two config files for the settings panel. Returns the subset
    the UI cares about (LLM config set + user_name + workspace + tts). Missing
    files / malformed JSON degrade to defaults — never raises."""
    app = _load_app_config()
    # The LLM block is now a SET of named configs + a selected default. load_registry handles
    # both the new schema and a legacy flat file (migrated to a single "Default" config).
    registry = chat_config.load_registry(_LLM_CONFIG_PATH)

    workspace = app.get('workspace') if isinstance(app.get('workspace'), dict) else {}
    tts = app.get('tts') if isinstance(app.get('tts'), dict) else {}
    eleven = tts.get('elevenlabs') if isinstance(tts.get('elevenlabs'), dict) else {}
    ui = app.get('ui') if isinstance(app.get('ui'), dict) else {}

    def _num(v, default):
        return v if isinstance(v, (int, float)) else default

    return {
        'llm': {
            'configs': [_llm_entry_to_wire(e) for e in registry.entries],
            'defaultId': registry.default_id,
        },
        'user_name': str(app.get('user_name') or ''),
        'workspace': {
            'allowedRoots': [r for r in (workspace.get('allowedRoots') or []) if isinstance(r, str)],
            'allowedCommandPrefixes': [s for s in (workspace.get('allowedCommandPrefixes') or []) if isinstance(s, str)],
            'deniedCommandPrefixes': [s for s in (workspace.get('deniedCommandPrefixes') or []) if isinstance(s, str)],
            'fullAccess': bool(workspace.get('fullAccess', False)),
        },
        'tts': {
            'provider': str(tts.get('provider') or 'pocket'),
            'elevenlabs': {
                'baseUrl': str(eleven.get('baseUrl') or ''),
                'apiKey': str(eleven.get('apiKey') or ''),
                'model': str(eleven.get('model') or ''),
                'stability': _num(eleven.get('stability'), 0.5),
                'similarityBoost': _num(eleven.get('similarityBoost'), 0.75),
                'useSpeakerBoost': bool(eleven.get('useSpeakerBoost', True)),
                'speed': _num(eleven.get('speed'), 1.0),
                'requestTimeoutSeconds': eleven.get('requestTimeoutSeconds')
                    if isinstance(eleven.get('requestTimeoutSeconds'), int) else 30,
            },
        },
        'ui': {
            'alwaysOnTop': bool(ui.get('alwaysOnTop', False)),
        },
    }


def _write_settings(data: dict) -> None:
    """Merge `data` (the settings UI's edited view) back into the two config
    files. Reads each file first, mutates only the keys the UI actually touches,
    and writes the merged result — so adjacent settings (TTS, STT) the panel
    doesn't expose are preserved intact."""
    llm_in = data.get('llm') if isinstance(data.get('llm'), dict) else {}
    user_name_in = data.get('user_name')
    workspace_in = data.get('workspace') if isinstance(data.get('workspace'), dict) else {}
    tts_in = data.get('tts') if isinstance(data.get('tts'), dict) else {}
    ui_in = data.get('ui') if isinstance(data.get('ui'), dict) else {}

    # ---- llm.config.json -----------------------------------------------------
    # The UI sends `llm: {configs: [...], defaultId}`. Re-read the existing file and merge per
    # config (matched by id) so unknown keys we don't expose (e.g. fallback_emotions) survive.
    llm_existing_raw: dict = {}
    if _LLM_CONFIG_PATH.exists():
        try:
            with open(_LLM_CONFIG_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                llm_existing_raw = raw
        except (OSError, json.JSONDecodeError) as e:
            print(f'[config] could not re-read {_LLM_CONFIG_PATH}: {e}', file=sys.stderr)

    # Map existing per-config dicts by id so we can preserve keys the UI doesn't touch. A legacy
    # flat file is treated as the single "default" config.
    existing_by_id: dict[str, dict] = {}
    existing_list = llm_existing_raw.get('configs')
    if isinstance(existing_list, list):
        for item in existing_list:
            if isinstance(item, dict) and str(item.get('id') or '').strip():
                existing_by_id[str(item['id']).strip()] = item
    elif llm_existing_raw:
        existing_by_id[chat_config.DEFAULT_CONFIG_ID] = {
            k: v for k, v in llm_existing_raw.items() if k != 'configs'}

    configs_in = llm_in.get('configs') if isinstance(llm_in.get('configs'), list) else []
    out_configs: list[dict] = []
    used_ids: set[str] = set()
    for i, cfg_in in enumerate(configs_in):
        if not isinstance(cfg_in, dict):
            continue
        cid = str(cfg_in.get('id') or '').strip() or f'{chat_config.DEFAULT_CONFIG_ID}_{i}'
        while cid in used_ids:
            cid = f'{cid}_{i}'
        used_ids.add(cid)
        merged = dict(existing_by_id.get(cid) or {})
        merged['id'] = cid
        merged['name'] = str(cfg_in.get('name') or '').strip() or cid
        for key in _LLM_WIRE_KEYS:
            if key in cfg_in:
                merged[key] = cfg_in[key]
        out_configs.append(merged)

    # Never write an empty config set — keep at least the existing/default one.
    if not out_configs:
        out_configs = list(existing_by_id.values()) or [
            {'id': chat_config.DEFAULT_CONFIG_ID, 'name': chat_config.DEFAULT_CONFIG_NAME}]

    default_id = str(llm_in.get('defaultId') or '').strip()
    if not any(c.get('id') == default_id for c in out_configs):
        default_id = out_configs[0].get('id') or chat_config.DEFAULT_CONFIG_ID

    llm_out = {'configs': out_configs, 'default_id': default_id}
    with open(_LLM_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(llm_out, f, indent=2, ensure_ascii=False)

    # ---- app.config.json ----------------------------------------------
    app_existing: dict = {}
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                app_existing = raw
        except (OSError, json.JSONDecodeError) as e:
            print(f'[config] could not re-read {_CONFIG_PATH}: {e}', file=sys.stderr)

    if isinstance(user_name_in, str):
        app_existing['user_name'] = user_name_in
    if workspace_in:
        ws = app_existing.get('workspace') if isinstance(app_existing.get('workspace'), dict) else {}
        if 'allowedRoots' in workspace_in:
            ws['allowedRoots'] = [str(r) for r in (workspace_in.get('allowedRoots') or []) if isinstance(r, str) and r]
        if 'allowedCommandPrefixes' in workspace_in:
            ws['allowedCommandPrefixes'] = [str(s) for s in (workspace_in.get('allowedCommandPrefixes') or []) if isinstance(s, str) and s]
        if 'deniedCommandPrefixes' in workspace_in:
            ws['deniedCommandPrefixes'] = [str(s) for s in (workspace_in.get('deniedCommandPrefixes') or []) if isinstance(s, str) and s]
        if 'fullAccess' in workspace_in:
            ws['fullAccess'] = bool(workspace_in.get('fullAccess'))
        app_existing['workspace'] = ws
    if tts_in:
        tts = app_existing.get('tts') if isinstance(app_existing.get('tts'), dict) else {}
        if 'provider' in tts_in:
            tts['provider'] = str(tts_in.get('provider') or 'pocket').lower()
        if isinstance(tts_in.get('elevenlabs'), dict):
            el_in = tts_in['elevenlabs']
            el = tts.get('elevenlabs') if isinstance(tts.get('elevenlabs'), dict) else {}
            if 'baseUrl' in el_in: el['baseUrl'] = str(el_in.get('baseUrl') or '')
            if 'apiKey' in el_in: el['apiKey'] = str(el_in.get('apiKey') or '')
            if 'model' in el_in: el['model'] = str(el_in.get('model') or '')
            if 'stability' in el_in: el['stability'] = float(el_in.get('stability') or 0.0)
            if 'similarityBoost' in el_in: el['similarityBoost'] = float(el_in.get('similarityBoost') or 0.0)
            if 'useSpeakerBoost' in el_in: el['useSpeakerBoost'] = bool(el_in.get('useSpeakerBoost'))
            if 'speed' in el_in: el['speed'] = float(el_in.get('speed') or 1.0)
            if 'requestTimeoutSeconds' in el_in:
                el['requestTimeoutSeconds'] = max(1, int(el_in.get('requestTimeoutSeconds') or 30))
            tts['elevenlabs'] = el
        app_existing['tts'] = tts
    if ui_in:
        ui = app_existing.get('ui') if isinstance(app_existing.get('ui'), dict) else {}
        if 'alwaysOnTop' in ui_in:
            ui['alwaysOnTop'] = bool(ui_in.get('alwaysOnTop'))
        app_existing['ui'] = ui
    with open(_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(app_existing, f, indent=2, ensure_ascii=False)


# ----------- helpers -----------

def new_id() -> str:
    return 'm_' + uuid.uuid4().hex


def default_response_for(env: dict) -> dict | None:
    t = env.get('type')
    if t == TYPE_ASK_QUESTION:
        multi = bool((env.get('payload') or {}).get('multiSelect'))
        return {'type': TYPE_QUESTION_ANSWER,
                'payload': {'cancelled': True, 'text': '', 'wasMultiSelect': multi}}
    if t == TYPE_REQUEST_PERMISSION:
        return {'type': TYPE_PERMISSION_DECISION, 'payload': {'allow': False, 'scope': 'Once'}}
    if t == TYPE_SHOW_REPORT:
        return {'type': TYPE_REPORT_CLOSED, 'payload': {}}
    return None


def default_size_for(env: dict) -> tuple[int, int]:
    t = env.get('type')
    if t == TYPE_SHOW_REPORT:        return (900, 700)
    if t == TYPE_REQUEST_PERMISSION: return (560, 360)
    # AskQuestion: wide enough to host an optional side-by-side preview pane next to
    # the option list when the model attached previews to its options. Previews only
    # render in single-select per the reference behavior.
    if t == TYPE_ASK_QUESTION:
        p = env.get('payload') or {}
        options = p.get('options') or []
        has_preview = (not p.get('multiSelect')) and any(
            isinstance(o, dict) and o.get('preview') for o in options
        )
        return (980, 560) if has_preview else (640, 460)
    return (640, 460)  # fallback


def title_for(env: dict) -> str:
    p = env.get('payload') or {}
    t = env.get('type')
    if t == TYPE_SHOW_REPORT:
        title = str(p.get('title') or '').strip()
        return f'Report — {title}' if title else 'Report'
    if t == TYPE_REQUEST_PERMISSION:
        tool = p.get('toolName') or 'tool'
        return f'Approve: {tool}'
    if t == TYPE_ASK_QUESTION:
        q = str(p.get('question') or '').strip()
        if not q:
            return 'Question'
        return f'Question — {q[:57]}…' if len(q) > 60 else f'Question — {q}'
    return 'Waifu Code'


def renderer_url(hash_fragment: str) -> str:
    """Build the URL pywebview loads for a window. Hash carries the envelope or kind."""
    if DEV_URL:
        return f"{DEV_URL.rstrip('/')}/#{hash_fragment}"
    if not DIST_INDEX.exists():
        raise FileNotFoundError(
            f"Renderer build not found at {DIST_INDEX}. Run `npm run build` in CompanionApp, "
            f"or set APP_DEV_URL to point at the vite dev server."
        )
    return DIST_INDEX.as_uri() + '#' + hash_fragment


# Note: `window.app` is defined entirely in the renderer's index.html — an inline
# stub queues calls until pywebview's `window.pywebview.api` is ready. This avoids the
# race where React's useEffect runs (mid-page-load) before pywebview has injected its
# API, which used to crash with "Cannot read properties of undefined". We push Chat.*
# envelopes into the page via `window.__chatPush(envJson)`, also defined in that stub.


# Speech-to-text (SttEngine, MicRecorder, VoiceController) lives in stt.py — Bridge
# delegates anything voice-shaped to a VoiceController instance via `self.voice`.

# ----------- bridge: shared state + WS plumbing -----------

class Bridge:
    """Owns the websocket, the open task windows, and the chat window. Singleton."""

    def __init__(self, unity_port: int):
        # Fixed loopback port Unity's AppBridge listens on (app.config.json → unity.port,
        # mirrored by AppBridge.port in Unity). The WS loop dials it and retries until Unity
        # is up, so the app can start in any order and reconnect across Unity restarts.
        self._unity_port = int(unity_port)

        self._ws: websocket.WebSocketApp | None = None
        self._ws_lock = threading.Lock()  # serialize ws.send across threads
        self._stopping = False

        # Whether the Unity WS is currently up. Drives the renderer's "reconnecting…"
        # overlay. Starts False; the first connect attempt flips it (success → True,
        # immediate failure → stays False and pushes the overlay).
        self._unity_connected = False
        self._conn_lock = threading.Lock()

        # Per-task entries keyed by request envelope id. Each tracks the window and whether
        # we've already replied (so window close doesn't synthesize a duplicate default).
        self.tasks: dict[str, TaskEntry] = {}
        self._tasks_lock = threading.Lock()

        # Persistent chat window. Buffer Chat.* envelopes that arrive before the renderer
        # signals it's mounted (via notify_chat_ready), then drain on signal.
        self.chat_window: webview.Window | None = None
        self.chat_ready = False
        self.chat_buffer: list[dict] = []
        self._chat_lock = threading.Lock()

        # Local TTS (replaces Unity's ElevenLabs-HTTP TtsClient). TtsController owns
        # the provider (pocket-tts ONNX or ElevenLabs proxy) and runs synthesis on
        # worker threads, streaming `Tts.AudioChunk` envelopes back over the WS.
        # Bridge routes `Tts.*` envelopes here in handle_envelope. VOICES_DIR is
        # where bare voiceMap names resolve to user-supplied reference WAVs.
        # Constructed BEFORE the STT controller so we can hand voice the hooks it
        # needs for hands-free barge-in (is_active / cancel_active).
        self.tts = tts.make_tts_controller(
            _CONFIG,
            send_envelope=self.send_envelope,
            voices_dir=VOICES_DIR,
        )
        # Fingerprint of the TTS config the controller was built from, so a later
        # save_config only hot-swaps the provider when something TTS-related
        # actually changed (a pocket-tts rebuild reloads the ONNX engine).
        self._tts_config_sig = tts.tts_config_signature(_CONFIG)
        # Standalone pocket-tts provider used purely to encode character voice clips
        # into embeddings when pocket ISN'T the active provider. Lazily built on first
        # encode (the model load is the "generate on demand" cost). Guarded so two
        # concurrent character saves don't both kick off a load.
        self._pocket_encoder: tts.PocketTtsProvider | None = None
        self._pocket_encoder_lock = threading.Lock()

        # Local STT (replaces Unity's whisper.unity setup). The VoiceController owns
        # the engine, mic, VAD, and busy/recording state machine. Two output
        # channels are injected: push_envelope (→ renderer, chat window UI) and
        # send_envelope (→ Unity, used for hands-free auto-submit + Tts.Cancel
        # barge-in). tts_hooks wires the STT path to the TTS controller's
        # cancel/is_active so a user starting to talk during an assistant reply
        # actually stops the audio stream rather than just shouting over it.
        self.voice = stt.make_voice_controller(
            _CONFIG,
            push_envelope=self._push_or_buffer,
            send_envelope=self.send_envelope,
            tts_hooks=(self.tts.is_active, self.tts.is_playing, self.tts.cancel_active),
        )

        # OCR (RapidOCR) for the read_image_ocr tool. Stays cold until the Unity
        # side sends its first Ocr.Extract; the engine and its three ONNX models
        # are loaded on demand. We don't kick off a background warm-up because
        # OCR is purely opt-in — vision-capable backends never hit this path.
        self.ocr = ocr.make_ocr_controller(send_envelope=self.send_envelope)

        # Pillow-based image transcode/resize for the read_image tool. Decodes any
        # source format Pillow knows (PNG/JPG/GIF/WEBP/BMP/TIFF/ICO + HEIC/AVIF if
        # the optional plugins are installed), downscales to the caller's pixel cap,
        # and re-encodes as JPEG. Done in Python because Unity's ImageConversion
        # can't decode WEBP and tends to flake on exotic encodings.
        self.image_proc = image_proc.make_image_controller(send_envelope=self.send_envelope)

        # Chat orchestrator (replaces the old Unity-side ChatManager/ChatOrchestrator
        # stack). Runs on its own asyncio loop in a dedicated thread so the existing
        # sync code in this Bridge (WS reader, pywebview callbacks) can keep calling
        # into it via run_coroutine_threadsafe / `schedule`.
        self._chat_loop = asyncio.new_event_loop()
        self._chat_loop_thread = threading.Thread(
            target=self._chat_loop_runner, name='chat-asyncio', daemon=True)
        self._chat_loop_thread.start()

        # Registry of named LLM configs (multiple + a selected default). The default's
        # ChatBackendConfig seeds the manager; per-chat switching picks others by id.
        llm_registry = chat_config.load_registry(_LLM_CONFIG_PATH)
        default_entry = llm_registry.default()
        chat_cfg = default_entry.config if default_entry is not None else chat_config.ChatBackendConfig()
        # User display name lives in app.config.json (alongside tts/stt user
        # preferences), not in llm.config.json which is for LLM transport settings.
        user_name = _CONFIG.get("user_name")
        if isinstance(user_name, str) and user_name.strip():
            chat_cfg.user_name = user_name.strip()
        save_mgr = SaveLoadManager()
        self.chat = ChatManager(
            config=chat_cfg,
            llm_registry=llm_registry,
            save_manager=save_mgr,
            send_to_unity=self.send_envelope,
            push_to_renderer=self._chat_push_to_renderer,
            send_unity_request=self._chat_send_request,
            tts_synthesize=self._chat_tts_synthesize,
            tts_cancel=self.tts.cancel_active,
            schedule=self._chat_schedule,
            open_modal=self._chat_open_modal,
            ask_modal=self._chat_ask_modal,
            # Sync helpers Read's image branch uses. The image controller hits
            # Pillow (vision path); the OCR controller hits RapidOCR (text-only
            # backend fallback). Both are sync; the tool wraps each call in
            # asyncio.to_thread so the chat loop stays responsive.
            image_processor=self.image_proc.process_image_sync,
            ocr_processor=self.ocr.ocr_image_sync,
            event_loop=self._chat_loop,
            voice_supported=True,
            verbose=False,
            # The (sync, heavy) encoder the character-save path calls when a pocket
            # voice clip changes; embeddings are written into the character's folder.
            encode_pocket_voice=self._encode_pocket_voice,
            # Whether a character's voices map yields a voice for the ACTIVE provider —
            # gates whether voice mode can be turned on (no voice → no-voice mode).
            voice_available=self.tts.has_character_voice,
            # Per-chat voice provider: switch the active TTS engine to a chat's chosen
            # provider on open, and read the active provider's key for the init payload.
            set_voice_provider=self._chat_set_voice_provider,
            voice_provider_getter=self.tts.provider_key,
            # Chat-loading overlay readiness gates: True once each engine's load has resolved
            # (loaded or terminally errored). The model stage is gated separately via Session.Ready.
            tts_ready_getter=self.tts.is_ready,
            stt_ready_getter=self.voice.ready_or_errored,
        )

    def _chat_set_voice_provider(self, name: str) -> None:
        """Switch the active TTS provider to `name` ("pocket"/"elevenlabs") for a chat that
        chose a specific engine. No-op if it's already active. Builds the provider from the
        current app.config.json with `tts.provider` overridden, so the engine-specific
        settings (pocket precision/language, elevenlabs base URL/key) still come from config.
        Keeps `_tts_config_sig` honest so a later settings save only reloads on a real change."""
        name = (name or '').strip().lower()
        if name not in ('pocket', 'elevenlabs'):
            return
        try:
            if self.tts.provider_key() == name:
                return
        except Exception:
            pass
        cfg = dict(_load_app_config())
        tts_cfg = dict(cfg.get('tts') if isinstance(cfg.get('tts'), dict) else {})
        tts_cfg['provider'] = name
        cfg['tts'] = tts_cfg
        try:
            self.tts.reload(tts.build_provider(cfg, VOICES_DIR))
            self._tts_config_sig = tts.tts_config_signature(cfg)
        except Exception as e:
            print(f'[tts] per-chat provider switch to {name!r} failed: {e}', file=sys.stderr)

    def _chat_open_modal(self, env: dict) -> None:
        """ChatManager-facing modal spawner. Pywebview's create_window has to be called
        on the GUI thread, so marshal onto the main thread via webview.windows[0]'s
        evaluate_js... actually create_window is documented as thread-safe under
        pywebview 5.x (it queues on the GUI loop internally). We just call it directly."""
        try:
            self.spawn_task_window(env)
        except Exception as e:
            print(f"[CompanionApp] spawn_task_window failed: {e}", file=sys.stderr)

    async def _chat_ask_modal(self, env: dict) -> dict:
        """Tool-facing modal helper used by AskUserQuestion and the approval gate.
        Spawns a task window whose TaskApi.reply resolves a Python future on the
        chat asyncio loop INSTEAD of sending the answer over the WS to Unity (the
        old Unity-side tools needed it; now everything is Python).

        Returns the reply payload dict. On window-close without a reply, returns
        the same default the WS path would have synthesized (cancelled / deny /
        closed-report)."""
        fut: asyncio.Future[dict] = self._chat_loop.create_future()

        def _on_local_reply(payload: dict):
            def _setter(p=payload):
                if not fut.done():
                    fut.set_result(p if isinstance(p, dict) else {})
            try:
                self._chat_loop.call_soon_threadsafe(_setter)
            except RuntimeError as e:
                print(f'[CompanionApp] ask_modal: loop closed: {e}', file=sys.stderr)

        try:
            self.spawn_task_window(env, local_reply=_on_local_reply)
        except Exception as e:
            print(f"[CompanionApp] ask_modal spawn failed: {e}", file=sys.stderr)
            return default_response_for(env)['payload'] if default_response_for(env) else {}
        return await fut

    def _chat_push_to_renderer(self, env: dict) -> None:
        """ChatManager-facing push to the chat window. Routes through
        VoiceController.patch_unity_envelope so it can patch Chat.Init's voiceSupported
        flag + append the synthesized Chat.VoiceBusy follow-up reflecting current STT
        model state. Same patcher the legacy route_to_chat used."""
        for e in self.voice.patch_unity_envelope(env):
            self._push_or_buffer(e)

    # ----- chat loop helpers -----

    def _chat_loop_runner(self) -> None:
        asyncio.set_event_loop(self._chat_loop)
        self._chat_loop.run_forever()

    def _chat_schedule(self, coro) -> None:
        """ChatManager-facing schedule callable. Runs the coroutine on the chat loop
        from any thread. Errors raised by the coroutine are logged but not propagated."""
        def _runner(c=coro):
            try:
                fut = asyncio.run_coroutine_threadsafe(c, self._chat_loop)
                fut.add_done_callback(self._log_chat_task_error)
            except RuntimeError as e:
                # Loop already closed during shutdown — swallow.
                print(f'[chat] schedule failed: {e}', file=sys.stderr)
        _runner()

    @staticmethod
    def _log_chat_task_error(fut) -> None:
        try:
            fut.result()
        except Exception as e:
            print(f'[chat] task failed: {type(e).__name__}: {e}', file=sys.stderr)

    def _chat_send_request(self, env_type: str, payload: dict) -> str:
        """ChatManager-facing helper: build a request envelope, fire it at Unity, return
        the id we generated so ChatManager can register a future under it."""
        env_id = new_id()
        self.send_envelope({'id': env_id, 'type': env_type, 'payload': payload or {}})
        return env_id

    async def _chat_tts_synthesize(self, text: str) -> None:
        """ChatManager-facing TTS bridge. Runs synthesize_text in a worker thread so
        the speech pipeline can await each sentence's completion without blocking the
        chat loop. The chunks are streamed to Unity from within the worker via
        TtsController.synthesize_text → _handle_synthesize → send_envelope (already
        thread-safe over the WS lock)."""
        if not text or not text.strip():
            return
        # Lowercase before synthesis. The renderer-facing transcript keeps its
        # original casing (this only affects what the TTS model reads), which
        # softens delivery and stops the model from spelling out ALL-CAPS words
        # letter by letter.
        text = text.lower()
        # Resolve the active character's voice for whatever provider is currently active
        # (pocket embedding path or ElevenLabs voice id). Empty → provider default.
        voices: dict = {}
        if self.chat is not None and self.chat.orchestrator is not None and self.chat.orchestrator.session is not None:
            voices = self.chat.orchestrator.session.character.voices or {}
        voice = self.tts.resolve_character_voice(voices)
        if not voice:
            # No voice for the active provider → no-voice mode (text lip-sync only).
            # Voice mode should already be gated off; this is the belt-and-braces guard
            # so we never speak in a stand-in voice.
            return
        await asyncio.to_thread(self.tts.synthesize_text, text, voice)

    def _encode_pocket_voice(self, clip_path: str, out_path: str) -> None:
        """Encode a reference voice clip into a pocket-tts embedding `.npy` at `out_path`.
        Reuses the active provider's engine when pocket is active; otherwise lazily builds
        a standalone encoder (the 'generate on demand' path). Sync + heavy — the chat
        manager calls it via asyncio.to_thread."""
        provider = self.tts.pocket_provider()
        if provider is None:
            with self._pocket_encoder_lock:
                if self._pocket_encoder is None:
                    print('[tts] building standalone pocket encoder (active provider isn\'t pocket)…')
                    self._pocket_encoder = tts.build_pocket_encoder(_CONFIG, VOICES_DIR)
                provider = self._pocket_encoder
        provider.encode_voice_to_file(clip_path, out_path)

    # ----- WS lifecycle -----

    def _set_unity_connected(self, connected: bool) -> None:
        """Update the connection flag and, on a state CHANGE, push it to the chat window.
        Called from the WS thread (connect/disconnect edges); guarded so repeated failed
        reconnect attempts don't spam the renderer."""
        with self._conn_lock:
            if self._unity_connected == connected:
                return
            self._unity_connected = connected
        self._push_unity_connection()

    def _push_unity_connection(self) -> None:
        self._push_or_buffer({
            'id': new_id(),
            'type': EVT_UNITY_CONNECTION,
            'payload': {'connected': self._unity_connected},
        })

    def start_ws_thread(self) -> None:
        t = threading.Thread(target=self._ws_loop, name='ws-client', daemon=True)
        t.start()

    def _ws_loop(self) -> None:
        backoff = 1.0
        while not self._stopping:
            # Fixed loopback port (no token — see AppBridge). Dial it and retry with backoff;
            # while Unity is down the connect just fails and the renderer keeps the overlay up.
            url = f'ws://127.0.0.1:{self._unity_port}/'
            opened = {'value': False}

            def on_open(ws):
                opened['value'] = True
                self._set_unity_connected(True)
                self.send_envelope({
                    'id': new_id(),
                    'type': TYPE_CLIENT_READY,
                    'payload': {'clientVersion': f'app-py-{PROTOCOL_VERSION}'},
                })
                # Auto-sync: Unity lost its avatar state on restart/reconnect, so re-push the
                # live session (reloads the character if needed + re-applies the current emotion).
                self._chat_schedule(self.chat.resync_unity_session())

            def on_message(_ws, raw):
                try:
                    env = json.loads(raw)
                except Exception as e:
                    print(f'[CompanionApp] Bad JSON from server: {e}', file=sys.stderr)
                    return
                self.handle_envelope(env)

            def on_error(_ws, err):
                # `close` fires after — let that handle reconnect.
                pass

            def on_close(_ws, _code, _reason):
                pass

            ws = websocket.WebSocketApp(
                url, on_open=on_open, on_message=on_message,
                on_error=on_error, on_close=on_close,
            )
            with self._ws_lock:
                self._ws = ws
            try:
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f'[CompanionApp] WS run_forever crashed: {e}', file=sys.stderr)
            with self._ws_lock:
                self._ws = None
            self._set_unity_connected(False)

            if self._stopping:
                return
            # Successful connect resets backoff; immediate failure backs off exponentially.
            if opened['value']:
                backoff = 1.0
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    def send_envelope(self, env: dict) -> None:
        with self._ws_lock:
            ws = self._ws
            if not ws or not ws.sock or not ws.sock.connected:
                print(f"[CompanionApp] Drop send: socket not open. {env.get('type')}", file=sys.stderr)
                return
            try:
                ws.send(json.dumps(env))
            except Exception as e:
                print(f'[CompanionApp] send failed: {e}', file=sys.stderr)

    def stop(self) -> None:
        self._stopping = True
        with self._ws_lock:
            ws = self._ws
        if ws:
            try: ws.close()
            except Exception: pass
        # Stop the chat asyncio loop. Cancel any pending tasks first so they get a
        # chance to clean up (LLM streams, TTS thread pool entries) before the loop dies.
        try:
            self._chat_loop.call_soon_threadsafe(self._chat_loop.stop)
        except RuntimeError:
            pass

    # ----- routing -----

    def handle_envelope(self, env: dict) -> None:
        t = env.get('type')
        if t == TYPE_HELLO:
            return  # we already announced ourselves on connect

        if t == TYPE_OPEN_CHAT_WINDOW:
            self.ensure_chat_window()
            return

        # Chat.ViewState is a real inbound Chat.* the orchestrator consumes (per-chat camera/window
        # framing) — route it before the legacy Chat.* drop below, which would otherwise discard it.
        if t == 'Chat.ViewState':
            self.chat.handle_unity_envelope(env)
            return

        # Chat orchestrator lives in Python now. Any (other) inbound Chat.* from Unity is
        # legacy — drop it so two sources don't race the renderer. Once the Unity-side
        # ChatUIController is deleted (phase 16) this branch becomes a no-op.
        if isinstance(t, str) and t.startswith(CHAT_TYPE_PREFIX):
            return

        # Chat orchestrator lives in Python now — these envelopes go to ChatManager
        # rather than being forwarded to the renderer (which is where they used to come
        # from Unity-side ChatUIController). The renderer still receives Chat.* — but
        # via ChatManager pushing them through _push_or_buffer, not via this path.
        # Tts.PlaybackStarted / Tts.PlaybackEnded similarly: ChatManager wants to know
        # so it can drive the speaking-edge UI; the TtsController also gets them for
        # the speakers-busy flag used by STT barge-in.
        if t in (
            'Character.InspectModelEmotionsResult',
            'Character.CaptureViewResult',
            'Tts.PlaybackStarted', 'Tts.PlaybackEnded',
            'Session.Ready',
            'Touch.Event',
            'Touch.ModeChanged',
        ):
            self.chat.handle_unity_envelope(env)
            # Tts.* still feeds the TtsController too — the speakers-busy flag matters
            # for STT echo suppression, separate from ChatManager's UI edge.
            if isinstance(t, str) and t.startswith('Tts.'):
                self.tts.handle_envelope(env)
            return

        # Tts.* envelopes never reach the renderer; the TtsController handles
        # `Tts.Synthesize` and streams audio chunks back to Unity directly.
        if self.tts.is_tts_envelope_type(t):
            self.tts.handle_envelope(env)
            return

        # Ocr.* — same pattern as Tts: bypass the renderer, run on a worker, and
        # reply with Ocr.ExtractResult (correlated by replyTo).
        if self.ocr.is_ocr_envelope_type(t):
            self.ocr.handle_envelope(env)
            return

        # Image.* — Pillow transcode/resize for the vision read_image tool.
        if self.image_proc.is_image_envelope_type(t):
            self.image_proc.handle_envelope(env)
            return

        if t not in INTERACTIVE_TYPES:
            return
        if not env.get('id'):
            return
        self.spawn_task_window(env)

    # ----- chat window -----

    def ensure_chat_window(self) -> None:
        """Bring the chat window to the foreground. The window is created at startup in
        `install_chat_window` (visible) so the OpenChatWindow envelope just restores it
        from a possibly-minimized state."""
        if self.chat_window is None:
            print('[chat] ensure_chat_window: window not installed yet', file=sys.stderr)
            return
        try:
            self.chat_window.restore()
            self.chat_window.show()
        except Exception as e:
            print(f'[chat] ensure_chat_window show failed: {e}', file=sys.stderr)

    def install_chat_window(self) -> webview.Window:
        """Create the chat window VISIBLE before webview.start(). It must be visible at
        creation, not `hidden=True` — WebView2 throttles JS timers and React effect
        scheduling for any window that's in 'background state' at load time, and
        unhiding later does NOT un-stick the already-scheduled microtasks. The window
        shows the 'Waiting for Unity…' placeholder briefly until Chat.Init arrives."""
        api = ChatApi(self)
        # Restore the persisted always-on-top (pin) preference at creation, so the window
        # comes up floating if the user left it pinned last session (see set_always_on_top).
        ui_cfg = _CONFIG.get('ui') if isinstance(_CONFIG.get('ui'), dict) else {}
        start_on_top = bool(ui_cfg.get('alwaysOnTop', False))
        win = webview.create_window(
            title='Waifu Code',
            url=renderer_url('kind=chat'),
            js_api=api,
            width=480,
            height=720,
            background_color='#15161a',
            on_top=start_on_top,
            # No native title bar — the renderer draws its own (Titlebar.tsx). easy_drag=False so
            # only the `.pywebview-drag-region` element drags the window (not the whole surface,
            # which would break text selection / scrolling).
            frameless=True,
            easy_drag=False,
        )
        api._window = win

        # Closing the chat window quits the app. It's a standalone app now (launched
        # via run-app.bat, not by Unity), so the user owns its lifecycle — closing
        # should exit, not minimize-and-strand. Setting _stopping first keeps the WS loop
        # from trying to reconnect during teardown; returning True lets the close proceed,
        # which ends webview.start() and runs shutdown().
        def on_closing():
            self._stopping = True
            return True

        win.events.closing += on_closing
        # Register the .wcc drop-import handler once the DOM is up. Python-side DOM drop
        # handlers are the only way to get a dropped file's full path (pywebviewFullPath);
        # a plain JS drop event only sees sandboxed File objects with no path.
        win.events.loaded += self._register_chat_drop_import
        self.chat_window = win
        self.chat_ready = False
        return win

    _drop_import_registered = False

    def _register_chat_drop_import(self, *_args) -> None:
        """Attach the drop handler to the chat window's body (idempotent — `loaded` can
        fire again on a reload). The renderer's document-level dragover preventDefault
        is what allows the drop to happen at all."""
        if self._drop_import_registered or self.chat_window is None:
            return
        try:
            from webview.dom import DOMEventHandler
            self.chat_window.dom.body.events.drop += DOMEventHandler(
                self._on_chat_file_drop, prevent_default=True, stop_propagation=True)
            self._drop_import_registered = True
        except Exception as e:
            print(f'[chat] drop-import registration failed: {e}', file=sys.stderr)

    def _on_chat_file_drop(self, event) -> None:
        """Forward the dropped files' FULL paths to the renderer as App.FilesDropped.
        Python is only the path source (a pywebview DOM handler is the sole way to see
        pywebviewFullPath); the RENDERER decides what a drop means for its active view —
        home page imports .wcc bundles, the chat view is free to treat drops as
        attachments later without the two features fighting over one handler."""
        try:
            files = (event.get('dataTransfer') or {}).get('files') or []
        except AttributeError:
            return
        paths = [str(f.get('pywebviewFullPath')).replace('\\', '/')
                 for f in files if isinstance(f, dict) and f.get('pywebviewFullPath')]
        if paths:
            self._push_or_buffer({'id': new_id(), 'type': 'App.FilesDropped',
                                  'payload': {'paths': paths}})

    def route_to_chat(self, env: dict) -> None:
        # VoiceController gets first crack at envelopes from Unity — for Chat.Init it
        # patches voiceSupported and appends a synthesized Chat.VoiceBusy reflecting
        # current model state. Everything else passes through unchanged.
        for e in self.voice.patch_unity_envelope(env):
            self._push_or_buffer(e)

    def _push_or_buffer(self, env: dict) -> None:
        if self.chat_window is None:
            with self._chat_lock:
                self.chat_buffer.append(env)
            self.ensure_chat_window()
            return
        with self._chat_lock:
            if not self.chat_ready:
                self.chat_buffer.append(env)
                return
        self._push_chat_env(env)

    def _push_chat_env(self, env: dict) -> None:
        if self.chat_window is None:
            return
        try:
            payload_js = json.dumps(json.dumps(env))  # double-encode so it's a JS string literal
            self.chat_window.evaluate_js(f'window.__chatPush({payload_js})')
        except Exception as e:
            print(f'[chat] _push_chat_env failed ({env.get("type")}): {e}', file=sys.stderr)

    def on_chat_ready(self) -> None:
        with self._chat_lock:
            already_ready = self.chat_ready
            self.chat_ready = True
            pending = self.chat_buffer[:]
            self.chat_buffer.clear()
        for env in pending:
            self._push_chat_env(env)
        # Tell the freshly-mounted renderer the current Unity connection state so it can
        # show the "reconnecting…" overlay right away if we're not connected yet.
        self._push_unity_connection()
        # Kick off STT + TTS model loads AFTER the renderer is mounted. The Python
        # GIL means any heavy Python-side work (import sherpa_onnx, hf_hub_download,
        # ORT session creation) blocks pywebview's JS bridge — so anything that
        # needs that bridge (button clicks, evaluate_js round-trips) freezes for the
        # duration. Triggering the loads post-mount keeps the initial chat experience
        # snappy. Guarded against re-fire if notify_chat_ready arrives more than
        # once (e.g., StrictMode dev double-mount).
        if not already_ready:
            self.voice.start_load()
            self.tts.start_load()
            # Bootstrap the chat session — load the most recent save or create-new
            # against the first registered character. Without this, the renderer
            # waits forever on its "Waiting for Unity…" placeholder.
            print("[chat] on_chat_ready: scheduling bootstrap", file=sys.stderr)
            self._chat_schedule(self.chat.bootstrap())

    # ----- task windows -----

    def spawn_task_window(self, env: dict, local_reply=None) -> None:
        """Spawn a modal task window (ShowReport / AskQuestion / RequestPermission).

        `local_reply` is an optional callback `(payload: dict) -> None` invoked when
        the user clicks a button in the modal. When set, TaskApi.reply calls it
        INSTEAD of sending the reply over the WS to Unity — used by the Python
        tool runner (AskUserQuestion + approval gate) which awaits a Python future.
        When None (the default), the legacy behavior applies: the reply envelope
        is sent over the WS, addressed `replyTo` the spawning envelope's id."""
        size = default_size_for(env)
        title = title_for(env)
        hash_str = 'envelope=' + quote(json.dumps(env, separators=(',', ':')))

        api = TaskApi(self, env, local_reply=local_reply)
        win = webview.create_window(
            title=title,
            url=renderer_url(hash_str),
            js_api=api,
            width=size[0],
            height=size[1],
            background_color='#15161a',
            on_top=True,  # brief focus-steal; we drop it after load
            # No native chrome — the renderer draws its own titlebar (Titlebar.tsx), with the
            # control set the window type warrants (report: min/max/close; dialog: close only).
            frameless=True,
            easy_drag=False,
        )
        api._window = win
        entry = TaskEntry(window=win, envelope=env, api=api)
        with self._tasks_lock:
            self.tasks[env['id']] = entry

        # Drop the always-on-top pin shortly after load so the user can stack other windows.
        def on_loaded():
            def drop_pin():
                time.sleep(0.2)
                try: win.on_top = False
                except Exception: pass
            threading.Thread(target=drop_pin, daemon=True).start()
        win.events.loaded += on_loaded

        def on_closed():
            with self._tasks_lock:
                entry2 = self.tasks.pop(env['id'], None)
            if entry2 and not entry2.api._replied:
                # User dismissed without explicit action — synthesize the safe default.
                d = default_response_for(env)
                if d is not None:
                    if local_reply is not None:
                        try: local_reply(d['payload'])
                        except Exception as e:
                            print(f'[CompanionApp] local_reply (default) failed: {e}', file=sys.stderr)
                    else:
                        self.send_envelope({
                            'id': new_id(),
                            'type': d['type'],
                            'replyTo': env['id'],
                            'payload': d['payload'],
                        })
        win.events.closed += on_closed

    def close_task_window(self, envelope_id: str) -> None:
        with self._tasks_lock:
            entry = self.tasks.get(envelope_id)
        if entry is None:
            return
        try: entry.window.destroy()
        except Exception: pass


class TaskEntry:
    __slots__ = ('window', 'envelope', 'api')

    def __init__(self, window: webview.Window, envelope: dict, api: 'TaskApi'):
        self.window = window
        self.envelope = envelope
        self.api = api


# ----------- JS API instances exposed to the renderer -----------

class _BaseApi:
    """Shared no-op stubs so the renderer shim can call any method on any window's API
    without TypeError — task windows ignore chat methods, chat ignores task `reply`.
    Subclasses set ``_log_tag`` so renderer logs are attributed to their window.

    CRITICAL: every instance attribute MUST start with `_`. pywebview's `get_functions`
    walks `dir(instance)`, recurses into any non-callable attribute that has `__module__`,
    and tries to expose its methods too. Storing the Bridge / webview.Window directly
    sends pywebview down a tree that ends at `Bounds.Empty.Empty.Empty…` (Rectangle.Empty
    returns a fresh Rectangle each access — infinite recursion). When that raises,
    EVERY method registration is dropped, so `api.log`, `api.notify_chat_ready`, etc.
    end up `undefined` on the JS side. Names starting with `_` are filtered out
    (util.py:193), so the underscore-prefix convention is what keeps us safe."""

    _log_tag = 'renderer'

    def reply(self, type_, payload): pass
    def send_chat(self, type_, payload): pass
    def notify_chat_ready(self): pass
    # Frameless-window controls. No-ops by default; ChatApi/TaskApi override the ones
    # their window actually offers (the renderer titlebar only shows the supported set).
    def minimize_window(self): pass
    def maximize_window(self): pass
    def restore_window(self): pass
    def set_always_on_top(self, on): pass
    def close_window(self): pass

    def confirm(self, opts):
        return False

    def log(self, level, message):
        """Renderer console + uncaught errors funnel here. We route through Python's
        configured stdout/stderr so they land in app.log alongside Python traces."""
        try:
            level_str = str(level or 'log').lower()
            text = str(message or '')
            line = f'[{self._log_tag}.{level_str}] {text}'
            (sys.stderr if level_str in ('warn', 'warning', 'error') else sys.stdout).write(line + '\n')
        except Exception:
            pass


class TaskApi(_BaseApi):
    def __init__(self, bridge: Bridge, envelope: dict, local_reply=None):
        # Underscore-prefixed so pywebview's dir()-based JS-API discovery skips them
        # (see _BaseApi docstring).
        self._bridge = bridge
        self._envelope = envelope
        self._replied = False
        self._window: webview.Window | None = None
        self._log_tag = f"task.{envelope.get('type', '?')}"
        # When set, reply() invokes this with the payload INSTEAD of sending the
        # answer over the WS. Used by the Python tool runner — see Bridge.ask_modal.
        self._local_reply = local_reply

    def reply(self, type_, payload):
        if self._replied:
            return
        self._replied = True
        if self._local_reply is not None:
            try:
                self._local_reply(payload if isinstance(payload, dict) else {})
            except Exception as e:
                print(f'[CompanionApp] local_reply failed: {e}', file=sys.stderr)
        else:
            self._bridge.send_envelope({
                'id': new_id(),
                'type': type_,
                'replyTo': self._envelope['id'],
                'payload': payload,
            })
        # Defer window destruction to a worker thread. Calling `destroy()` directly
        # from inside the JS bridge handler can deadlock on Windows — pywebview's
        # bridge is still waiting for THIS function to return while destroy() tries
        # to dispose the WebView2 host, and they end up wedged on the GUI thread
        # ("window is not responding"). A brief sleep lets the bridge return first,
        # then destroy runs cleanly off-thread.
        win = self._window
        if win is not None:
            def _destroy_off_thread():
                time.sleep(0.05)
                try: win.destroy()
                except Exception: pass
            threading.Thread(target=_destroy_off_thread, name='task-destroy',
                             daemon=True).start()

    # ----- Frameless-window controls (the renderer draws its own titlebar) -----
    def minimize_window(self):
        try:
            if self._window is not None:
                self._window.minimize()
        except Exception as e:
            print(f'[task] minimize_window failed: {e}', file=sys.stderr)

    def maximize_window(self):
        try:
            if self._window is not None:
                self._window.maximize()
        except Exception as e:
            print(f'[task] maximize_window failed: {e}', file=sys.stderr)

    def restore_window(self):
        try:
            if self._window is not None:
                self._window.restore()
        except Exception as e:
            print(f'[task] restore_window failed: {e}', file=sys.stderr)

    def close_window(self):
        """Close (destroy) just this task window. on_closed then synthesizes the safe
        default reply (ReportClosed / cancelled / deny) if the user hadn't acted yet.
        Destroy off-thread for the same anti-deadlock reason as reply()."""
        win = self._window
        if win is None:
            return
        def _destroy_off_thread():
            time.sleep(0.05)
            try: win.destroy()
            except Exception: pass
        threading.Thread(target=_destroy_off_thread, name='task-close', daemon=True).start()

    def confirm(self, opts):
        return _confirm_dialog(self._window, opts)


class ChatApi(_BaseApi):
    _log_tag = 'chat'

    def __init__(self, bridge: Bridge):
        self._bridge = bridge
        self._window: webview.Window | None = None

    # ----- Renderer → ChatManager: `send_chat` used to forward Chat.* envelopes
    #       to Unity over the WS. With the orchestrator moved to Python, the
    #       ChatManager here handles them directly — submit/load/rollback/etc.
    #       Renderer↔app-local commands (voice, hands-free) still have their
    #       own direct methods below. -----

    def send_chat(self, type_, payload):
        if not isinstance(type_, str) or not type_.startswith(CHAT_TYPE_PREFIX):
            return
        env = {'id': new_id(), 'type': type_, 'payload': payload or {}}
        self._bridge.chat.handle_renderer_chat(env)

    def notify_chat_ready(self):
        self._bridge.on_chat_ready()

    def open_player(self):
        """Launch the Unity player — the "Open Player" button on the disconnected overlay.
        Reuses the autostart launch path; a no-op if the player is already running. Returns
        {ok, launched?, error?} so the overlay can show feedback."""
        return launch_player()

    # ----- Voice command methods (renderer → app, no Unity round-trip).
    #       Each pairs with a public method on VoiceController. -----

    def start_recording(self):
        self._bridge.voice.start_ptt()

    def stop_recording(self):
        self._bridge.voice.stop_ptt()

    def set_hands_free(self, enabled):
        self._bridge.voice.set_hands_free(bool(enabled))

    def set_auto_submit(self, enabled):
        self._bridge.voice.set_auto_submit(bool(enabled))

    def set_wake_word(self, enabled, phrase=None):
        self._bridge.voice.set_wake_word(bool(enabled), phrase if isinstance(phrase, str) else None)

    def clear_voice_transcript(self):
        """Drops the recognizer's in-flight utterance + wake-word arm so the next
        endpoint doesn't surface text the user already cancelled."""
        self._bridge.voice.clear_in_flight_utterance()

    # ----- Settings panel API. Reads/writes the two on-disk config files and
    #       exposes a folder picker for the workspace root. The renderer's
    #       ConfigView uses these methods directly via window.pywebview.api. -----

    def get_config(self):
        """Return everything the config UI needs in one shot: the LLM block
        (api_url / api_key / model / temperature / thinking), the user_name,
        the workspace allowedRoots, and the tts block. The renderer
        decides what to surface."""
        return _read_settings()

    def save_config(self, data):
        """Persist the config UI's edited values back to the two JSON files
        (`llm.config.json` carries the LLM block; `app.config.json` carries
        user_name + workspace + tts), then hot-reload them into the
        live ChatManager and TTS controller so the change applies immediately —
        no restart needed. Returns {ok: True} on success or {ok: False, error:
        str} on any IO/JSON failure — the renderer surfaces the error inline."""
        if not isinstance(data, dict):
            return {'ok': False, 'error': 'invalid payload'}
        try:
            _write_settings(data)
        except (OSError, ValueError) as e:
            print(f'[config] save failed: {type(e).__name__}: {e}', file=sys.stderr)
            return {'ok': False, 'error': str(e)}
        # Re-read disk + push into ChatManager. Hot reload mutates the existing
        # dataclasses in place so LlmClient / ToolManager / each tool see the new
        # values on their next call.
        try:
            fresh_app = _load_app_config()
            fresh_registry = chat_config.load_registry(_LLM_CONFIG_PATH)
            fresh_default = fresh_registry.default()
            fresh_chat_cfg = fresh_default.config if fresh_default is not None else chat_config.ChatBackendConfig()
            fresh_user = fresh_app.get('user_name')
            if isinstance(fresh_user, str) and fresh_user.strip():
                fresh_chat_cfg.user_name = fresh_user.strip()
            self._bridge.chat.reload_config(fresh_chat_cfg, fresh_app, fresh_registry)
            # Hot-swap the TTS provider in place (keeps the controller identity so
            # the STT/chat hooks captured at startup stay valid) — only when the
            # TTS config actually changed, since a pocket-tts rebuild reloads the
            # ONNX engine.
            new_tts_sig = tts.tts_config_signature(fresh_app)
            if new_tts_sig != self._bridge._tts_config_sig:
                self._bridge.tts.reload(tts.build_provider(fresh_app, VOICES_DIR))
                self._bridge._tts_config_sig = new_tts_sig
                # The active provider changed — the current character may now have (or
                # lack) a voice for it, so re-gate voice mode and tell the renderer.
                self._bridge._chat_schedule(self._bridge.chat.refresh_voice_availability())
        except Exception as e:
            # Save succeeded but reload failed — surface a soft warning so the
            # user knows they need to restart, even though the file is correct on disk.
            print(f'[config] reload after save failed: {type(e).__name__}: {e}', file=sys.stderr)
            return {'ok': True, 'error': f'Saved, but couldn\'t hot-reload: {e}. Restart to apply.'}
        return {'ok': True}

    def pick_directory(self, current=None):
        """Open the OS folder picker rooted at `current` (or the user's home
        if unset). Returns the chosen absolute path with forward slashes, or
        None when the user cancels. Used by the workspace setting's "Browse…"
        button."""
        win = self._window
        if win is None:
            return None
        start = str(current) if isinstance(current, str) and current else ''
        try:
            result = win.create_file_dialog(
                FD_FOLDER, directory=start, allow_multiple=False,
            )
        except Exception as e:
            print(f'[config] pick_directory failed: {e}', file=sys.stderr)
            return None
        if not result:
            return None
        # pywebview returns a tuple/list of paths even for single-selection.
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return None
        return str(path).replace('\\', '/')

    def pick_vrm_file(self, current=None):
        """Open the OS file picker for a character model (VRM .vrm or KK .kkm). Returns
        the chosen absolute path with forward slashes, or None when the user cancels. Used by
        the character-creation page's model "Browse…" button. The app stores the path
        as-is; Unity routes by extension (.vrm -> VRM loader, .kkm -> KK loader)."""
        win = self._window
        if win is None:
            return None
        start = str(current) if isinstance(current, str) and current else ''
        try:
            result = win.create_file_dialog(
                FD_OPEN, directory=start, allow_multiple=False,
                file_types=('Character models (*.vrm;*.kkm)',
                            'VRM models (*.vrm)', 'KK models (*.kkm)'),
            )
        except Exception as e:
            print(f'[chat] pick_vrm_file failed: {e}', file=sys.stderr)
            return None
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return None
        return str(path).replace('\\', '/')

    def pick_audio_file(self, current=None):
        """Open the OS file picker for a voice reference clip. Returns the chosen
        absolute path with forward slashes, or None when the user cancels. Used by the
        character editor's pocket-tts voice "Browse…" button. soundfile/the engine reads
        wav/flac/ogg natively; other formats depend on the installed libsndfile."""
        win = self._window
        if win is None:
            return None
        start = str(current) if isinstance(current, str) and current else ''
        try:
            result = win.create_file_dialog(
                FD_OPEN, directory=start, allow_multiple=False,
                file_types=('Audio clips (*.wav;*.flac;*.ogg;*.mp3;*.m4a)', 'All files (*.*)'),
            )
        except Exception as e:
            print(f'[chat] pick_audio_file failed: {e}', file=sys.stderr)
            return None
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return None
        return str(path).replace('\\', '/')

    def pick_profile_image(self, current=None):
        """Open the OS file picker for a character profile picture, process it to the
        stored form (white-flattened, center-cropped square, ≤250px, JPEG q90) and return
        it as a base64 `data:` URL — or None on cancel/failure. The processed image is
        what the editor previews and saves with the character; no source path is kept."""
        win = self._window
        if win is None:
            return None
        start = str(current) if isinstance(current, str) and current else ''
        try:
            result = win.create_file_dialog(
                FD_OPEN, directory=start, allow_multiple=False,
                file_types=('Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif)',),
            )
        except Exception as e:
            print(f'[chat] pick_profile_image failed: {e}', file=sys.stderr)
            return None
        if not result:
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return None
        try:
            with open(path, 'rb') as f:
                raw = f.read()
        except OSError as e:
            print(f'[chat] pick_profile_image read failed: {e}', file=sys.stderr)
            return None
        jpeg, err = self._bridge.image_proc.process_avatar_sync(raw)
        if err or not jpeg:
            print(f'[chat] profile image processing failed: {err}', file=sys.stderr)
            return None
        return 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode('ascii')

    def export_character(self, char_id, suggested_name=None):
        """Export a character as a .wcc bundle (definition + model + voice + profile
        picture): save-file dialog, then the store zips the bundle there. Returns
        {ok, path?, error?}; ok=False with no error means the user cancelled."""
        win = self._window
        if win is None or not char_id:
            return {'ok': False, 'error': 'No window or character id.'}
        # Suggested file name from the character's display name, scrubbed of characters
        # Windows filenames can't contain.
        base = re.sub(r'[<>:"/\\|?*]+', '_', str(suggested_name or '').strip()) or 'character'
        try:
            result = win.create_file_dialog(
                FD_SAVE, save_filename=f'{base}.wcc',
                file_types=('Character bundle (*.wcc)',),
            )
        except Exception as e:
            print(f'[chat] export_character dialog failed: {e}', file=sys.stderr)
            return {'ok': False, 'error': str(e)}
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return {'ok': False}
        path = str(path)
        if not path.lower().endswith('.wcc'):
            path += '.wcc'
        err = self._bridge.chat._character_store.export_bundle(str(char_id), path)
        if err:
            print(f'[chat] export_character failed: {err}', file=sys.stderr)
            return {'ok': False, 'error': err}
        return {'ok': True, 'path': path.replace('\\', '/')}

    def import_character(self):
        """Pick a .wcc character bundle and hand it to the ChatManager, which imports it
        (fresh id, paths rewritten) and pushes the refreshed character list. Returns the
        picked path, or None on cancel — errors surface via the Chat.Error banner."""
        win = self._window
        if win is None:
            return None
        try:
            result = win.create_file_dialog(
                FD_OPEN, allow_multiple=False,
                file_types=('Character bundle (*.wcc)', 'All files (*.*)'),
            )
        except Exception as e:
            print(f'[chat] import_character dialog failed: {e}', file=sys.stderr)
            return None
        path = result[0] if isinstance(result, (list, tuple)) else result
        if not path:
            return None
        path = str(path).replace('\\', '/')
        self._bridge.chat.handle_renderer_chat({
            'id': new_id(), 'type': 'Chat.ImportCharacter', 'payload': {'path': path},
        })
        return path

    def pick_attachments(self, current=None):
        """Open the OS file picker (multi-select, any file type) for message attachments.
        Returns a list of absolute paths (forward slashes), or [] when cancelled. Used by
        the composer's attach button."""
        win = self._window
        if win is None:
            return []
        start = str(current) if isinstance(current, str) and current else ''
        try:
            result = win.create_file_dialog(
                FD_OPEN, directory=start, allow_multiple=True,
            )
        except Exception as e:
            print(f'[chat] pick_attachments failed: {e}', file=sys.stderr)
            return []
        if not result:
            return []
        paths = result if isinstance(result, (list, tuple)) else [result]
        return [str(p).replace('\\', '/') for p in paths if p]

    def store_clipboard_image(self, base64_data, ext='png'):
        """Persist pasted clipboard IMAGE DATA (a screenshot, an image copied from a
        browser — bytes with no file on disk) to a temp file and return its path with
        forward slashes, so it can ride the normal attachment flow. None on failure.
        Hash-named, so pasting the same image twice reuses one file."""
        try:
            raw = base64.b64decode(str(base64_data or ''), validate=False)
        except Exception:
            raw = b''
        if not raw:
            return None
        safe_ext = re.sub(r'[^a-z0-9]', '', str(ext or 'png').lower()) or 'png'
        folder = Path(tempfile.gettempdir()) / 'waifucode_clipboard'
        try:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f'pasted_{hashlib.sha1(raw).hexdigest()[:12]}.{safe_ext}'
            path.write_bytes(raw)
        except OSError as e:
            print(f'[chat] store_clipboard_image failed: {e}', file=sys.stderr)
            return None
        return str(path).replace('\\', '/')

    def read_image_data_url(self, path):
        """Read an image file and return it as a base64 `data:` URL of the ORIGINAL bytes
        (no resize/convert) for inline thumbnails in the chat. Returns None if missing, not an
        image, or larger than the preview cap. The mime is inferred from the extension."""
        try:
            if not isinstance(path, str) or not path:
                return None
            p = os.path.abspath(path)
            ext = os.path.splitext(p)[1].lower().lstrip('.')
            mimes = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                     'gif': 'image/gif', 'webp': 'image/webp', 'bmp': 'image/bmp'}
            mime = mimes.get(ext)
            if mime is None or not os.path.isfile(p):
                return None
            if os.path.getsize(p) > 30 * 1024 * 1024:  # safety cap for the inline preview
                return None
            import base64
            with open(p, 'rb') as f:
                raw = f.read()
            return f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')
        except Exception as e:
            print(f'[chat] read_image_data_url failed: {e}', file=sys.stderr)
            return None

    def read_model_screenshot(self, model_path, filename):
        """Read an outfit screenshot from inside a .kkm model archive (a zip; the filenames come
        from KK_Coordinates.json's screenshots entries) and return it as a base64 `data:` URL for
        the outfit-picker previews. Returns None when missing/unreadable."""
        try:
            if not isinstance(model_path, str) or not model_path:
                return None
            if not isinstance(filename, str) or not filename:
                return None
            p = os.path.abspath(model_path)
            if os.path.splitext(p)[1].lower() != '.kkm' or not os.path.isfile(p):
                return None
            base = filename.replace('\\', '/').rsplit('/', 1)[-1].lower()
            ext = base.rsplit('.', 1)[-1] if '.' in base else ''
            mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                    'webp': 'image/webp'}.get(ext)
            if mime is None:
                return None
            import base64
            import zipfile
            with zipfile.ZipFile(p) as zf:
                # Match by basename in case the export ever nests screenshots under a folder.
                entry = next((n for n in zf.namelist()
                              if n.rsplit('/', 1)[-1].lower() == base), None)
                if entry is None or zf.getinfo(entry).file_size > 8 * 1024 * 1024:
                    return None
                raw = zf.read(entry)
            return f'data:{mime};base64,' + base64.b64encode(raw).decode('ascii')
        except Exception as e:
            print(f'[chat] read_model_screenshot failed: {e}', file=sys.stderr)
            return None

    def read_clipboard_files(self):
        """Return file paths currently on the OS clipboard (files copied in Explorer). Used by
        the composer's paste handler so Ctrl+V attaches copied files. [] if none/unsupported."""
        try:
            return _read_clipboard_file_paths()
        except Exception as e:
            print(f'[chat] read_clipboard_files failed: {e}', file=sys.stderr)
            return []

    def open_workspace_folder(self):
        """Open the configured workspace root in the OS file explorer. Returns
        {ok: True} or {ok: False, error: str}. Used by the sidebar's Workspace button."""
        try:
            settings = _read_settings()
            roots = settings.get('workspace', {}).get('allowedRoots') or []
            root = roots[0] if roots else None
            if not root:
                return {'ok': False, 'error': 'No workspace folder is configured.'}
            path = os.path.abspath(root)
            if not os.path.isdir(path):
                return {'ok': False, 'error': f'Workspace folder does not exist: {path}'}
            if sys.platform.startswith('win'):
                os.startfile(path)  # noqa: S606 — opens the folder in Explorer
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
            return {'ok': True}
        except Exception as e:
            print(f'[config] open_workspace_folder failed: {e}', file=sys.stderr)
            return {'ok': False, 'error': str(e)}

    def confirm(self, opts):
        return _confirm_dialog(self._window, opts)

    def minimize_window(self):
        """Minimize the chat window. Drives the renderer titlebar's minimize button (the
        window is frameless, so there's no native control)."""
        try:
            if self._window is not None:
                self._window.minimize()
        except Exception as e:
            print(f'[chat] minimize_window failed: {e}', file=sys.stderr)

    def set_always_on_top(self, on):
        """Toggle whether the chat window floats above all other windows. Drives the renderer
        titlebar's pin button (the window is frameless, so there's no native control).

        Apply off-thread: the `on_top` setter marshals onto the GUI thread, which is still
        busy waiting for THIS bridge call to return — setting it inline wedges both (same
        Windows/WebView2 deadlock that reply()/close_window() avoid)."""
        win = self._window
        if win is None:
            return
        value = bool(on)
        def _apply_off_thread():
            time.sleep(0.05)
            try:
                win.on_top = value
            except Exception as e:
                print(f'[chat] set_always_on_top failed: {e}', file=sys.stderr)
            # Remember the choice so the window restores it next launch.
            _persist_ui_always_on_top(value)
        threading.Thread(target=_apply_off_thread, name='chat-on-top', daemon=True).start()

    def close_window(self):
        """Close the chat window — which quits the app (it's a standalone app). Mirrors
        the native close button the frameless window no longer has. Set _stopping first so the
        WS loop doesn't try to reconnect during teardown (same as the close-intercept handler)."""
        self._bridge._stopping = True
        try:
            if self._window is not None:
                self._window.destroy()
        except Exception as e:
            print(f'[chat] close_window failed: {e}', file=sys.stderr)


def _window_hwnd(window: webview.Window | None) -> int:
    """Best-effort Win32 handle of a pywebview window (its WinForms Form on the
    EdgeChromium backend). 0 when unavailable — MessageBoxW accepts a NULL owner."""
    try:
        handle = getattr(getattr(window, 'native', None), 'Handle', None)
        if handle is not None:
            return int(str(handle))  # .NET IntPtr → decimal string → int
    except Exception:
        pass
    return 0


def _confirm_dialog(window: webview.Window | None, opts: Any) -> bool:
    """Native yes/no dialog. Replaces the Electron dialog.showMessageBox we used to call
    via ipcMain — same reason: window.confirm() in sandboxed renderers wedges keyboard
    focus on close on some platforms.

    On Windows this is a raw MessageBoxW with MB_TOPMOST: pywebview's
    create_confirmation_dialog isn't topmost-aware, so when the chat window is pinned
    always-on-top the dialog would open BEHIND it and look like the app hung."""
    if not isinstance(opts, dict):
        opts = {}
    message = str(opts.get('message') or '')
    detail = str(opts.get('detail') or '')
    title = message
    body = detail if detail else message
    if not body:
        body = '?'
    if window is None:
        return False
    if sys.platform == 'win32':
        try:
            import ctypes
            MB_OKCANCEL = 0x00000001
            MB_ICONQUESTION = 0x00000020
            MB_SETFOREGROUND = 0x00010000
            MB_TOPMOST = 0x00040000
            IDOK = 1
            res = ctypes.windll.user32.MessageBoxW(
                _window_hwnd(window), body, title or 'Confirm',
                MB_OKCANCEL | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST)
            return res == IDOK
        except Exception as e:
            print(f'[CompanionApp] MessageBoxW confirm failed, falling back: {e}', file=sys.stderr)
    try:
        return bool(window.create_confirmation_dialog(title or 'Confirm', body))
    except Exception as e:
        print(f'[CompanionApp] confirm dialog failed: {e}', file=sys.stderr)
        return False


# ----------- entry point -----------

def _port_in_use(port: int) -> bool:
    """True if something is already listening on 127.0.0.1:port (i.e. Unity is up)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(('127.0.0.1', int(port))) == 0
    except OSError:
        return False


def _ensure_unity_job():
    """Create (once) a Win32 Job Object set to kill its processes when the job handle closes.
    We keep the handle open for the app's whole lifetime and never close it explicitly, so when
    THIS process dies for ANY reason — clean exit, crash, or Task-Manager force-kill — the OS
    closes the handle and the assigned player is terminated with us. Returns the handle, or None
    off-Windows / on failure (we then fall back to the best-effort terminate in close_player)."""
    global _unity_job
    if sys.platform != 'win32':
        return None
    if _unity_job is not None:
        return _unity_job
    try:
        import ctypes
        from ctypes import wintypes
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            return None
        _unity_job = job
        return _unity_job
    except Exception as e:
        print(f'[CompanionApp] job object setup failed: {e}', file=sys.stderr)
        return None


def _assign_to_unity_job(proc) -> None:
    """Put a freshly-spawned player into the kill-on-close job so it dies with this app even on
    a non-graceful exit. Best-effort: a failure just means we rely on close_player()."""
    job = _ensure_unity_job()
    if not job:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            print('[CompanionApp] AssignProcessToJobObject failed; player may outlive a force-kill.',
                  file=sys.stderr)
    except Exception as e:
        print(f'[CompanionApp] AssignProcessToJobObject error: {e}', file=sys.stderr)


def launch_player() -> dict:
    """Launch the Unity player on the configured port if it isn't already serving. Used by
    autostart at boot AND the disconnected-overlay "Open Player" button. Tracks the spawned
    process and binds it to the kill-on-close Job Object so it's closed with the app (see
    close_player + _ensure_unity_job). The exe path is resolved in main() against APP_ROOT, so a
    portable release (player beside the app exe) works wherever the folder is moved. Returns
    {ok, launched?, error?}:
      - exe missing                  -> {ok: False, error}
      - port already serving / alive -> {ok: True, launched: False}   (nothing to do)
      - spawned                      -> {ok: True, launched: True}
    """
    global _unity_proc
    p = _unity_exe_path
    if not p:
        return {'ok': False, 'error': 'No player path configured.'}
    if not os.path.isfile(p):
        print(f'[CompanionApp] launch: player exe not found: {p}', file=sys.stderr)
        return {'ok': False, 'error': f'Player not found: {os.path.basename(p)}'}
    if _port_in_use(_unity_port):
        print(f'[CompanionApp] launch: port {_unity_port} in use — player already running.', file=sys.stderr)
        return {'ok': True, 'launched': False}
    if _unity_proc is not None and _unity_proc.poll() is None:
        return {'ok': True, 'launched': False}   # the one we spawned is still coming up
    try:
        # Pass the port so the player binds the same one we dial (AppBridge reads --port).
        _unity_proc = subprocess.Popen([p, '--port', str(_unity_port)], cwd=os.path.dirname(p))
        _assign_to_unity_job(_unity_proc)   # die with us even on a non-graceful exit
        print(f'[CompanionApp] launch: started player → {p} --port {_unity_port}', file=sys.stderr)
        return {'ok': True, 'launched': True}
    except Exception as e:
        print(f'[CompanionApp] launch: failed: {e}', file=sys.stderr)
        return {'ok': False, 'error': str(e)}


def close_player() -> None:
    """Promptly terminate the player WE launched when the app closes — regardless of the
    autostart setting. (The Job Object already guarantees it dies with us; this just makes a
    graceful close immediate.) No-op for a player the user started themselves: we never tracked
    it, never put it in our job, so we never kill it."""
    global _unity_proc
    proc, _unity_proc = _unity_proc, None
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print('[CompanionApp] stopped the player we launched.', file=sys.stderr)
    except Exception as e:
        print(f'[CompanionApp] failed to stop player: {e}', file=sys.stderr)


# Single-instance guard (the Unity player has its own). A Windows named mutex held for
# the process lifetime; the OS releases it on exit, however the process dies.
_SINGLE_INSTANCE_MUTEX = 'Local\\WaifuCode.App.SingleInstance'
_single_instance_handle = None


def acquire_single_instance() -> bool:
    """Try to become THE app instance. Returns False when another instance already
    holds the mutex. Fail-open: if the mutex can't even be created, don't block
    startup over the guard itself."""
    global _single_instance_handle
    if sys.platform != 'win32':
        return True
    try:
        import ctypes
        ERROR_ALREADY_EXISTS = 183
        handle = ctypes.windll.kernel32.CreateMutexW(None, False, _SINGLE_INSTANCE_MUTEX)
        if not handle:
            return True
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            ctypes.windll.kernel32.CloseHandle(handle)
            return False
        _single_instance_handle = handle  # keep alive; released by the OS on exit
        return True
    except Exception as e:
        print(f'[CompanionApp] single-instance guard failed: {e}', file=sys.stderr)
        return True


def focus_existing_instance() -> None:
    """Bring the already-running instance's chat window to the foreground so a second
    launch feels like 'open the app' rather than silently doing nothing."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, 'Waifu Code')
        if hwnd:
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
    except Exception as e:
        print(f'[CompanionApp] could not focus the running instance: {e}', file=sys.stderr)


def main() -> int:
    global _unity_exe_path, _unity_port
    if not acquire_single_instance():
        print('[CompanionApp] another instance is already running — exiting.', file=sys.stderr)
        focus_existing_instance()
        return 0
    unity_cfg = _CONFIG.get('unity') if isinstance(_CONFIG.get('unity'), dict) else {}
    _unity_port = int(unity_cfg.get('port') or _DEFAULT_UNITY_PORT)
    autostart = bool(unity_cfg.get('autostart', False))
    # Default to the player sitting beside us (packaged-release layout). Resolved against
    # APP_ROOT so a relative path works wherever the release folder is moved.
    exe_path = str(unity_cfg.get('exePath') or 'Waifu Code Player.exe')
    _unity_exe_path = os.path.abspath(exe_path if os.path.isabs(exe_path) else str(APP_ROOT / exe_path))
    # Any player WE launch (autostart now, or the overlay button later) is bound to a kill-on-
    # close job and closed with the app — regardless of this autostart flag.
    if autostart:
        launch_player()

    bridge = Bridge(unity_port=_unity_port)
    bridge.start_ws_thread()

    # Install the chat window up-front. pywebview needs at least one window registered
    # before webview.start(), and creating it visible at load time avoids WebView2's
    # background-throttling of JS timers.
    bridge.install_chat_window()
    # STT model load is deferred until the renderer fires notify_chat_ready (see
    # Bridge.on_chat_ready). Loading on the bridge thread would race the renderer's
    # mount and freeze the UI before it even appears.

    def shutdown():
        close_player()   # close the player we launched (job object handles non-graceful exits)
        bridge.stop()

    # webview.start() blocks until the last window is destroyed. Our close-intercept
    # on the chat window keeps it alive (it minimizes instead of closing), so start()
    # returns only when the process is killed by Unity (or the user). Maps to Electron's
    # `app.on('before-quit')`.
    try:
        webview.start(gui=None)
    finally:
        shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
