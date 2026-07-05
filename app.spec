# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build of the CompanionApp (pywebview host: python/app.py).

Produces a folder build: <distpath>/WaifuCodeApp/ with `Waifu Code App.exe` + an
_internal/ dir of Python + native deps. The build script (build-app.bat) then
copies the static resources (dist/ UI, system_prompt.txt, the *.config.json files,
vendor/ripgrep) NEXT TO the exe so APP_ROOT (= the exe folder when frozen) resolves
them — see chat/app_paths.py.

Models are NOT bundled: STT (sherpa-onnx) and TTS (pocket-tts-onnx, fetched as an HF
snapshot at runtime) download into a models/ folder beside the exe on first run. torch /
torchaudio are excluded — they are dead weight from the unused `silero-vad` pip package;
the real VAD is silero_vad.onnx via sherpa-onnx.
"""
import os
from PyInstaller.utils.hooks import collect_all, collect_submodules

PYTHON_DIR = os.path.join(SPECPATH, "python")

datas, binaries, hiddenimports = [], [], []

# Packages with native libs / data files / dynamically-loaded submodules that the import
# tracer can't fully follow on its own. collect_all grabs their modules + binaries + data.
_COLLECT = [
    "webview",               # pywebview + WebView2 loader DLLs (Windows EdgeChromium backend)
    "clr_loader",            # .NET runtime shim used by the EdgeChromium backend
    "pythonnet",
    "sherpa_onnx",           # STT engine — native .pyd + dlls
    "onnxruntime",           # ONNX runtime — native dlls
    "rapidocr_onnxruntime",  # OCR — bundled onnx models + yaml config
    "sounddevice",           # PortAudio binaries
    "soundfile",             # libsndfile
    "sentencepiece",
    "safetensors",          # used by the vendored pocket_tts_onnx engine
    "scipy",
    "huggingface_hub",
    "ddgs",
    "markdownify",
    "bs4",
    "certifi",
]
for _pkg in _COLLECT:
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# Tool plugins are imported inside a factory function in chat/tools/__init__.py; collect
# them explicitly so none are missed.
hiddenimports += collect_submodules("chat.tools")
# The vendored pocket-TTS engine is imported lazily inside tts.ensure_loaded(); pin it so it
# (and its statically-traced deps: wave, sentencepiece, safetensors, soundfile, scipy) ship.
hiddenimports += ["pocket_tts_onnx"]

a = Analysis(
    [os.path.join(PYTHON_DIR, "app.py")],
    pathex=[PYTHON_DIR],            # resolve top-level local modules: image_proc, ocr, stt, tts, chat
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchaudio", "silero_vad", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Waifu Code App",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # GUI app — app.py redirects stdout/stderr to app.log
    icon=os.path.join(SPECPATH, "app.ico"),  # generated from app_icon.png — exe + window/taskbar icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WaifuCodeApp",
)
