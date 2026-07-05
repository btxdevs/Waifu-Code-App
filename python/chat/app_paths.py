"""Single source of truth for the application's base directory.

Every read-only resource (the built UI under dist/, system_prompt.txt, the *.config.json
files) and every writable user-data directory (models/, characters/, saves/,
memory/) lives under one root. In a source checkout that root is CompanionApp/; in a
PyInstaller onedir build it's the folder that contains the .exe — the build script copies
the static resources next to the .exe so the same relative layout holds either way.

Resolving everything through APP_ROOT (instead of each module walking __file__ parents)
is what lets the frozen build find its files: once frozen, module __file__ points inside
the bundle's _internal/ dir, not next to the .exe.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _resolve_app_root() -> Path:
    # PyInstaller sets sys.frozen and points sys.executable at the launcher .exe. For a
    # onedir build the user-facing app root is the directory holding that .exe.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Source layout: this file is CompanionApp/python/chat/app_paths.py, so the app root
    # (CompanionApp/) is parents[2] — chat → python → CompanionApp.
    return Path(__file__).resolve().parents[2]


APP_ROOT = _resolve_app_root()
