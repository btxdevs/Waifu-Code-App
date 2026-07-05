"""Workspace sandbox for filesystem-touching tools.

Port of Assets/Scripts/ChatTools/Workspace/WorkspaceConfig.cs + WorkspacePath.cs.
The single rule: every absolute path resolved from LLM input must live inside one
of the configured roots, or the tool refuses. The first root is also used as the
cwd for relative paths.

Loaded from app.config.json under the top-level `workspace` key. Falls
back to a default that allows the project root.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


from ..app_paths import APP_ROOT

# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
_CONFIG_PATH = APP_ROOT / "app.config.json"


@dataclass
class WorkspaceConfig:
    """Filesystem sandbox + a few execution caps. Mirrors the C# ScriptableObject."""
    allowed_roots: list[str] = field(default_factory=list)
    max_read_file_bytes: int = 5 * 1024 * 1024
    max_glob_results: int = 500
    max_grep_results: int = 500
    default_exec_timeout_seconds: int = 15
    ignored_directory_names: list[str] = field(default_factory=lambda: [
        ".git", ".svn", ".hg",
        "node_modules", ".venv", "venv", "__pycache__",
        "Library", "Temp", "obj", "bin",
        ".idea", ".vs", ".vscode",
    ])
    follow_symlinks: bool = False
    allowed_command_prefixes: list[str] = field(default_factory=list)
    denied_command_prefixes: list[str] = field(default_factory=list)
    # Full-permission mode: when True the sandbox is bypassed — any resolvable path is allowed,
    # in or out of the configured roots. Paired with ApprovalGate.full_access (no prompts).
    full_access: bool = False


def load_workspace_config(path: str | os.PathLike | None = None) -> WorkspaceConfig:
    """Reads the `workspace` block out of app.config.json. Missing file or
    missing block → defaults that allow the parent of CompanionApp/ (the Unity
    project root) so the tools have somewhere to live out of the box."""
    p = Path(path) if path else _CONFIG_PATH
    data: dict = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data = raw.get("workspace") if isinstance(raw.get("workspace"), dict) else {}
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as e:
        print(f"[workspace] failed to read {p}: {e}; using defaults", file=sys.stderr)
        data = {}

    cfg = WorkspaceConfig()
    roots = data.get("allowedRoots")
    if isinstance(roots, list) and roots:
        cfg.allowed_roots = [str(r) for r in roots if isinstance(r, str) and r]
    if not cfg.allowed_roots:
        # Sensible default: the Unity project root (parent of CompanionApp/).
        cfg.allowed_roots = [str(p.parent.parent.resolve())]

    if isinstance(data.get("maxReadFileBytes"), int):
        cfg.max_read_file_bytes = int(data["maxReadFileBytes"])
    if isinstance(data.get("maxGlobResults"), int):
        cfg.max_glob_results = int(data["maxGlobResults"])
    if isinstance(data.get("maxGrepResults"), int):
        cfg.max_grep_results = int(data["maxGrepResults"])
    if isinstance(data.get("defaultExecTimeoutSeconds"), int):
        cfg.default_exec_timeout_seconds = int(data["defaultExecTimeoutSeconds"])
    ig = data.get("ignoredDirectoryNames")
    if isinstance(ig, list):
        cfg.ignored_directory_names = [str(n) for n in ig if isinstance(n, str) and n]
    if isinstance(data.get("followSymlinks"), bool):
        cfg.follow_symlinks = bool(data["followSymlinks"])
    allow_p = data.get("allowedCommandPrefixes")
    if isinstance(allow_p, list):
        cfg.allowed_command_prefixes = [str(n) for n in allow_p if isinstance(n, str) and n]
    deny_p = data.get("deniedCommandPrefixes")
    if isinstance(deny_p, list):
        cfg.denied_command_prefixes = [str(n) for n in deny_p if isinstance(n, str) and n]
    if isinstance(data.get("fullAccess"), bool):
        cfg.full_access = bool(data["fullAccess"])
    return cfg


@dataclass
class PathResolution:
    ok: bool
    absolute_path: str = ""
    root: str = ""
    error: str = ""


def resolve_path(cfg: WorkspaceConfig, user_input: str | None) -> PathResolution:
    """Resolves an LLM-supplied path against the workspace. Relative paths are
    resolved against the first allowed root. Returns ok=False with a
    human-readable error on rejection — never raises."""
    if cfg is None:
        return PathResolution(False, error="workspace config is not assigned to this tool — refusing.")
    if not cfg.allowed_roots and not cfg.full_access:
        return PathResolution(False, error="workspace has no allowed roots configured — refusing.")
    if not user_input or not user_input.strip():
        return PathResolution(False, error="path is empty.")

    raw = user_input.strip()
    # ~ expansion — convenience for prompts that emit unix-style home paths.
    if raw == "~" or raw.startswith("~/") or raw.startswith("~\\"):
        home = os.path.expanduser("~")
        raw = home if len(raw) <= 2 else os.path.join(home, raw[2:])

    # Relative paths resolve against the first root, or the cwd in full-access mode with no roots.
    base_dir = _normalize_root(cfg.allowed_roots[0]) if cfg.allowed_roots else os.getcwd()
    combined = raw if os.path.isabs(raw) else os.path.join(base_dir, raw)

    try:
        absolute = os.path.abspath(combined)
    except (OSError, ValueError) as e:
        return PathResolution(False, error=f"could not resolve path: {e}")

    # Full-permission mode: accept any resolvable path, in or out of the roots.
    if cfg.full_access:
        return PathResolution(True, absolute_path=absolute, root=base_dir)

    for raw_root in cfg.allowed_roots:
        root = _normalize_root(raw_root)
        if not root:
            continue
        if path_starts_with(absolute, root):
            return PathResolution(True, absolute_path=absolute, root=root)

    # The tool-result spill dir is always readable regardless of the configured
    # roots — oversized outputs are parked there with a "Read this path" pointer,
    # and per-chat root overrides must not cut the model off from them.
    from .spill import spill_dir
    sd = spill_dir()
    if path_starts_with(absolute, sd):
        return PathResolution(True, absolute_path=absolute, root=sd)

    return PathResolution(False, error=f"path is outside the allowed workspace roots: {absolute}")


def suggest_similar(abs_path: str) -> str | None:
    """Closest-named sibling of a missing file, for "Did you mean X?" hints on
    not-found errors (converts a dead-end into a self-correction — typo'd or
    stale filenames are the common case). None when the parent doesn't exist,
    isn't listable, or nothing is close enough."""
    import difflib
    parent = os.path.dirname(abs_path)
    name = os.path.basename(abs_path)
    if not name or not parent or not os.path.isdir(parent):
        return None
    try:
        siblings = os.listdir(parent)
    except OSError:
        return None
    match = difflib.get_close_matches(name, siblings, n=1, cutoff=0.6)
    return os.path.join(parent, match[0]) if match else None


def path_starts_with(absolute: str, root: str) -> bool:
    """True if `absolute` sits at or under `root`. Case-insensitive on Windows."""
    if not absolute or not root:
        return False
    a = _trim_trailing_sep(absolute)
    r = _trim_trailing_sep(root)
    # Case-insensitive on Windows/macOS; ordinal would be technically right on Linux
    # but pragmatically we ignore case across the board to match the C# helper.
    cf = (lambda s: s.casefold())
    if cf(a).startswith(cf(r)):
        if len(a) == len(r):
            return True
        next_ch = a[len(r)]
        return next_ch in (os.sep, os.altsep or os.sep)
    return False


def default_root(cfg: WorkspaceConfig) -> str | None:
    if cfg is None or not cfg.allowed_roots:
        return None
    return _normalize_root(cfg.allowed_roots[0])


def _normalize_root(root: str) -> str:
    if not root or not root.strip():
        return ""
    try:
        return _trim_trailing_sep(os.path.abspath(root.strip()))
    except (OSError, ValueError):
        return _trim_trailing_sep(root.strip())


def _trim_trailing_sep(s: str) -> str:
    if not s:
        return s
    end = len(s)
    while end > 0 and s[end - 1] in (os.sep, os.altsep or os.sep):
        end -= 1
    return s if end == len(s) else s[:end]
