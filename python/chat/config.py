"""Loads `CompanionApp/llm.config.json` — the replacement for the Unity-side
BackendConfig ScriptableObject. Separate file (not folded into app.config.json)
because it carries the LLM API key, which usually wants different ownership / git rules
than the rest of the app settings.

The file holds MULTIPLE named LLM configs and a selected default. New schema:

  {
    "configs": [
      {
        "id":          "default",            // stable id (referenced per-chat)
        "name":        "Default",            // human label shown in the pickers
        "api_url":     "https://api.deepseek.com/chat/completions",
        "api_key":     "sk-...",
        "model":       "deepseek-chat",
        "temperature": 1,
        "request_timeout_seconds": 30,
        "thinking":    "unset",              // "unset" | "disabled" | "enabled"
        "send_system_prompt_as_user": false,
        "supports_vision": false,
        "max_tool_call_rounds": 0,           // 0 = uncapped
        "vision_max_image_edge_pixels": 2000,
        "vision_jpeg_quality": 85,
        "vision_max_images": 10              // most-recent image messages kept per origin (0 = none)
      }
    ],
    "default_id": "default"
  }

Backward compatibility: a legacy FLAT file (the LLM fields at the top level, no
"configs" key) is read as a single config with id "default" / name "Default".

The file lives next to app.config.json in CompanionApp/. Falls back to defaults
if missing; logs and falls back to defaults on malformed JSON. Never raises.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .llm_client import LlmConfig


from .app_paths import APP_ROOT

# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
_DEFAULT_CONFIG_PATH = APP_ROOT / "llm.config.json"

# Stable id used for the single config a legacy flat file migrates to.
DEFAULT_CONFIG_ID = "default"
DEFAULT_CONFIG_NAME = "Default"


@dataclass
class ChatBackendConfig:
    """Everything the orchestrator + LlmClient need at runtime. Mirrors the
    Unity-side BackendConfig ScriptableObject 1:1."""
    llm: LlmConfig = field(default_factory=LlmConfig)
    # 0 = uncapped (matches the C# default). Positive = runaway-loop guard.
    max_tool_call_rounds: int = 0
    # Vision settings — only consulted when llm.supports_vision is true.
    vision_max_image_edge_pixels: int = 2000
    vision_jpeg_quality: int = 85
    # Display name of the human user. Substituted into {{user}} tokens in
    # character prompts. Sourced from app.config.json (top-level "user_name")
    # by app.py at startup — NOT loaded by chat/config.py. Defaulted here
    # so headless tests that don't touch app.config still see a sane value.
    user_name: str = "User"


@dataclass
class LlmConfigEntry:
    """One named LLM config in the registry: a stable id, a display name, and the
    resolved ChatBackendConfig the orchestrator/LlmClient consume when it's active."""
    id: str
    name: str
    config: ChatBackendConfig


@dataclass
class LlmConfigRegistry:
    """All LLM configs loaded from llm.config.json plus which one is the default.
    A chat references a config by id; an unknown / empty id falls back to the default."""
    entries: list[LlmConfigEntry] = field(default_factory=list)
    default_id: str = DEFAULT_CONFIG_ID

    def get(self, config_id: str | None) -> LlmConfigEntry | None:
        """The entry with this id, or None when the id is empty/unknown."""
        if not config_id:
            return None
        for e in self.entries:
            if e.id == config_id:
                return e
        return None

    def default(self) -> LlmConfigEntry | None:
        """The default entry (by default_id), falling back to the first entry, or None
        when the registry is empty."""
        return self.get(self.default_id) or (self.entries[0] if self.entries else None)


def load_registry(path: str | os.PathLike | None = None) -> LlmConfigRegistry:
    """Reads the config file into a registry of named configs. Handles both the new
    `{configs: [...], default_id}` schema and the legacy flat single-config file. Never
    raises — missing/malformed files degrade to a single empty default config."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    raw: dict = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            raw = loaded
        else:
            print(f"[chat.config] {p} root must be an object; using defaults", file=sys.stderr)
    except FileNotFoundError:
        # Defaults are fine; the orchestrator will fail at the first LLM call with a
        # clear "api_key empty" message rather than crashing on startup.
        pass
    except (OSError, json.JSONDecodeError) as e:
        print(f"[chat.config] failed to read {p}: {e}; using defaults", file=sys.stderr)

    entries: list[LlmConfigEntry] = []
    configs_raw = raw.get("configs")
    if isinstance(configs_raw, list) and configs_raw:
        seen_ids: set[str] = set()
        for i, item in enumerate(configs_raw):
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "").strip() or f"{DEFAULT_CONFIG_ID}_{i}"
            # Guard against duplicate ids so per-chat references resolve unambiguously.
            while cid in seen_ids:
                cid = f"{cid}_{i}"
            seen_ids.add(cid)
            name = str(item.get("name") or "").strip() or cid
            entries.append(LlmConfigEntry(id=cid, name=name, config=_config_from_dict(item)))
        default_id = str(raw.get("default_id") or "").strip()
    else:
        # Legacy flat file (or empty): treat the whole object as a single config.
        entries.append(LlmConfigEntry(
            id=DEFAULT_CONFIG_ID, name=DEFAULT_CONFIG_NAME, config=_config_from_dict(raw)))
        default_id = DEFAULT_CONFIG_ID

    if not entries:
        entries.append(LlmConfigEntry(
            id=DEFAULT_CONFIG_ID, name=DEFAULT_CONFIG_NAME, config=ChatBackendConfig()))

    # Make sure default_id points at a real entry.
    if not any(e.id == default_id for e in entries):
        default_id = entries[0].id

    return LlmConfigRegistry(entries=entries, default_id=default_id)


def _config_from_dict(data: dict) -> ChatBackendConfig:
    """Builds a ChatBackendConfig from one flat config dict (the per-config fields).
    Missing fields fall back to the dataclass defaults (matching the old BackendConfig)."""
    if not isinstance(data, dict):
        data = {}
    llm = LlmConfig(
        api_url=str(data.get("api_url", LlmConfig.api_url)) if "api_url" in data else LlmConfig.api_url,
        api_key=str(data.get("api_key", "") or ""),
        model=str(data.get("model", LlmConfig.model)) if "model" in data else LlmConfig.model,
        temperature=float(data.get("temperature", LlmConfig.temperature)),
        request_timeout_seconds=int(data.get("request_timeout_seconds", LlmConfig.request_timeout_seconds)),
        thinking=_clean_thinking(data.get("thinking")),
        send_system_prompt_as_user=bool(data.get("send_system_prompt_as_user", False)),
        supports_vision=bool(data.get("supports_vision", False)),
        vision_max_images=int(data.get("vision_max_images", 10)),
    )

    return ChatBackendConfig(
        llm=llm,
        max_tool_call_rounds=int(data.get("max_tool_call_rounds", 0)),
        vision_max_image_edge_pixels=int(data.get("vision_max_image_edge_pixels", 2000)),
        vision_jpeg_quality=int(data.get("vision_jpeg_quality", 85)),
        # user_name intentionally NOT loaded here; lives in app.config.json.
        # app.py overlays it onto the active config after this call.
    )


def _clean_thinking(value) -> str:
    if not isinstance(value, str):
        return "unset"
    v = value.strip().lower()
    return v if v in ("unset", "disabled", "enabled") else "unset"
