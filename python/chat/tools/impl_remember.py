"""Long-term memory tools: Remember / Forget / RecallMemory.

Two scopes (see chat/memory_store.py):
  * "character" — facts about this character or the user's history with them (per character id).
  * "project"   — facts about the current workspace/codebase (shared across characters; only
                  available when the chat has a workspace root).

The manager owns the stores and scope resolution (ChatManager.memory_add / memory_remove /
memory_get); these tools are thin wrappers. A compact index of every in-scope memory is already in
the system prompt each turn (plus the bodies most relevant to the message), so RecallMemory is only
for pulling a specific body whose one-line summary wasn't enough.
"""
from __future__ import annotations

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult

_SCOPES = ["character", "project"]


def _scope(arguments: dict) -> str:
    return str(arguments.get("scope") or "").strip().lower()


class RememberTool(ToolExecutor):
    name = "Remember"
    permission = ToolPermission.READ_ONLY   # app-local write; no workspace approval needed
    activity_label = "Saving a memory…"
    description = (
        "Save a durable fact to long-term memory so you recall it in future conversations. "
        "scope 'character' = about you or your history with the user; scope 'project' = about the "
        "current workspace/codebase (shared across characters). Save only things genuinely worth "
        "remembering later — preferences, decisions, ongoing context — not trivia or one-off chatter."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string", "enum": _SCOPES,
                    "description": "'character' for facts about you / your history with the user; "
                                   "'project' for facts about the current workspace (shared).",
                },
                "name": {
                    "type": "string",
                    "description": "A short title (a few words). Shown in your memory index and used "
                                   "to Forget / RecallMemory it later — make it distinctive.",
                },
                "description": {
                    "type": "string",
                    "description": "A specific one-line summary. This is the relevance key that decides "
                                   "when the full memory is surfaced, so make it concrete.",
                },
                "text": {"type": "string", "description": "The full fact to remember."},
            },
            "required": ["scope", "name", "text"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        scope = _scope(arguments)
        name = str(arguments.get("name") or "").strip()
        description = str(arguments.get("description") or "").strip()
        text = str(arguments.get("text") or "").strip()
        if scope not in _SCOPES:
            return ToolResult(result_text=f"Error: scope must be one of {_SCOPES}.", error="bad scope")
        if not name or not text:
            return ToolResult(result_text="Error: 'name' and 'text' are required.", error="bad args")
        ok, msg = ctx.chat_manager.memory_add(scope, ctx.session, name, description, text)
        return ToolResult(result_text=msg if ok else f"Error: {msg}", error=None if ok else msg)


class ForgetTool(ToolExecutor):
    name = "Forget"
    permission = ToolPermission.READ_ONLY
    activity_label = "Forgetting…"
    description = "Remove a saved memory by its name (as shown in your memory index)."

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": _SCOPES,
                          "description": "Which memory scope the entry lives in."},
                "name": {"type": "string", "description": "The exact name of the memory to remove."},
            },
            "required": ["scope", "name"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        scope = _scope(arguments)
        name = str(arguments.get("name") or "").strip()
        if scope not in _SCOPES:
            return ToolResult(result_text=f"Error: scope must be one of {_SCOPES}.", error="bad scope")
        if not name:
            return ToolResult(result_text="Error: 'name' is required.", error="bad args")
        ok, msg = ctx.chat_manager.memory_remove(scope, ctx.session, name)
        return ToolResult(result_text=msg if ok else f"Error: {msg}", error=None if ok else msg)


class RecallMemoryTool(ToolExecutor):
    name = "RecallMemory"
    permission = ToolPermission.READ_ONLY
    activity_label = "Recalling…"
    description = ("Read the full text of a saved memory by its name (from your memory index), when "
                   "the one-line summary in the index isn't enough.")

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": _SCOPES,
                          "description": "Which memory scope the entry lives in."},
                "name": {"type": "string", "description": "The exact name of the memory to read."},
            },
            "required": ["scope", "name"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        scope = _scope(arguments)
        name = str(arguments.get("name") or "").strip()
        if scope not in _SCOPES:
            return ToolResult(result_text=f"Error: scope must be one of {_SCOPES}.", error="bad scope")
        if not name:
            return ToolResult(result_text="Error: 'name' is required.", error="bad args")
        entry = ctx.chat_manager.memory_get(scope, ctx.session, name)
        if entry is None:
            return ToolResult(result_text=f"No {scope} memory named '{name}'.", error="not found")
        return ToolResult(result_text=f"{entry.name}: {entry.text}")
