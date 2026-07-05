"""Write tool — port of Assets/Scripts/ChatTools/WriteFileToolExecutor.cs.

Overwrites or creates a file. Gates behind: must-have-read-first for existing
files, mtime staleness check, and an approval modal showing path + size + preview.
"""
from __future__ import annotations

import os

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .read_state import ReadStateEntry
from .workspace import resolve_path


class WriteTool(ToolExecutor):
    name = "Write"
    permission = ToolPermission.WORKSPACE_WRITE
    activity_label = "Writing file…"
    defer_until_speech_caught_up = False
    description = (
        "Writes a file to the local filesystem.\n\n"
        "Usage:\n"
        "- This tool will overwrite the existing file if there is one at the provided path.\n"
        "- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n"
        "- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.\n"
        "- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.\n"
        "- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to write (must be absolute, not relative)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["file_path", "content"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        raw_path = arguments.get("file_path")
        if not isinstance(raw_path, str):
            raw_path = ""
        resolution = resolve_path(ctx.workspace, raw_path)
        if not resolution.ok:
            return ToolResult(result_text="Error: " + resolution.error, error=resolution.error)
        abs_path = resolution.absolute_path

        if os.path.isdir(abs_path):
            return ToolResult(
                result_text=f"Error: EISDIR: path is a directory, not a file: {abs_path}",
                error="is a directory",
            )

        content = arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(result_text="Error: 'content' must be a string.", error="bad content")

        exists = os.path.exists(abs_path)
        if exists:
            err = _check_read_freshness(abs_path, content, ctx)
            if err is not None:
                return ToolResult(result_text="Error: " + err, error=err)

        size_kb = len(content.encode("utf-8", errors="replace")) / 1024.0
        preview = content if len(content) <= 200 else (content[:200] + "…")

        decision = await ctx.approval.request(
            ApprovalRequest(
                tool_name=self.name,
                summary=("Overwrite " if exists else "Create ") + abs_path,
                details={
                    "path": abs_path,
                    "exists": exists,
                    "sizeKb": round(size_kb, 1),
                    "preview": preview,
                },
            ),
            scope_key=abs_path,
        )
        if not decision.allow:
            return ToolResult(
                result_text=f"Error: user declined to write {abs_path}.",
                error="user-declined",
            )

        try:
            parent = os.path.dirname(abs_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(abs_path, "w", encoding="utf-8", newline="") as f:
                f.write(content)
        except OSError as e:
            return ToolResult(result_text=f"Error: write failed: {e}", error=str(e))

        ctx.read_state.set(abs_path, ReadStateEntry(
            content=content,
            timestamp=ctx.read_state.mtime_ticks(abs_path),
            offset=1,
            limit=None,
        ))
        return ToolResult(result_text=f"File written successfully to {abs_path}")


def _check_read_freshness(abs_path: str, new_content: str, ctx: ToolContext) -> str | None:
    """Returns None if the LLM may proceed, or an error string if it must read first."""
    prior = ctx.read_state.get(abs_path)
    if prior is None:
        return (
            "you must use the Read tool on this file before writing it. "
            "This protects against accidental overwrites of changes the LLM hasn't seen."
        )
    mtime = ctx.read_state.mtime_ticks(abs_path)
    if mtime != prior.timestamp:
        # Fallback for full reads: if the on-disk content equals what we stored,
        # the mtime change is benign (e.g., touch).
        if prior.content is not None:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    current = f.read()
                if current == prior.content:
                    return None
            except OSError:
                pass
        return (
            f"file {abs_path} was modified after the last Read in this conversation. "
            "Read it again before writing."
        )
    return None
