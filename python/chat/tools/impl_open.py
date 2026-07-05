"""Open tool — port of Assets/Scripts/ChatTools/OpenFileToolExecutor.cs.

Opens a file with the OS default program. Blocks executables outright. Files
outside the workspace require an approval modal.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .workspace import default_root, path_starts_with


_BLOCKED_EXECUTABLE_EXTS = {
    ".exe", ".bat", ".cmd", ".com", ".msi", ".ps1", ".vbs", ".wsf",
}


class OpenTool(ToolExecutor):
    name = "Open"
    permission = ToolPermission.READ_ONLY
    activity_label = "Opening file…"
    defer_until_speech_caught_up = False
    description = (
        "Open a file with the default OS program (image viewer, text editor, etc.). "
        "Works like double-clicking in File Explorer. Use for images, documents, videos, "
        "or any file the user wants to view or edit in its native application."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Path to the file to open. Relative paths resolve against the workspace. "
                        "Absolute paths outside the workspace require approval."
                    ),
                },
            },
            "required": ["path"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult(result_text="Error: 'path' is required.", error="bad args")

        # Resolve manually so we can handle out-of-workspace paths.
        raw = raw_path.strip()
        if raw == "~" or raw.startswith("~/") or raw.startswith("~\\"):
            home = os.path.expanduser("~")
            raw = home if len(raw) <= 2 else os.path.join(home, raw[2:])
        if os.path.isabs(raw):
            absolute = os.path.abspath(raw)
        else:
            root = default_root(ctx.workspace) or os.getcwd()
            absolute = os.path.abspath(os.path.join(root, raw))

        if not os.path.exists(absolute):
            return ToolResult(result_text=f"Error: File does not exist: {absolute}", error="not found")
        ext = os.path.splitext(absolute)[1].lower()
        if ext in _BLOCKED_EXECUTABLE_EXTS:
            return ToolResult(
                result_text=f"Error: refusing to open executable file: {absolute}",
                error="executable blocked",
            )

        in_workspace = _is_in_workspace(absolute, ctx)
        if not in_workspace:
            decision = await ctx.approval.request(
                ApprovalRequest(
                    tool_name=self.name,
                    summary=f"Open file outside workspace: {absolute}",
                    details={"path": absolute, "outsideWorkspace": True},
                    risk_level="danger",
                ),
                scope_key=absolute,
            )
            if not decision.allow:
                return ToolResult(
                    result_text=f"Error: user declined to open {absolute}.",
                    error="user-declined",
                )

        try:
            _open_with_default(absolute)
        except OSError as e:
            return ToolResult(result_text=f"Error: failed to open: {e}", error=str(e))
        return ToolResult(result_text=f"Opened {absolute} with the default application.")


def _is_in_workspace(absolute: str, ctx: ToolContext) -> bool:
    for root in (ctx.workspace.allowed_roots or []):
        if path_starts_with(absolute, root):
            return True
    return False


def _open_with_default(path: str) -> None:
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
