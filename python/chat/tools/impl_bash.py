"""Bash tool — port of Assets/Scripts/ChatTools/BashToolExecutor.cs.

Three-layer model:
  1. Hard-block via is_dangerous_bash() (curl|sh, fork bomb, etc.)
  2. Workspace allow/deny prefix classification
  3. Approval modal (skipped when allow-listed)
"""
from __future__ import annotations

import os

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .shell import (
    ShellClassification, classify_command, format_output, is_dangerous_bash,
    resolve_bash, run_command, start_detached, DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS,
)
from .workspace import default_root, resolve_path


class BashTool(ToolExecutor):
    name = "Bash"
    permission = ToolPermission.DANGER_FULL_ACCESS
    activity_label = "Running shell command…"
    defer_until_speech_caught_up = False
    description = (
        "Executes a given bash command and returns its output.\n\n"
        "The working directory persists between commands, but shell state does not. The shell environment is initialized from the user's profile (bash or zsh).\n\n"
        "IMPORTANT: Avoid using this tool to run `find`, `grep`, `cat`, `head`, `tail`, `sed`, `awk`, or `echo` commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task. Instead, use the appropriate dedicated tool:\n"
        "- File search: Use Glob (NOT find or ls)\n"
        "- Content search: Use Grep (NOT grep or rg)\n"
        "- Read files: Use Read (NOT cat/head/tail)\n"
        "- Edit files: Use Edit (NOT sed/awk)\n"
        "- Write files: Use Write (NOT echo >/cat <<EOF)\n"
        "- Communication: Output text directly (NOT echo/printf)\n\n"
        "Usage:\n"
        "- If your command will create new directories or files, first run `ls` to verify the parent directory exists and is the correct location.\n"
        "- Always quote file paths that contain spaces with double quotes.\n"
        "- Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`. You may use `cd` if the User explicitly requests it.\n"
        f"- You may specify an optional timeout in seconds (up to {MAX_TIMEOUT_SECONDS}s). By default, your command will timeout after {DEFAULT_TIMEOUT_SECONDS}s.\n"
        "- If the output exceeds 30,000 characters, output will be truncated with a [N lines truncated] marker.\n"
        "- You can use the `run_in_background` parameter to run the command in the background. Only use this if you don't need the result immediately. Stdout/stderr are NOT captured in background mode.\n"
        "- When issuing multiple commands:\n"
        "  - If the commands are independent and can run in parallel, make multiple Bash tool calls in a single message.\n"
        "  - If the commands depend on each other and must run sequentially, chain them with `&&` in a single Bash call.\n"
        "  - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail.\n"
        "  - DO NOT use newlines to separate commands (newlines are ok in quoted strings).\n"
        "- For git commands:\n"
        "  - Prefer to create a new commit rather than amending an existing commit.\n"
        "  - Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative.\n"
        "  - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign) unless the user explicitly asks for it."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to execute"},
                "description": {"type": "string", "description": "Clear, concise description of what this command does in active voice."},
                "timeout": {
                    "type": "integer",
                    "description": f"Optional timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS}s, max {MAX_TIMEOUT_SECONDS}s).",
                    "minimum": 1,
                },
                "cwd": {"type": "string", "description": "Optional working directory inside the workspace. Defaults to the first allowed root."},
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "If true, spawn the command detached and return immediately with the PID. "
                        "Use for GUI scripts, dev servers, and anything that should outlive the tool call. "
                        "Stdout/stderr are NOT captured in this mode."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(result_text="Error: 'command' is required.", error="bad args")

        block_reason = is_dangerous_bash(command)
        if block_reason:
            return ToolResult(
                result_text=f"Error: command blocked by safety rule: {block_reason}",
                error="dangerous",
            )

        classification = classify_command(
            command,
            ctx.workspace.allowed_command_prefixes if ctx.workspace else [],
            ctx.workspace.denied_command_prefixes if ctx.workspace else [],
        )
        if classification == ShellClassification.DENIED:
            return ToolResult(
                result_text=f"Error: command blocked by workspace deny-list: {command}",
                error="denied",
            )

        interp = resolve_bash()
        if interp is None:
            return ToolResult(
                result_text="Error: bash is not available on PATH (and Git Bash wasn't found on Windows).",
                error="no bash",
            )

        cwd_arg = arguments.get("cwd")
        cwd = None
        if isinstance(cwd_arg, str) and cwd_arg.strip():
            res = resolve_path(ctx.workspace, cwd_arg)
            if not res.ok:
                return ToolResult(result_text="Error: " + res.error, error=res.error)
            cwd = res.absolute_path
        else:
            cwd = default_root(ctx.workspace) or os.getcwd()

        timeout = arguments.get("timeout")
        try:
            timeout = int(timeout) if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_SECONDS
        background = bool(arguments.get("run_in_background", False))

        if classification != ShellClassification.ALLOWED:
            decision = await ctx.approval.request(
                ApprovalRequest(
                    tool_name=self.name,
                    summary="Run shell command",
                    details={
                        "shell": "bash",
                        "command": command,
                        "cwd": cwd,
                        "background": background,
                        "timeoutSeconds": timeout,
                    },
                    risk_level="danger",
                ),
                scope_key=_normalize_for_scope(command),
            )
            if not decision.allow:
                return ToolResult(
                    result_text=f"Error: user declined to run: {command}",
                    error="user-declined",
                )

        argv = [interp.path, "-lc", command]
        if background:
            pid = start_detached(argv, cwd)
            if pid is None:
                return ToolResult(
                    result_text="Error: failed to spawn detached process.",
                    error="spawn failed",
                )
            return ToolResult(result_text=f"Started background process (pid={pid}).")
        result = await run_command(argv, cwd=cwd, timeout_seconds=timeout)
        return ToolResult(result_text=format_output(result))


def _normalize_for_scope(command: str) -> str:
    """Scope key for "approve for this session" — uses the first whitespace-token
    so `git status` and `git log` both stick on `git`."""
    if not command:
        return ""
    parts = command.strip().split()
    return parts[0] if parts else ""
