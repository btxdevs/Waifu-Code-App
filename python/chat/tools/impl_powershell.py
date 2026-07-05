"""PowerShell tool — port of Assets/Scripts/ChatTools/PowerShellToolExecutor.cs.

Same three-layer model as Bash. Edition detection (pwsh vs powershell.exe)
influences only the trailing description line; both interpreters run with
`-NoProfile -NonInteractive -Command <command>`.
"""
from __future__ import annotations

import os

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .shell import (
    ShellClassification, classify_command, format_output, is_dangerous_powershell,
    resolve_powershell, run_command,
    DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS,
)
from .workspace import default_root, resolve_path


_BASE_DESCRIPTION = (
    "Executes a given PowerShell command with optional timeout. Working directory persists between commands; shell state (variables, functions) does not.\n\n"
    "IMPORTANT: This tool is for terminal operations via PowerShell: git, npm, docker, and PS cmdlets. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.\n\n"
    "PowerShell Syntax Notes:\n"
    "   - Variables use $ prefix: $myVar = \"value\"\n"
    "   - Escape character is backtick (`), not backslash\n"
    "   - Use Verb-Noun cmdlet naming: Get-ChildItem, Set-Location, New-Item, Remove-Item\n"
    "   - Common aliases: ls (Get-ChildItem), cd (Set-Location), cat (Get-Content), rm (Remove-Item)\n"
    "   - Pipe operator | works similarly to bash but passes objects, not text\n"
    "   - Use Select-Object, Where-Object, ForEach-Object for filtering and transformation\n"
    "   - String interpolation: \"Hello $name\" or \"Hello $($obj.Property)\"\n"
    "   - Registry access uses PSDrive prefixes: `HKLM:\\SOFTWARE\\...`, `HKCU:\\...` — NOT raw `HKEY_LOCAL_MACHINE\\...`\n"
    "   - Environment variables: read with `$env:NAME`, set with `$env:NAME = \"value\"` (NOT `Set-Variable` or bash `export`)\n"
    "   - Call native exe with spaces in path via call operator: `& \"C:\\Program Files\\App\\app.exe\" arg1 arg2`\n\n"
    "Interactive and blocking commands (will hang — this tool runs with -NonInteractive):\n"
    "   - NEVER use `Read-Host`, `Get-Credential`, `Out-GridView`, `$Host.UI.PromptForChoice`, or `pause`\n"
    "   - Destructive cmdlets (`Remove-Item`, `Stop-Process`, `Clear-Content`, etc.) may prompt for confirmation. Add `-Confirm:$false` when you intend the action to proceed. Use `-Force` for read-only/hidden items.\n"
    "   - Never use `git rebase -i`, `git add -i`, or other commands that open an interactive editor\n\n"
    "Usage notes:\n"
    "  - The command argument is required.\n"
    f"  - You can specify an optional timeout in seconds (up to {MAX_TIMEOUT_SECONDS}s). If not specified, commands will timeout after {DEFAULT_TIMEOUT_SECONDS}s. On timeout the whole process tree is killed.\n"
    "  - If the output exceeds 30,000 characters, the START is truncated (the most recent output is kept) with a [N lines truncated from start] marker.\n"
    "  - You can use the `run_in_background` parameter to run long commands (builds, tests, batch jobs) in the background: you keep chatting while it runs, output IS captured, and the result comes back to you later as a [System Message] you react to. Background commands are killed when the chat session ends. To launch a GUI app that should stay open, use `Start-Process` in a normal (foreground) call instead.\n"
    "  - Avoid using PowerShell to run commands that have dedicated tools, unless explicitly instructed:\n"
    "    - File search: Use Glob (NOT Get-ChildItem -Recurse)\n"
    "    - Content search: Use Grep (NOT Select-String)\n"
    "    - Read files: Use Read (NOT Get-Content)\n"
    "    - Edit files: Use Edit\n"
    "    - Write files: Use Write (NOT Set-Content/Out-File)\n"
    "    - Communication: Output text directly (NOT Write-Output/Write-Host)\n"
    "  - When issuing multiple commands:\n"
    "    - If the commands are independent and can run in parallel, make multiple PowerShell tool calls in a single message.\n"
    "    - If the commands depend on each other and must run sequentially, chain them in a single PowerShell call (see edition-specific chaining syntax above).\n"
    "    - Use `;` only when you need to run commands sequentially but don't care if earlier commands fail.\n"
    "    - DO NOT use newlines to separate commands (newlines are ok in quoted strings and here-strings)\n"
    "  - Do NOT prefix commands with `cd` or `Set-Location` -- the working directory is already set to the correct project directory automatically.\n"
    "  - For git commands:\n"
    "    - Prefer to create a new commit rather than amending an existing commit.\n"
    "    - Before running destructive operations, consider whether there is a safer alternative.\n"
    "    - Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign) unless the user has explicitly asked for it."
)


class PowerShellTool(ToolExecutor):
    name = "PowerShell"
    permission = ToolPermission.DANGER_FULL_ACCESS
    activity_label = "Running PowerShell…"
    defer_until_speech_caught_up = False
    description = _BASE_DESCRIPTION

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The PowerShell command to execute"},
                "description": {"type": "string", "description": "Clear, concise description of what this command does in active voice."},
                "timeout": {
                    "type": "integer",
                    "description": f"Optional timeout in seconds (default {DEFAULT_TIMEOUT_SECONDS}s, max {MAX_TIMEOUT_SECONDS}s).",
                    "minimum": 1,
                },
                "cwd": {"type": "string", "description": "Optional working directory inside the workspace."},
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "If true, run the command as a background task: returns immediately, output is "
                        "captured, and the result comes back later as a [System Message] you react to. "
                        f"Use for builds/tests/long jobs (default timeout in this mode: {MAX_TIMEOUT_SECONDS}s). "
                        "NOT for GUI apps that should stay open — use Start-Process in a foreground call "
                        "for those. Background commands die with the chat session."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(result_text="Error: 'command' is required.", error="bad args")

        block_reason = is_dangerous_powershell(command)
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

        interp = resolve_powershell()
        if interp is None:
            return ToolResult(
                result_text="Error: PowerShell (pwsh / powershell.exe) is not available on PATH.",
                error="no powershell",
            )

        cwd_arg = arguments.get("cwd")
        if isinstance(cwd_arg, str) and cwd_arg.strip():
            res = resolve_path(ctx.workspace, cwd_arg)
            if not res.ok:
                return ToolResult(result_text="Error: " + res.error, error=res.error)
            cwd = res.absolute_path
        else:
            cwd = default_root(ctx.workspace) or os.getcwd()

        background = bool(arguments.get("run_in_background", False))
        timeout = arguments.get("timeout")
        try:
            # Background jobs are long by nature — without an explicit timeout they
            # get the generous cap instead of the 2-minute foreground default.
            fallback = MAX_TIMEOUT_SECONDS if background else DEFAULT_TIMEOUT_SECONDS
            timeout = int(timeout) if timeout is not None else fallback
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT_SECONDS

        if classification != ShellClassification.ALLOWED:
            decision = await ctx.approval.request(
                ApprovalRequest(
                    tool_name=self.name,
                    summary="Run PowerShell command",
                    details={
                        "shell": interp.edition or "powershell",
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

        argv = [interp.path, "-NoProfile", "-NonInteractive", "-Command", command]
        if background:
            # Route through the ChatManager's background-task registry (same
            # machinery as background UwU helpers): output captured, completion
            # folded back as a notification turn the character reacts to.
            start = getattr(ctx.chat_manager, "bg_shell_start", None)
            if start is None:
                return ToolResult(
                    result_text="Error: background commands are not available here.",
                    error="no bg runner",
                )
            desc = arguments.get("description")
            label = (str(desc).strip() if isinstance(desc, str) and desc.strip()
                     else (command[:48] + "…" if len(command) > 48 else command))
            tid = start(argv, cwd, timeout, label)
            return ToolResult(result_text=(
                f"Command started in the background (task {tid}, \"{label}\"). Output is being "
                f"captured; when it finishes, the result arrives as a [System Message] and you "
                f"react to it then. For now, just tell the user it's running. (CheckUwUHelpers "
                f"shows its progress; DismissUwUHelper cancels it.)"))
        result = await run_command(argv, cwd=cwd, timeout_seconds=timeout)
        return ToolResult(result_text=format_output(result))


def _normalize_for_scope(command: str) -> str:
    if not command:
        return ""
    parts = command.strip().split()
    return parts[0] if parts else ""
