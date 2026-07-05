"""Edit tool — port of Assets/Scripts/ChatTools/EditFileToolExecutor.cs.

Exact-string replacement in an existing file. Gates behind:
  * must-have-read-first
  * mtime staleness check
  * old_string occurrence count (must be unique unless replace_all)
  * approval modal showing path + occurrence count + diff preview

Includes the curly-quote fallback: if `old_string` doesn't match, try again with
both directions of quote normalization to handle LLMs that round-tripped through
smart-quote text. When matched on the fallback path, the new_string is
re-quote-styled to match what's actually in the file.
"""
from __future__ import annotations

import os
import re

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .read_state import ReadStateEntry
from .workspace import resolve_path, suggest_similar


class EditTool(ToolExecutor):
    name = "Edit"
    permission = ToolPermission.WORKSPACE_WRITE
    activity_label = "Editing file…"
    defer_until_speech_caught_up = False
    description = (
        "Performs exact string replacements in files.\n\n"
        "Usage:\n"
        "- You must use your `Read` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.\n"
        "- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + tab. Everything after that is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.\n"
        "- ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.\n"
        "- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.\n"
        "- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.\n"
        "- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to modify",
                },
                "old_string": {
                    "type": "string",
                    "description": "The text to replace",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with (must be different from old_string)",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences of old_string (default false)",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        raw_path = arguments.get("file_path")
        if not isinstance(raw_path, str):
            raw_path = ""
        resolution = resolve_path(ctx.workspace, raw_path)
        if not resolution.ok:
            return ToolResult(result_text="Error: " + resolution.error, error=resolution.error)
        abs_path = resolution.absolute_path

        if abs_path.lower().endswith(".ipynb"):
            return ToolResult(
                result_text="Error: editing Jupyter notebook (.ipynb) files isn't supported. Use a dedicated notebook tool.",
                error="ipynb-rejected",
            )

        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        replace_all = bool(arguments.get("replace_all", False))
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return ToolResult(
                result_text="Error: 'old_string' and 'new_string' must both be strings.",
                error="bad args",
            )
        if old_string == new_string:
            return ToolResult(
                result_text="Error: 'old_string' and 'new_string' must differ.",
                error="no-op edit",
            )

        exists = os.path.exists(abs_path)
        # Allow the file-creation-via-Edit fallback: empty old_string + file doesn't
        # exist → create.
        if not exists and old_string == "":
            return await self._create_via_edit(abs_path, new_string, ctx)
        if not exists:
            hint = suggest_similar(abs_path)
            return ToolResult(
                result_text=(f"Error: File does not exist: {abs_path}"
                             + (f" Did you mean: {hint}?" if hint else "")),
                error="not found",
            )
        if old_string == "":
            return ToolResult(
                result_text="Error: 'old_string' is empty but the file already exists. Use Write to overwrite the whole file.",
                error="empty old_string",
            )

        err = _check_read_freshness(abs_path, ctx)
        if err is not None:
            return ToolResult(result_text="Error: " + err, error=err)

        try:
            # newline="" = no translation, so we can see the file's real line endings.
            with open(abs_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                raw_content = f.read()
        except OSError as e:
            return ToolResult(result_text=f"Error: could not read file: {e}", error=str(e))

        # Preserve the file's line endings. Matching must happen in \n-space (the
        # Read output the model copied old_string from is \n-normalized), so
        # normalize for the replacement and restore CRLF on write. Without this,
        # every Edit rewrote a CRLF file wholesale to LF (whole-file git diffs).
        # Mixed-ending files come out uniformly CRLF — acceptable normalization.
        uses_crlf = "\r\n" in raw_content
        file_content = raw_content.replace("\r\n", "\n") if uses_crlf else raw_content

        actual_old, actual_new, count = _find_actual_string(file_content, old_string, new_string)
        if count == 0:
            return ToolResult(
                result_text="Error: 'old_string' was not found in the file. Check whitespace, indentation, and surrounding context.",
                error="not found in file",
            )
        if count > 1 and not replace_all:
            return ToolResult(
                result_text=(
                    f"Error: 'old_string' was found {count} times in the file. "
                    "Either provide a longer 'old_string' that's unique, or pass replace_all=true to replace every occurrence."
                ),
                error="not unique",
            )

        decision = await ctx.approval.request(
            ApprovalRequest(
                tool_name=self.name,
                summary=f"Edit {abs_path}" + (f" — {count} occurrences" if count > 1 else ""),
                details={
                    "path": abs_path,
                    "occurrences": count,
                    "replaceAll": replace_all,
                    "preview": {"old": actual_old[:200], "new": actual_new[:200]},
                },
            ),
            scope_key=abs_path,
        )
        if not decision.allow:
            return ToolResult(
                result_text=f"Error: user declined to edit {abs_path}.",
                error="user-declined",
            )

        if replace_all:
            new_content = file_content.replace(actual_old, actual_new)
        else:
            new_content = file_content.replace(actual_old, actual_new, 1)

        # Restore the original line endings on disk (see the normalization above).
        disk_content = new_content.replace("\n", "\r\n") if uses_crlf else new_content
        try:
            with open(abs_path, "w", encoding="utf-8", newline="") as f:
                f.write(disk_content)
        except OSError as e:
            return ToolResult(result_text=f"Error: write failed: {e}", error=str(e))

        ctx.read_state.set(abs_path, ReadStateEntry(
            # \n-normalized, matching Read's normalization — the freshness fallback
            # compares against a universal-newlines re-read.
            content=new_content,
            timestamp=ctx.read_state.mtime_ticks(abs_path),
            offset=1,
            limit=None,
        ))
        return ToolResult(
            result_text=f"Successfully edited {abs_path} ({count} replacement{'s' if count != 1 else ''}).",
        )

    async def _create_via_edit(self, abs_path: str, content: str, ctx: ToolContext) -> ToolResult:
        decision = await ctx.approval.request(
            ApprovalRequest(
                tool_name=self.name,
                summary=f"Create {abs_path}",
                details={
                    "path": abs_path,
                    "exists": False,
                    "preview": content[:200],
                },
            ),
            scope_key=abs_path,
        )
        if not decision.allow:
            return ToolResult(
                result_text=f"Error: user declined to create {abs_path}.",
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
        return ToolResult(result_text=f"File created at {abs_path}.")


def _check_read_freshness(abs_path: str, ctx: ToolContext) -> str | None:
    prior = ctx.read_state.get(abs_path)
    if prior is None:
        return (
            "you must use the Read tool on this file before editing it. "
            "This protects against accidental overwrites of changes the LLM hasn't seen."
        )
    mtime = ctx.read_state.mtime_ticks(abs_path)
    if mtime != prior.timestamp:
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
            "Read it again before editing."
        )
    return None


# ----------------------------------------------------------------------------
# Curly-quote fallback: LLMs sometimes paste smart-quote text into old_string.
# Try the exact match first; if that misses, try with quotes normalized in both
# directions and re-stylize new_string to match what's actually in the file.
# ----------------------------------------------------------------------------

_CURLY_DOUBLE = ("“", "”")
_CURLY_SINGLE = ("‘", "’")


def _find_actual_string(file_content: str, old: str, new: str) -> tuple[str, str, int]:
    """Returns (actual_old_to_replace, actual_new_to_insert, count). The "actual"
    pair is whichever quote styling matches the file's content."""
    count = file_content.count(old)
    if count > 0:
        return old, new, count

    # Normalize curly → straight in old and look again.
    straight_old = _normalize_quotes(old)
    if straight_old != old:
        count = file_content.count(straight_old)
        if count > 0:
            return straight_old, _normalize_quotes(new), count

    # Try the opposite direction: convert straight quotes in old to curly.
    curly_old = _apply_curly(old)
    if curly_old != old:
        count = file_content.count(curly_old)
        if count > 0:
            return curly_old, _apply_curly(new), count
    return old, new, 0


def _normalize_quotes(s: str) -> str:
    out = s.replace(_CURLY_DOUBLE[0], '"').replace(_CURLY_DOUBLE[1], '"')
    out = out.replace(_CURLY_SINGLE[0], "'").replace(_CURLY_SINGLE[1], "'")
    return out


def _apply_curly(s: str) -> str:
    """Heuristic straight → curly: opening quote when preceded by whitespace /
    start, closing otherwise. Good enough for the fallback's use case."""
    if not s:
        return s
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch == '"':
            opening = i == 0 or s[i - 1].isspace() or s[i - 1] in "([{<"
            out.append(_CURLY_DOUBLE[0] if opening else _CURLY_DOUBLE[1])
        elif ch == "'":
            opening = i == 0 or s[i - 1].isspace() or s[i - 1] in "([{<"
            out.append(_CURLY_SINGLE[0] if opening else _CURLY_SINGLE[1])
        else:
            out.append(ch)
    return "".join(out)
