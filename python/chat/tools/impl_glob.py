"""Glob tool — port of Assets/Scripts/ChatTools/GlobSearchToolExecutor.cs.

Recursive file walk with `**` / `*` glob matching. Sorts results by mtime newest-
first and relativizes paths against the workspace root to save tokens.
"""
from __future__ import annotations

import os
import re

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .workspace import default_root, resolve_path


_DEFAULT_LIMIT = 100


class GlobTool(ToolExecutor):
    name = "Glob"
    permission = ToolPermission.READ_ONLY
    activity_label = "Finding files…"
    defer_until_speech_caught_up = False
    description = (
        "- Fast file pattern matching tool that works with any codebase size\n"
        "- Supports glob patterns like \"**/*.js\" or \"src/**/*.ts\"\n"
        "- Returns matching file paths sorted by modification time\n"
        "- Use this tool when you need to find files by name patterns\n"
        "- When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead"
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "The directory to search in. If not specified, the current working "
                        "directory will be used. IMPORTANT: Omit this field to use the "
                        "default directory. DO NOT enter \"undefined\" or \"null\" - simply "
                        "omit it for the default behavior. Must be a valid directory path if provided."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(result_text="Error: 'pattern' is required.", error="bad args")

        raw_path = arguments.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            resolution = resolve_path(ctx.workspace, raw_path)
            if not resolution.ok:
                return ToolResult(result_text="Error: " + resolution.error, error=resolution.error)
            search_root = resolution.absolute_path
            if not os.path.isdir(search_root):
                return ToolResult(
                    result_text=f"Error: path is not a directory: {search_root}",
                    error="not a directory",
                )
        else:
            search_root = default_root(ctx.workspace)
            if not search_root:
                return ToolResult(
                    result_text="Error: workspace has no allowed roots configured.",
                    error="no root",
                )

        try:
            regex = re.compile(_glob_to_regex(pattern))
        except re.error as e:
            return ToolResult(result_text=f"Error: invalid glob pattern: {e}", error=str(e))

        ignored = {n.casefold() for n in (ctx.workspace.ignored_directory_names or []) if n}
        max_results = max(_DEFAULT_LIMIT, ctx.workspace.max_glob_results if ctx.workspace else _DEFAULT_LIMIT)
        follow_symlinks = ctx.workspace.follow_symlinks if ctx.workspace else False

        matches: list[tuple[float, str]] = []
        cap_hit = False
        for abs_path in _walk_files(search_root, ignored, follow_symlinks):
            rel = os.path.relpath(abs_path, search_root).replace(os.sep, "/")
            # Match against the rel path so `**/*.cs` works the way callers expect.
            if not (regex.fullmatch(rel) or regex.fullmatch(os.path.basename(rel))):
                continue
            try:
                mtime = os.path.getmtime(abs_path)
            except OSError:
                mtime = 0.0
            matches.append((mtime, abs_path))
            if len(matches) >= max_results:
                cap_hit = True
                break

        if not matches:
            return ToolResult(result_text="No files found")

        matches.sort(key=lambda kv: kv[0], reverse=True)
        ws_root = default_root(ctx.workspace) or ""
        lines: list[str] = []
        for _, p in matches:
            if ws_root and p.startswith(ws_root):
                rel = p[len(ws_root):].lstrip(os.sep).lstrip(os.altsep or os.sep)
                lines.append(rel.replace(os.sep, "/"))
            else:
                lines.append(p.replace(os.sep, "/"))
        if cap_hit:
            lines.append(f"…[truncated at {max_results} results — use a more specific path or pattern]")
        return ToolResult(result_text="\n".join(lines))


# ----------------------------------------------------------------------------
# Glob → regex compiler. Mirrors Assets/Scripts/ChatTools/Workspace/GlobMatcher.cs.
# Supports `**` (any path segment incl. /), `*` (any chars in one segment), `?`
# (one char), `[abc]` and `[!abc]` character classes.
# ----------------------------------------------------------------------------

def _glob_to_regex(pattern: str) -> str:
    """Translate the glob into a regex that matches against rel paths using "/"
    as the separator."""
    p = pattern.replace("\\", "/")
    out: list[str] = ["^"]
    i = 0
    n = len(p)
    while i < n:
        c = p[i]
        if c == "*":
            if i + 1 < n and p[i + 1] == "*":
                # `**` — any chars including /. May be followed by / to mean "any
                # number of full segments".
                j = i + 2
                if j < n and p[j] == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = i + 1
            if j < n and p[j] == "!":
                out.append("[^")
                j += 1
            else:
                out.append("[")
            while j < n and p[j] != "]":
                if p[j] == "\\" and j + 1 < n:
                    out.append(re.escape(p[j + 1]))
                    j += 2
                else:
                    out.append(p[j])
                    j += 1
            out.append("]")
            i = j + 1
        elif c == "{":
            # `{a,b,c}` — alternation.
            j = p.find("}", i + 1)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                inner = p[i + 1:j].split(",")
                out.append("(?:" + "|".join(re.escape(s) for s in inner) + ")")
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append("$")
    return "".join(out)


def _walk_files(root: str, ignored_dirs: set[str], follow_symlinks: bool):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Prune ignored directory names in-place so os.walk stops descending.
        dirnames[:] = [d for d in dirnames if d.casefold() not in ignored_dirs]
        for fname in filenames:
            yield os.path.join(dirpath, fname)
