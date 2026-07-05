"""Grep tool — port of Assets/Scripts/ChatTools/GrepSearchToolExecutor.cs.

Runs `rg` (ripgrep). A platform-matched binary is vendored with the app under
python/vendor/ripgrep/, so the tool behaves identically on every machine and
never depends on a system install (no system-PATH `rg` required). The schema is
rg-shaped because that's what the LLM was trained on for this tool.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .workspace import default_root, resolve_path


_DEFAULT_HEAD_LIMIT = 250
_MAX_LINE_WIDTH = 500


class GrepTool(ToolExecutor):
    name = "Grep"
    permission = ToolPermission.READ_ONLY
    activity_label = "Searching content…"
    defer_until_speech_caught_up = False
    description = (
        "A powerful search tool built on ripgrep\n\n"
        "  Usage:\n"
        "  - ALWAYS use Grep for search tasks. NEVER invoke `grep`, `rg` or `Select-String` as a shell command. The Grep tool has been optimized for correct permissions and access.\n"
        "  - Supports full regex syntax (e.g., \"log.*Error\", \"function\\s+\\w+\")\n"
        "  - Filter files with glob parameter (e.g., \"*.js\", \"**/*.tsx\") or type parameter (e.g., \"js\", \"py\", \"rust\")\n"
        "  - Output modes: \"content\" shows matching lines, \"files_with_matches\" shows only file paths (default), \"count\" shows match counts\n"
        "  - Use Agent tool for open-ended searches requiring multiple rounds\n"
        "  - Pattern syntax: Uses ripgrep (not grep) - literal braces need escaping (use `interface\\{\\}` to find `interface{}` in Go code)\n"
        "  - Multiline matching: By default patterns match within single lines only. For cross-line patterns like `struct \\{[\\s\\S]*?field`, use `multiline: true`"
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (rg PATH). Defaults to current working directory.",
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. \"*.js\", \"*.{ts,tsx}\") - maps to rg --glob",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "Output mode: \"content\" shows matching lines (supports -A/-B/-C context, "
                        "-n line numbers, head_limit), \"files_with_matches\" shows file paths "
                        "(supports head_limit), \"count\" shows match counts (supports head_limit). "
                        "Defaults to \"files_with_matches\"."
                    ),
                },
                "-B": {"type": "number", "description": "Number of lines to show before each match (rg -B). Requires output_mode: \"content\", ignored otherwise."},
                "-A": {"type": "number", "description": "Number of lines to show after each match (rg -A). Requires output_mode: \"content\", ignored otherwise."},
                "-C": {"type": "number", "description": "Alias for context."},
                "context": {"type": "number", "description": "Number of lines to show before and after each match (rg -C). Requires output_mode: \"content\", ignored otherwise."},
                "-n": {"type": "boolean", "description": "Show line numbers in output (rg -n). Requires output_mode: \"content\", ignored otherwise. Defaults to true."},
                "-i": {"type": "boolean", "description": "Case insensitive search (rg -i)"},
                "type": {"type": "string", "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."},
                "head_limit": {
                    "type": "number",
                    "description": (
                        "Limit output to first N lines/entries, equivalent to \"| head -N\". Works across all output modes: "
                        "content (limits output lines), files_with_matches (limits file paths), count (limits count entries). "
                        "Defaults to 250 when unspecified. Pass 0 for unlimited (use sparingly — large result sets waste context)."
                    ),
                },
                "offset": {"type": "number", "description": "Skip first N lines/entries before applying head_limit, equivalent to \"| tail -n +N | head -N\". Works across all output modes. Defaults to 0."},
                "multiline": {"type": "boolean", "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false."},
            },
            "required": ["pattern"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolResult(result_text="Error: 'pattern' is required.", error="bad args")

        raw_path = arguments.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            resolution = resolve_path(ctx.workspace, raw_path)
            if not resolution.ok:
                return ToolResult(result_text="Error: " + resolution.error, error=resolution.error)
            search_root = resolution.absolute_path
        else:
            search_root = default_root(ctx.workspace)
            if not search_root:
                return ToolResult(
                    result_text="Error: workspace has no allowed roots configured.",
                    error="no root",
                )

        output_mode = arguments.get("output_mode") or "files_with_matches"
        if output_mode not in ("content", "files_with_matches", "count"):
            output_mode = "files_with_matches"

        head_limit = _read_int(arguments, "head_limit", _DEFAULT_HEAD_LIMIT)
        offset = max(0, _read_int(arguments, "offset", 0))
        case_insensitive = bool(arguments.get("-i", False))
        multiline = bool(arguments.get("multiline", False))
        show_line_nums = bool(arguments.get("-n", True))
        before = _read_int(arguments, "-B", 0)
        after = _read_int(arguments, "-A", 0)
        context = _read_int(arguments, "-C", _read_int(arguments, "context", 0))
        if context > 0:
            before = max(before, context)
            after = max(after, context)
        glob_filter = arguments.get("glob") if isinstance(arguments.get("glob"), str) else None
        type_filter = arguments.get("type") if isinstance(arguments.get("type"), str) else None

        rg = _find_ripgrep()
        if not rg:
            return ToolResult(
                result_text=(
                    "Error: ripgrep (rg) is not available. A copy is expected to ship at "
                    "python/vendor/ripgrep/rg.exe — reinstall or restore the app files."
                ),
                error="ripgrep not found",
            )
        return await self._run_with_ripgrep(
            rg, search_root, pattern, output_mode, head_limit, offset,
            case_insensitive, multiline, show_line_nums, before, after,
            glob_filter, type_filter, ctx,
        )

    async def _run_with_ripgrep(
        self, rg, search_root, pattern, output_mode, head_limit, offset,
        case_insensitive, multiline, show_line_nums, before, after,
        glob_filter, type_filter, ctx,
    ) -> ToolResult:
        # --max-columns-preview: without it rg OMITS long matching lines entirely
        # ("[Omitted long matching line]"); with it the truncated prefix still shows.
        args = [rg, "--hidden", "--color=never",
                "--max-columns", str(_MAX_LINE_WIDTH), "--max-columns-preview"]
        # VCS exclusions.
        for vcs in (".git", ".svn", ".hg", ".bzr", ".jj", ".sl"):
            args += ["--glob", f"!{vcs}"]
        for ig in (ctx.workspace.ignored_directory_names or []):
            args += ["--glob", f"!**/{ig}/**"]
        if case_insensitive:
            args.append("-i")
        if multiline:
            args += ["-U", "--multiline-dotall"]
        if output_mode == "files_with_matches":
            args.append("-l")
        elif output_mode == "count":
            args.append("-c")
        else:
            args += ["-n" if show_line_nums else "-N", "--no-heading", "--with-filename"]
            if before > 0:
                args += ["-B", str(before)]
            if after > 0:
                args += ["-A", str(after)]
        if type_filter:
            args += ["--type", type_filter]
        if glob_filter:
            for tok in _split_globs(glob_filter):
                args += ["--glob", tok]
        # Pattern — use `-e` so leading dashes don't get parsed as flags.
        args += ["-e", pattern]
        args.append(search_root)

        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        except asyncio.TimeoutError:
            return ToolResult(result_text="Error: grep timed out after 30s.", error="timeout")
        except OSError as e:
            return ToolResult(result_text=f"Error: failed to run ripgrep: {e}", error=str(e))

        stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
        stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""
        rc = proc.returncode or 0
        # ripgrep exits 1 when there are no matches — that's not an error.
        if rc not in (0, 1):
            err = (stderr.strip().splitlines()[:5] or ["(no stderr)"])[0]
            return ToolResult(result_text=f"Error: ripgrep failed (exit {rc}): {err}", error=err)
        return _format_rg_output(stdout, output_mode, head_limit, offset, search_root, ctx)


# Vendored ripgrep. From source it lives at CompanionApp/python/vendor/ripgrep/; the
# packaged build copies vendor/ next to the .exe (APP_ROOT/vendor/ripgrep/).
from ..app_paths import APP_ROOT

_VENDOR_RIPGREP_DIR = (
    str(APP_ROOT / "vendor" / "ripgrep")
    if getattr(sys, "frozen", False)
    else str(APP_ROOT / "python" / "vendor" / "ripgrep")
)


def _find_ripgrep() -> str | None:
    """Locate ripgrep. Prefer the binary vendored with the app so search behavior
    is identical on every machine and never depends on a system install; only fall
    back to a PATH `rg` if the vendored copy is missing (e.g. an unbundled platform)."""
    exe = "rg.exe" if os.name == "nt" else "rg"
    vendored = os.path.join(_VENDOR_RIPGREP_DIR, exe)
    if os.path.isfile(vendored):
        return vendored
    return shutil.which("rg")


def _read_int(args: dict, key: str, fallback: int) -> int:
    v = args.get(key)
    if v is None:
        return fallback
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _split_globs(g: str) -> list[str]:
    """Split a glob filter string. Keeps `{a,b}` groups intact, splits on spaces
    and commas otherwise."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in g:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif depth == 0 and ch in " ,":
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [t for t in out if t]


def _format_rg_output(stdout: str, output_mode: str, head_limit: int, offset: int,
                     search_root: str, ctx: ToolContext) -> ToolResult:
    lines = [l for l in stdout.split("\n") if l]
    # Relativize to search_root to save tokens. rg echoes each path with the same
    # separators the root was passed with, so strip by prefix LENGTH (normcase
    # compare — Windows paths are case-insensitive) and leave the rest of the
    # line untouched (content-mode lines carry arbitrary text after the path).
    if search_root:
        root_cmp = os.path.normcase(search_root)
        rl = len(search_root)
        lines = [
            l[rl:].lstrip("\\/") if len(l) > rl and os.path.normcase(l[:rl]) == root_cmp else l
            for l in lines
        ]
    return _format_list(lines, head_limit, offset)


def _format_list(lines: list[str], head_limit: int, offset: int) -> ToolResult:
    if not lines:
        return ToolResult(result_text="No matches found")
    total = len(lines)
    sliced = lines[offset:]
    if head_limit and head_limit > 0:
        sliced = sliced[:head_limit]
    out = list(sliced)
    if total > offset + len(sliced):
        out.append(f"…[showing {offset + 1}-{offset + len(sliced)} of {total}; pass offset/head_limit to page]")
    return ToolResult(result_text="\n".join(out))
