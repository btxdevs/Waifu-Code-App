"""WebPageRead tool — port of Assets/Scripts/ChatTools/WebPageReadToolExecutor.cs.

Pages through the cached markdown body using the same offset/limit shape as Read.
"""
from __future__ import annotations

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .web_cache import WebPageReaderCache


_DEFAULT_LINE_LIMIT = 2000
_MAX_LINES_PER_CALL = 2000
_MAX_LINE_LENGTH = 2000


class WebPageReadTool(ToolExecutor):
    name = "WebPageRead"
    permission = ToolPermission.READ_ONLY
    activity_label = "Reading webpage…"
    defer_until_speech_caught_up = False
    description = (
        f"Read a web page's markdown content. Same offset/limit shape as the Read tool — by default "
        f"returns the first {_DEFAULT_LINE_LIMIT} lines of the page; pass offset/limit to page through larger documents.\n\n"
        "Usage:\n"
        "- The url parameter must be an absolute URL\n"
        f"- By default, it reads up to {_DEFAULT_LINE_LIMIT} lines starting from line 1\n"
        "- You can optionally specify a line offset (1-indexed) and limit for paging through long pages\n"
        "- Results are returned using cat -n format, with line numbers starting at 1 — matching the Read tool's output, so you can quote specific lines back\n"
        "- For very large pages, call WebPageOutline first to learn the section structure, then call WebPageRead with the offset/limit you need\n"
        "- The page cache is shared with WebFetch and WebPageOutline so repeated reads against the same URL don't refetch (~10-minute TTL)"
    )

    def __init__(self, cache: WebPageReaderCache) -> None:
        self._cache = cache

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute URL of the web page to read. Should match the URL passed to WebPageOutline."},
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from (1-indexed). Only provide if the page is too large to read at once.",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": f"The number of lines to read. Only provide if the page is too large to read at once. Capped at {_MAX_LINES_PER_CALL}.",
                    "minimum": 1,
                },
            },
            "required": ["url"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(result_text="Error: 'url' is required.", error="bad args")
        offset = arguments.get("offset")
        try:
            offset = int(offset) if offset is not None else 1
        except (TypeError, ValueError):
            offset = 1
        if offset < 1:
            offset = 1
        limit = arguments.get("limit")
        try:
            limit = int(limit) if limit is not None else _DEFAULT_LINE_LIMIT
        except (TypeError, ValueError):
            limit = _DEFAULT_LINE_LIMIT
        if limit < 1:
            limit = 1
        if limit > _MAX_LINES_PER_CALL:
            limit = _MAX_LINES_PER_CALL

        try:
            page = await self._cache.get_page(url)
        except Exception as e:
            return ToolResult(result_text=f"Error: failed to fetch {url}: {e}", error=str(e))

        total = len(page.lines)
        if total == 0:
            return ToolResult(result_text=f"URL: {url}\n(page has no content)")
        start = offset - 1
        end = min(total, start + limit)
        out: list[str] = [f"URL: {url}", f"Lines {start + 1}–{end} of {total}", "---"]
        for i in range(start, end):
            l = page.lines[i]
            if len(l) > _MAX_LINE_LENGTH:
                l = l[:_MAX_LINE_LENGTH] + "…"
            out.append(f"{i + 1}\t{l}")
        if end < total:
            out.append(f"…[truncated; call WebPageRead with offset={end + 1} to continue]")
        return ToolResult(result_text="\n".join(out))
