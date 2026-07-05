"""WebPageOutline tool — port of Assets/Scripts/ChatTools/WebPageOutlineToolExecutor.cs.

Returns the cached page's section outline (markdown headings) and a body
preview if no headings were detected.
"""
from __future__ import annotations

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .web_cache import WebPageReaderCache


class WebPageOutlineTool(ToolExecutor):
    name = "WebPageOutline"
    permission = ToolPermission.READ_ONLY
    activity_label = "Reading webpage…"
    defer_until_speech_caught_up = False
    description = (
        "Fetch a web page and return a markdown outline: total line count plus a list of sections "
        "with their heading text, depth, and line range. Pair with WebPageRead: call this first to "
        "learn which ranges exist, then call WebPageRead for the specific section(s) you need.\n\n"
        "Use this instead of WebFetch when the page is large and you only need parts of it — outline "
        "is cheap (just headings), so you can skim before deciding what to read in full. The page is "
        "cached (~10 minutes) and the cache is shared with WebFetch and WebPageRead, so calling any "
        "of them again on the same URL is free."
    )

    def __init__(self, cache: WebPageReaderCache) -> None:
        self._cache = cache

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute URL of the web page to read."},
            },
            "required": ["url"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(result_text="Error: 'url' is required.", error="bad args")
        try:
            page = await self._cache.get_page(url)
        except Exception as e:
            return ToolResult(result_text=f"Error: failed to fetch {url}: {e}", error=str(e))

        total = len(page.lines)
        out: list[str] = [f"URL: {url}", f"Total lines: {total}"]
        if not page.sections:
            out.append("No markdown headings detected. Use WebPageRead with offset/limit to read the body.")
            preview = page.lines[:8]
            if preview:
                out.append("---")
                out.extend(preview)
        else:
            out.append("Sections (heading | level | offset | end_line):")
            for s in page.sections:
                out.append(f"- {s.title} | L{s.level} | {s.start_line}-{s.end_line}")
        return ToolResult(result_text="\n".join(out))
