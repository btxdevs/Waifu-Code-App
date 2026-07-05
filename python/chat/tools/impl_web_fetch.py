"""WebFetch tool — port of Assets/Scripts/ChatTools/WebFetchToolExecutor.cs.

Fetches a URL, converts the page HTML to markdown locally (see web_cache), applies
optional allow/deny domain filters, and returns the markdown (truncated at MARKDOWN_MAX_CHARS).
"""
from __future__ import annotations

from urllib.parse import urlparse

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .web_cache import WebPageReaderCache


_MARKDOWN_MAX_CHARS = 100_000


class WebFetchTool(ToolExecutor):
    name = "WebFetch"
    permission = ToolPermission.READ_ONLY
    activity_label = "Fetching page…"
    defer_until_speech_caught_up = False
    description = (
        "Fetches a URL and returns its content as markdown.\n"
        "- Takes a URL and an optional prompt as input\n"
        "- Fetches the URL content and converts HTML to markdown\n"
        "- Returns the markdown directly — you read it and apply the prompt in your next response\n"
        "- Use this tool when you need to retrieve and analyze web content\n\n"
        "Usage notes:\n"
        "  - The URL must be a fully-formed valid URL\n"
        "  - The prompt is recorded for the user's reference but does not gate the fetch — the full page (truncated to length limit) is returned regardless.\n"
        "  - This tool is read-only and does not modify any files\n"
        "  - Pages are cached for ~10 minutes; repeat calls on the same URL are free. The cache is shared with WebPageOutline / WebPageRead.\n"
        "  - For very large pages, consider WebPageOutline + WebPageRead — they return the markdown by section, which is cheaper when you already know which section you need.\n"
        "  - Use allowed_domains / blocked_domains to restrict which hosts may be fetched."
    )

    def __init__(self, cache: WebPageReaderCache) -> None:
        self._cache = cache

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri", "description": "Absolute URL to fetch (http or https)."},
                "prompt": {
                    "type": "string",
                    "description": (
                        "What you're looking for on the page. It is echoed back above the content for "
                        "context — the page's full markdown is returned and YOU extract the answer from "
                        "it in your next response. (No sub-model runs over the page.)"
                    ),
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of host suffixes to allow. When set, the URL's host must match "
                        "(exact or subdomain) at least one entry; otherwise the request is refused. "
                        "Example: ['github.com', 'docs.python.org']."
                    ),
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of host suffixes to refuse. Takes precedence over allowed_domains.",
                },
            },
            "required": ["url", "prompt"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        url = arguments.get("url")
        prompt = arguments.get("prompt")
        if not isinstance(url, str) or not url:
            return ToolResult(result_text="Error: 'url' is required.", error="bad args")
        if not isinstance(prompt, str) or not prompt:
            return ToolResult(result_text="Error: 'prompt' is required.", error="bad args")

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or parsed.scheme not in ("http", "https"):
            return ToolResult(result_text=f"Error: invalid URL: {url}", error="bad url")

        blocked = arguments.get("blocked_domains") or []
        allowed = arguments.get("allowed_domains") or []
        if isinstance(blocked, list) and any(_host_matches(host, b) for b in blocked if isinstance(b, str)):
            return ToolResult(
                result_text=f"Error: host '{host}' is on the blocked_domains list.",
                error="blocked host",
            )
        if isinstance(allowed, list) and allowed and not any(
            _host_matches(host, a) for a in allowed if isinstance(a, str)
        ):
            return ToolResult(
                result_text=f"Error: host '{host}' is not on the allowed_domains list.",
                error="not allowed",
            )

        try:
            page = await self._cache.get_page(url)
        except Exception as e:
            return ToolResult(result_text=f"Error: failed to fetch {url}: {e}", error=str(e))

        body = page.markdown or ""
        if len(body) > _MARKDOWN_MAX_CHARS:
            # Spill the full markdown so nothing is lost — the model can Read the
            # file, or navigate by section with WebPageOutline + WebPageRead.
            from .spill import spill_text
            path = spill_text(body, self.name)
            note = f"\n…[truncated at {_MARKDOWN_MAX_CHARS} chars"
            if path:
                note += f"; full page markdown saved to: {path} (Read it if needed)"
            note += "; or use WebPageOutline + WebPageRead to page through by section]"
            body = body[:_MARKDOWN_MAX_CHARS] + note
        return ToolResult(
            result_text=(
                f"URL: {url}\nPrompt: {prompt}\n---\n{body}"
            ),
        )


def _host_matches(host: str, suffix: str) -> bool:
    if not host or not suffix:
        return False
    h = host.lower()
    s = suffix.lower().lstrip(".")
    return h == s or h.endswith("." + s)
