"""WebSearch tool — queries the web via the `ddgs` metasearch library (Dux Distributed
Global Search, the renamed duckduckgo_search).

ddgs aggregates results from multiple public engines (DuckDuckGo, Brave, Google, Startpage,
Mojeek, …) in-process — there's no instance to host and no API keys. Tunables (results per
call, page cap, request timeout, retries) come from app.config.json's `webSearch` block.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from urllib.parse import urlparse

from ddgs import DDGS

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


_DEFAULT_PAGE_CAP = 10
_DEFAULT_RESULTS_PER_CALL = 8
_REQUEST_TIMEOUT_SECONDS = 15
_MAX_ATTEMPTS = 2
# Comma-delimited ddgs text backends. All engines except grokipedia, mojeek, and wikipedia.
# (Images only support bing/duckduckgo, so that path stays on ddgs's "auto".)
_DEFAULT_TEXT_BACKEND = "bing,brave,duckduckgo,google,startpage,yandex,yahoo"


@dataclass
class WebSearchConfig:
    """Pulled from app.config.json's `webSearch` block."""
    results_per_call: int = _DEFAULT_RESULTS_PER_CALL
    hard_cap_page: int = _DEFAULT_PAGE_CAP
    request_timeout_seconds: int = _REQUEST_TIMEOUT_SECONDS
    max_attempts: int = _MAX_ATTEMPTS
    # Which ddgs engines to query for text search ("auto", a single name, or a comma-delimited list).
    text_backend: str = _DEFAULT_TEXT_BACKEND


def load_web_search_config() -> WebSearchConfig:
    from ..app_paths import APP_ROOT
    p = APP_ROOT / "app.config.json"
    cfg = WebSearchConfig()
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        block = raw.get("webSearch") if isinstance(raw, dict) else None
        if not isinstance(block, dict):
            return cfg
        if isinstance(block.get("resultsPerCall"), int):
            cfg.results_per_call = max(1, block["resultsPerCall"])
        if isinstance(block.get("hardCapPage"), int):
            cfg.hard_cap_page = max(1, block["hardCapPage"])
        if isinstance(block.get("requestTimeoutSeconds"), int):
            cfg.request_timeout_seconds = max(1, block["requestTimeoutSeconds"])
        if isinstance(block.get("maxAttempts"), int):
            cfg.max_attempts = max(1, block["maxAttempts"])
        if isinstance(block.get("textBackend"), str) and block["textBackend"].strip():
            cfg.text_backend = block["textBackend"].strip()
    except (OSError, json.JSONDecodeError):
        pass
    return cfg


class WebSearchTool(ToolExecutor):
    name = "WebSearch"
    permission = ToolPermission.READ_ONLY
    activity_label = "Searching the web…"
    defer_until_speech_caught_up = False
    description = (
        "Search the web and return results. Use search_type='general' (default) for web pages with "
        "titles/URLs/snippets when the user asks about facts, current events, or anything that needs "
        "up-to-date external information. Use search_type='images' when the user asks for pictures, "
        "photos, or visual references — returns image URLs and thumbnails.\n\n"
        "Use page (1-indexed) to fetch additional results when the first page is insufficient. "
        "Use allowed_domains / blocked_domains to focus on or exclude specific hosts.\n\n"
        "IMPORTANT: When you use information from these results in your reply, cite the source as a "
        "numbered footnote or by quoting the URL. Never present web-derived facts as if they were "
        "your own knowledge."
    )

    def __init__(self, config: WebSearchConfig | None = None) -> None:
        self.config = config or load_web_search_config()

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The web search query.", "minLength": 2},
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of host suffixes. When set, results whose URL host doesn't match "
                        "(exact or subdomain) any entry are dropped. Example: ['github.com', 'docs.python.org']."
                    ),
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of host suffixes to drop from results. Takes precedence over "
                        "allowed_domains. Example: ['pinterest.com'] also drops www.pinterest.com."
                    ),
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.config.hard_cap_page,
                    "description": (
                        f"1-indexed result page (default 1, max {self.config.hard_cap_page}). Bump this to keep exploring when the first page didn't surface what you need."
                    ),
                },
                "search_type": {
                    "type": "string",
                    "enum": ["general", "images"],
                    "description": (
                        "Type of search. 'general' (default) returns web pages with snippets. "
                        "'images' returns image results with direct image URLs, thumbnails, and resolutions."
                    ),
                },
            },
            "required": ["query"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or len(query.strip()) < 2:
            return ToolResult(result_text="Error: 'query' must be at least 2 characters.", error="bad args")

        page = arguments.get("page")
        try:
            page = int(page) if page is not None else 1
        except (TypeError, ValueError):
            page = 1
        page = max(1, min(self.config.hard_cap_page, page))

        search_type = arguments.get("search_type") or "general"
        if search_type not in ("general", "images"):
            search_type = "general"

        allowed = arguments.get("allowed_domains") or []
        blocked = arguments.get("blocked_domains") or []
        if not isinstance(allowed, list):
            allowed = []
        if not isinstance(blocked, list):
            blocked = []

        last_err: str | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                results = await asyncio.to_thread(self._fetch, query.strip(), page, search_type)
            except Exception as e:
                last_err = str(e)
                continue
            if results:
                filtered = _apply_filters(results, allowed, blocked, search_type)
                if filtered:
                    return ToolResult(result_text=_format_results(query, page, filtered, search_type))
            # Empty result on first attempt is worth retrying (transient backend hiccup).
        if last_err:
            return ToolResult(result_text=f"Error: web search failed: {last_err}", error=last_err)
        return ToolResult(result_text=f"No results for: {query}")

    def _fetch(self, query: str, page: int, search_type: str) -> list[dict]:
        """Run the ddgs query on a worker thread and normalize results into the keys the
        formatter/filter expect (url / content for text; img_src / thumbnail_src / resolution
        for images), so the rest of the pipeline stays backend-agnostic."""
        ddgs = DDGS(timeout=self.config.request_timeout_seconds)
        n = self.config.results_per_call
        if search_type == "images":
            raw = ddgs.images(query, safesearch="moderate", max_results=n, page=page) or []
            out: list[dict] = []
            for r in raw:
                if not isinstance(r, dict):
                    continue
                w, h = r.get("width"), r.get("height")
                out.append({
                    "title": r.get("title"),
                    "img_src": r.get("image"),
                    "thumbnail_src": r.get("thumbnail"),
                    "url": r.get("url"),
                    "source": r.get("source"),
                    "resolution": f"{w}x{h}" if w and h else "",
                })
            return out
        raw = ddgs.text(query, safesearch="moderate", max_results=n, page=page,
                        backend=self.config.text_backend) or []
        out = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            out.append({
                "title": r.get("title"),
                "url": r.get("href"),
                "content": r.get("body"),
            })
        return out


def _apply_filters(results: list[dict], allowed: list, blocked: list, search_type: str) -> list[dict]:
    out: list[dict] = []
    seen_urls: set[str] = set()
    for r in results:
        if not isinstance(r, dict):
            continue
        primary_url = str(r.get("url") or r.get("img_src") or "")
        if not primary_url:
            continue
        if primary_url in seen_urls:
            continue
        seen_urls.add(primary_url)
        host = (urlparse(primary_url).hostname or "").lower()
        if any(_host_matches(host, b) for b in blocked if isinstance(b, str)):
            continue
        if allowed and not any(_host_matches(host, a) for a in allowed if isinstance(a, str)):
            continue
        out.append(r)
    return out


def _format_results(query: str, page: int, results: list[dict], search_type: str) -> str:
    if search_type == "images":
        lines = [f"Image results for: `{query}`"]
        for i, r in enumerate(results, start=1):
            lines.append(f"{i}. {r.get('title') or '(no title)'}")
            if r.get("img_src"):
                lines.append(f"   Image: {r['img_src']}")
            if r.get("thumbnail_src") or r.get("thumbnail"):
                lines.append(f"   Thumbnail: {r.get('thumbnail_src') or r.get('thumbnail')}")
            if r.get("resolution"):
                lines.append(f"   Resolution: {r['resolution']}")
            if r.get("url"):
                lines.append(f"   Page: {r['url']}")
            if r.get("source"):
                lines.append(f"   Source: {r['source']}")
        lines.append("")
        lines.append("(Cite the source page URL when referencing these results.)")
        return "\n".join(lines)
    lines = [f"Search results for: `{query}` (page {page})"]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r.get('title') or '(no title)'}")
        if r.get("url"):
            lines.append(f"   URL: {r['url']}")
        content = r.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(f"   Snippet: {content.strip()}")
    lines.append("")
    lines.append("(Cite results as numbered footnotes or by quoting the URL.)")
    return "\n".join(lines)


def _host_matches(host: str, suffix: str) -> bool:
    if not host or not suffix:
        return False
    h = host.lower()
    s = suffix.lower().lstrip(".")
    return h == s or h.endswith("." + s)
