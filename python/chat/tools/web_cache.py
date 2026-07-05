"""WebPageReaderCache — shared infrastructure for WebFetch / WebPageOutline / WebPageRead.

Fetch strategy (see `_fetch`):
  1. If the URL looks like a MediaWiki article (`/wiki/Title` or `index.php?title=`), try the
     MediaWiki action API first (`api.php?action=parse`). The API returns clean parser HTML and
     is usually NOT behind the same anti-bot/Cloudflare gate as the rendered pages, so it both
     dodges bot detection and skips the site chrome.
  2. Otherwise (or if the API attempt fails) fall back to a direct browser-UA HTML fetch and
     convert locally with `markdownify` after a BeautifulSoup pre-clean that drops chrome
     (scripts, nav, header/footer, sidebars).
Either way we walk ATX headings to build a section outline and cache the result ~10 minutes.

This replaced an earlier Jina Reader (https://r.jina.ai/) round-trip — Jina added latency and
silently failed on anti-bot-gated pages (e.g. Miraheze wikis returned a 403 interstitial),
whereas a direct browser-UA fetch (and, for wikis, the API) gets the real page. markdownify
(the Python analog of the JS WebFetch's turndown) was chosen for the most consistent
ATX-heading fidelity, which the section outline below depends on.

Cache lives in-process; cleared on app restart.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as _html_to_md


_CACHE_TTL_SECONDS = 600
_REQUEST_TIMEOUT_SECONDS = 30

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

# Chrome/boilerplate elements stripped before conversion so the markdown is mostly content.
_STRIP_TAGS = (
    "script", "style", "noscript", "template", "svg", "iframe",
    "nav", "header", "footer", "aside", "form",
)
# CSS selectors for common site chrome (notably MediaWiki, which game wikis use heavily).
_STRIP_SELECTORS = (
    "#mw-navigation", "#mw-panel", "#mw-head", "#footer", ".mw-footer",
    ".navbox", ".printfooter", ".mw-editsection", ".mw-jump-link",
    "#toc", ".toc", "#catlinks", ".catlinks", ".vector-toc", ".sidebar",
)
# Containers that, when present, hold the actual page content — preferred over <body> so we
# trim the surrounding chrome even when it isn't tagged with the elements/selectors above.
# `.mw-parser-output` is what the MediaWiki API's action=parse returns.
_MAIN_SELECTORS = (".mw-parser-output", "#mw-content-text", "main", "article", "#content", '[role="main"]')

_HEADING_RX = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RX = re.compile(r"^\s*(```|~~~)")


@dataclass
class Section:
    level: int
    title: str
    start_line: int  # 1-indexed
    end_line: int    # 1-indexed inclusive


@dataclass
class CachedPage:
    url: str
    markdown: str
    lines: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    fetched_at: float = 0.0


class WebPageReaderCache:
    """Shared in-memory cache."""

    def __init__(self, ttl_seconds: int = _CACHE_TTL_SECONDS,
                 timeout_seconds: int = _REQUEST_TIMEOUT_SECONDS):
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._entries: dict[str, CachedPage] = {}

    def invalidate(self, url: str | None = None) -> None:
        if url is None:
            self._entries.clear()
            return
        self._entries.pop(self._key(url), None)

    async def get_page(self, url: str) -> CachedPage:
        """Returns a cached page or fetches and caches it. Raises on transport error."""
        import asyncio
        key = self._key(url)
        cached = self._entries.get(key)
        now = time.monotonic()
        if cached is not None and (now - cached.fetched_at) < self._ttl:
            return cached
        # urllib is sync — push to a thread so the asyncio loop isn't blocked.
        page = await asyncio.to_thread(self._fetch, url)
        self._entries[key] = page
        return page

    def _fetch(self, url: str) -> CachedPage:
        # MediaWiki-looking URLs: try the action API first (clean content + usually not
        # anti-bot gated). Any failure silently falls through to the direct HTML fetch.
        markdown = _try_mediawiki_api(url, self._timeout)
        if markdown is None:
            markdown = self._fetch_html_markdown(url)
        lines = markdown.split("\n")
        sections = _parse_sections(lines)
        return CachedPage(
            url=url,
            markdown=markdown,
            lines=lines,
            sections=sections,
            fetched_at=time.monotonic(),
        )

    def _fetch_html_markdown(self, url: str) -> str:
        """Direct browser-UA fetch of the rendered page, converted locally."""
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            with httpx.Client(follow_redirects=True, timeout=self._timeout, headers=headers) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                text = resp.text
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code} {e.response.reason_phrase}") from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"transport error: {e}") from e

        # HTML → markdown locally; non-HTML (plain text, raw markdown, JSON…) is used as-is.
        if "html" in content_type.lower() or (not content_type and "<html" in text[:2048].lower()):
            return _html_to_markdown(text)
        return text

    @staticmethod
    def _key(url: str) -> str:
        return (url or "").strip().casefold()


def _html_to_markdown(html: str, title: str | None = None) -> str:
    """Strip site chrome, isolate the main content container, and convert to ATX markdown.
    Pass `title` to force the leading H1 (used for API content, which has no #firstHeading)."""
    soup = BeautifulSoup(html, "html.parser")

    if title is None:
        # Page title before we trim — MediaWiki keeps it in #firstHeading, outside the content body.
        title_el = soup.select_one("h1#firstHeading") or soup.find("h1") or soup.title
        title = title_el.get_text(strip=True) if title_el else ""

    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    for selector in _STRIP_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    container = None
    for selector in _MAIN_SELECTORS:
        container = soup.select_one(selector)
        if container is not None:
            break
    if container is None:
        container = soup.body or soup

    markdown = _html_to_md(str(container), heading_style="ATX")
    # Prepend the page title as a top-level heading when the body doesn't already lead with one
    # (MediaWiki's content body starts below the page title).
    if title and not markdown.lstrip().startswith("#"):
        markdown = f"# {title}\n\n{markdown}"
    # Collapse the runs of blank lines markdownify leaves between stripped blocks.
    markdown = re.sub(r"\n[ \t]*\n[ \t]*(\n[ \t]*)+", "\n\n", markdown)
    return markdown.strip() + "\n"


# Title prefixes that MediaWiki's parser API can't render as articles — skip the API for these.
_NON_ARTICLE_PREFIXES = ("Special:", "Media:")


def _mediawiki_targets(url: str) -> tuple[str, list[str]] | None:
    """If `url` looks like a MediaWiki article, return (page_title, [api.php candidates]).
    Returns None when the URL doesn't match a MediaWiki article pattern."""
    parts = urlsplit(url)
    if not parts.scheme.startswith("http") or not parts.netloc:
        return None

    path = parts.path or ""
    title = ""
    base_prefix = ""  # path segment before the article entry point, e.g. "" or "/some/wiki"
    if "/wiki/" in path:
        idx = path.index("/wiki/")
        base_prefix = path[:idx]
        title = path[idx + len("/wiki/"):]
    else:
        # index.php?title=Foo style (also /w/index.php?title=Foo)
        q = parse_qs(parts.query)
        if q.get("title"):
            title = q["title"][0]
            base_prefix = path.rsplit("/", 1)[0] if "/" in path else ""

    title = unquote(title or "").strip()
    if not title or "/" in title or title.startswith(_NON_ARTICLE_PREFIXES):
        return None

    origin = f"{parts.scheme}://{parts.netloc}"
    candidates: list[str] = []
    for prefix in (base_prefix, f"{base_prefix}/w", "/w", ""):
        api = f"{origin}{prefix}/api.php"
        if api not in candidates:
            candidates.append(api)
    return title, candidates


def _try_mediawiki_api(url: str, timeout: int) -> str | None:
    """Try to render `url` via the MediaWiki action API. Returns markdown on success, or None
    if the URL isn't a wiki article or no api.php candidate responds with parsed content."""
    targets = _mediawiki_targets(url)
    if targets is None:
        return None
    title, candidates = targets
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
            for api in candidates:
                try:
                    resp = client.get(api, params=params)
                except httpx.HTTPError:
                    continue  # try the next candidate location
                if resp.status_code != 200:
                    continue
                try:
                    data = resp.json()
                except (json.JSONDecodeError, ValueError):
                    continue  # not the API (likely an HTML 404 page) — try next
                parse = data.get("parse") if isinstance(data, dict) else None
                html = parse.get("text") if isinstance(parse, dict) else None
                if isinstance(html, str) and html.strip():
                    display = parse.get("title") or title
                    return _html_to_markdown(html, title=str(display))
                # Valid API JSON but no content (e.g. {"error": "missingtitle"}): it IS MediaWiki,
                # so don't keep probing other paths — let the caller fall back to the HTML fetch.
                if isinstance(data, dict) and ("error" in data or "parse" in data):
                    return None
    except Exception:
        return None
    return None


def _parse_sections(lines: list[str]) -> list[Section]:
    """ATX-heading outline. Skips content inside fenced code blocks."""
    sections: list[Section] = []
    in_fence = False
    for idx, line in enumerate(lines):
        if _FENCE_RX.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RX.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        sections.append(Section(level=level, title=title, start_line=idx + 1, end_line=idx + 1))
    # Walk back through sections to compute end_line as "right before next heading
    # at same or shallower depth".
    for i, s in enumerate(sections):
        end = len(lines)
        for j in range(i + 1, len(sections)):
            if sections[j].level <= s.level:
                end = sections[j].start_line - 1
                break
        s.end_line = end
    return sections
