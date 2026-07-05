"""Screenshot tool — capture the user's real screen (fullscreen) or a specific window.

OS-level capture on the app host (the same machine as Unity), so no Unity round-trip:
  * fullscreen → Pillow's ImageGrab (whole virtual desktop, all monitors).
  * window     → pywin32 PrintWindow with PW_RENDERFULLCONTENT, which grabs a window's real
                 contents even when it's occluded by other windows or minimized.

The captured image rides the next LLM round as a vision attachment (same path as Read /
LookAtYourself), so this is only registered on vision-capable backends.

Privacy: a desktop/window capture can expose anything on screen (passwords, private messages),
so every capture goes through the approval gate (Allow once / Allow this session / Deny).
"""
from __future__ import annotations

import asyncio
import base64
import io

from .approval import ApprovalRequest
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


# Capture + encode is local (no WS), but guard against a hung GDI call anyway.
_CAPTURE_TIMEOUT_SECONDS = 20.0
# Cap the window list returned on a miss so a busy desktop doesn't flood the context.
_MAX_WINDOW_LIST = 60


class ScreenshotTool(ToolExecutor):
    name = "Screenshot"
    permission = ToolPermission.DANGER_FULL_ACCESS
    activity_label = "Taking a screenshot…"
    defer_until_speech_caught_up = False
    description = (
        "Capture a screenshot of the user's screen so you can see what's on it.\n\n"
        "Modes (via `target`):\n"
        "- \"fullscreen\": the entire desktop (all monitors).\n"
        "- \"window\": a single application window matched by `window_title` (case-insensitive "
        "substring of its title bar). Captures the window's real contents even if it's behind "
        "other windows or minimized.\n\n"
        "If you don't know the exact title, call with target=\"window\" and an empty or approximate "
        "window_title — the result lists the open window titles so you can retry with a real one.\n"
        "The user is asked to approve each capture (it can expose private on-screen content). "
        "On success the next message you receive contains the captured image."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["fullscreen", "window"],
                    "description": "What to capture: the whole desktop, or a single window.",
                },
                "window_title": {
                    "type": "string",
                    "description": "For target=\"window\": a case-insensitive substring of the window's title bar text.",
                },
            },
            "required": ["target"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        if ctx.image_processor is None:
            return ToolResult(
                result_text="Error: image processing helper is not wired up — cannot take a screenshot.",
                error="no image_processor",
            )

        target = str(arguments.get("target") or "").strip().lower()
        if target not in ("fullscreen", "window"):
            return ToolResult(result_text='Error: \'target\' must be "fullscreen" or "window".', error="bad args")
        window_title = str(arguments.get("window_title") or "").strip()

        # Resolve the window first so we can (a) help the AI when nothing matches and (b) show the
        # user the exact window in the approval prompt.
        matched_title: str | None = None
        if target == "window":
            try:
                titles = await asyncio.to_thread(_list_window_titles)
            except Exception as e:
                return ToolResult(result_text=f"Error: could not enumerate windows: {e}", error=str(e))
            matched_title = _match_title(window_title, titles)
            if matched_title is None:
                listing = "\n".join(f"- {t}" for t in titles[:_MAX_WINDOW_LIST])
                hint = "no window_title given" if not window_title else f"no window matches '{window_title}'"
                return ToolResult(
                    result_text=(
                        f"No screenshot taken ({hint}). Open windows you can target:\n{listing}\n\n"
                        "Call again with window_title set to a substring of one of these."
                    ),
                    error="no match",
                )

        # Privacy gate. Scope a "this session" approval to the target so it's remembered per
        # fullscreen / per specific window rather than blanket-approving all screenshots.
        what = "the entire screen" if target == "fullscreen" else f"the window “{matched_title}”"
        scope_key = "fullscreen" if target == "fullscreen" else f"window:{matched_title}"
        decision = await ctx.approval.request(
            ApprovalRequest(
                tool_name=self.name,
                summary=f"Capture a screenshot of {what}",
                details={"target": target, "window": matched_title or ""},
                risk_level="danger",
            ),
            scope_key=scope_key,
        )
        if not decision.allow:
            return ToolResult(result_text=f"Error: the user declined the screenshot of {what}.", error="user-declined")

        # Capture (blocking GDI / ImageGrab work → worker thread).
        try:
            png_bytes, _w, _h = await asyncio.wait_for(
                asyncio.to_thread(_capture, target, matched_title),
                timeout=_CAPTURE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return ToolResult(result_text="Error: the screenshot timed out.", error="timeout")
        except Exception as e:  # never raise out of a tool
            return ToolResult(result_text=f"Error: screenshot failed: {e}", error=str(e))
        if not png_bytes:
            return ToolResult(result_text="Error: the screenshot produced no image.", error="empty")

        # Downscale + JPEG for the vision model (same helper Read's image branch uses).
        jpeg_bytes, jw, jh, perr = await asyncio.to_thread(
            ctx.image_processor, png_bytes,
            ctx.vision_max_edge_pixels, ctx.vision_jpeg_quality,
        )
        if perr or not jpeg_bytes:
            return ToolResult(
                result_text=f"Error: image processing failed: {perr or '(empty result)'}",
                error=perr or "empty",
            )

        out_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        label = "the screen" if target == "fullscreen" else f"the window “{matched_title}”"
        attachment = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Screenshot of {label} ({jw}x{jh}):"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + out_b64}},
            ],
        }
        return ToolResult(
            result_text=f"Captured {label} ({jw}x{jh}). The next message contains the screenshot — describe what you see.",
            pending_attachments=[attachment],
        )


# ---------------------------------------------------------------------------
# Capture helpers (Windows). Run on a worker thread (blocking GDI calls).
# ---------------------------------------------------------------------------

def _list_window_titles() -> list[str]:
    import win32gui

    out: list[str] = []
    seen: set[str] = set()

    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        t = win32gui.GetWindowText(hwnd)
        if t and t.strip() and t not in seen:
            seen.add(t)
            out.append(t)

    win32gui.EnumWindows(_cb, None)
    return out


def _match_title(query: str, titles: list[str]) -> str | None:
    """Case-insensitive: prefer an exact title, fall back to the first substring match."""
    if not query:
        return None
    q = query.lower()
    for t in titles:
        if t.lower() == q:
            return t
    for t in titles:
        if q in t.lower():
            return t
    return None


def _capture(target: str, window_title: str | None):
    if target == "fullscreen":
        return _capture_fullscreen()
    return _capture_window(window_title or "")


def _capture_fullscreen():
    from PIL import ImageGrab

    img = ImageGrab.grab(all_screens=True)
    return _encode_png(img)


def _capture_window(title: str):
    import win32con
    import win32gui
    import win32ui
    from ctypes import windll
    from PIL import Image

    hwnd = _find_hwnd_by_exact_title(title)
    if not hwnd:
        raise RuntimeError(f"window not found: {title}")

    # Restore if minimized so PrintWindow has rendered content to copy.
    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        raise RuntimeError("window has zero size")

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bmp)
    try:
        # PW_RENDERFULLCONTENT = 3 — captures DWM-composited content (Chromium, Electron, most apps).
        windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
        info = bmp.GetInfo()
        bits = bmp.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]), bits, "raw", "BGRX", 0, 1)
    finally:
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
    return _encode_png(img)


def _find_hwnd_by_exact_title(title: str) -> int:
    import win32gui

    found: list[int] = []

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == title:
            found.append(hwnd)

    win32gui.EnumWindows(_cb, None)
    return found[0] if found else 0


def _encode_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), img.width, img.height
