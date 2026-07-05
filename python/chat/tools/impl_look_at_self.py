"""LookAtYourself tool — lets the AI see its own avatar.

Captures a screenshot of the avatar's current on-screen view in Unity (RPC over the
app↔Unity WS, served by CharacterDataExposer), downscales it through the same
image pipeline Read uses, and rides it on the next LLM round as a vision attachment.

Vision-only: a screenshot is meaningless to a text-only backend (OCR would find no
text), so this tool requires a vision-capable model and otherwise returns an error.
"""
from __future__ import annotations

import asyncio
import base64

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


# Unity captures after end-of-frame then base64-encodes a JPEG; localhost round-trip is fast,
# but guard against the avatar window being down so the tool can't hang the whole turn.
_CAPTURE_TIMEOUT_SECONDS = 15.0


class LookAtYourselfTool(ToolExecutor):
    name = "LookAtYourself"
    permission = ToolPermission.READ_ONLY
    activity_label = "Looking at myself…"
    defer_until_speech_caught_up = False
    description = (
        "Look at yourself — captures the current view of your avatar on screen and lets you see it.\n\n"
        "Use this when you want to check how you currently look: your appearance, outfit, facial "
        "expression, pose, or what's visible around you. Takes no arguments.\n"
        "The next message you receive will contain the captured image — describe or react to what you see."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        cm = ctx.chat_manager
        if cm is None or not hasattr(cm, "capture_self_view"):
            return ToolResult(
                result_text="Error: self-view capture is not available right now.",
                error="no chat_manager",
            )
        # Vision support is guaranteed by registration (the tool is only registered when the model
        # supports vision — see build_tool_manager), so no need to re-check ctx.supports_vision here.
        if ctx.image_processor is None:
            return ToolResult(
                result_text="Error: image processing helper is not wired up — cannot capture the view.",
                error="no image_processor",
            )

        try:
            reply = await asyncio.wait_for(cm.capture_self_view(), timeout=_CAPTURE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return ToolResult(
                result_text="Error: timed out waiting for the avatar view (is the avatar window running?).",
                error="timeout",
            )
        except Exception as e:  # never raise out of a tool
            return ToolResult(result_text=f"Error: self-view capture failed: {e}", error=str(e))

        reply = reply or {}
        err = str(reply.get("error") or "")
        if err:
            return ToolResult(result_text=f"Error: could not capture the view: {err}", error=err)
        b64 = str(reply.get("base64") or "")
        if not b64:
            return ToolResult(
                result_text="Error: no image was returned — the avatar window may not be connected.",
                error="empty",
            )

        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            return ToolResult(result_text=f"Error: bad image data from the avatar: {e}", error=str(e))

        # Downscale + re-encode as JPEG on a worker thread (same helper Read's vision branch uses).
        jpeg_bytes, w, h, perr = await asyncio.to_thread(
            ctx.image_processor, raw,
            ctx.vision_max_edge_pixels, ctx.vision_jpeg_quality,
        )
        if perr or not jpeg_bytes:
            return ToolResult(
                result_text=f"Error: image processing failed: {perr or '(empty result)'}",
                error=perr or "empty",
            )

        out_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        # Ride the image on the next LLM round as a native vision block (see ReadTool._read_image_vision
        # and orchestrator._drain_pending_attachments for the consuming side).
        attachment = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Your current on-screen view (a screenshot of yourself):"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + out_b64}},
            ],
        }
        return ToolResult(
            result_text=(
                f"Captured your current view ({w}x{h}). The next message contains the image — "
                "look at it and describe/react to how you appear."
            ),
            pending_attachments=[attachment],
        )
