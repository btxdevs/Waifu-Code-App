"""Read tool — port of Assets/Scripts/ChatTools/ReadFileToolExecutor.cs.

Single entry for reading files. Text → cat -n format. Image → either inline OCR
text (when the backend has no vision) or a queued image_url block for the next
LLM round (when supports_vision is on). Vision and OCR paths route through the
app's existing image_proc / ocr controllers via sync helpers exposed
through ToolContext.
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from .read_state import ReadStateEntry
from .workspace import resolve_path, suggest_similar


_MAX_LINES_TO_READ = 2000

_FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier Read tool_result in "
    "this conversation is still current — refer to that instead of re-reading."
)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"}


class ReadTool(ToolExecutor):
    name = "Read"
    permission = ToolPermission.READ_ONLY
    activity_label = "Reading file…"
    defer_until_speech_caught_up = False
    description = (
        "Reads a file from the local filesystem. You can access any file directly by using this tool.\n"
        "Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.\n\n"
        "Usage:\n"
        "- The file_path parameter must be an absolute path, not a relative path\n"
        f"- By default, it reads up to {_MAX_LINES_TO_READ} lines starting from the beginning of the file\n"
        "- You can optionally specify a line offset and limit (especially handy for long files), but it's recommended to read the whole file by not providing these parameters\n"
        "- Results are returned using cat -n format, with line numbers starting at 1\n"
        "- This tool allows you to read image files (png, jpg, jpeg, gif, webp, bmp, tiff). On vision-capable backends, the next message you receive will contain the loaded image(s) — describe what you see. On text-only backends, the image's recognized text is returned via OCR.\n"
        "- This tool can only read files, not directories.\n"
        "- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."
    )

    def __init__(self, max_line_length: int = 2000) -> None:
        self.max_line_length = max_line_length

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The absolute path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "The line number to start reading from. Only provide if the file is too large to read at once",
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "The number of lines to read. Only provide if the file is too large to read at once.",
                    "minimum": 1,
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        raw_path = arguments.get("file_path")
        if not isinstance(raw_path, str):
            raw_path = ""
        resolution = resolve_path(ctx.workspace, raw_path)
        if not resolution.ok:
            return ToolResult(result_text="Error: " + resolution.error, error=resolution.error)

        abs_path = resolution.absolute_path
        if not os.path.exists(abs_path):
            hint = suggest_similar(abs_path)
            return ToolResult(
                result_text=(f"Error: File does not exist: {abs_path}"
                             + (f" Did you mean: {hint}?" if hint else "")),
                error="not found",
            )
        if os.path.isdir(abs_path):
            return ToolResult(
                result_text=f"Error: EISDIR: path is a directory, not a file: {abs_path}",
                error="is a directory",
            )

        ext = os.path.splitext(abs_path)[1].lower()
        if ext in _IMAGE_EXTS:
            if ctx.supports_vision:
                return await self._read_image_vision(abs_path, ctx)
            return await self._read_image_ocr(abs_path, ctx)

        return await self._read_text(abs_path, arguments, ctx)

    # ----------------------------------------------------------------------
    # Image branches
    # ----------------------------------------------------------------------

    async def _read_image_vision(self, abs_path: str, ctx: ToolContext) -> ToolResult:
        """Pillow → resize → JPEG, then queue an image_url user message on
        session.pending_attachments so the next LLM round actually sees the
        picture natively. Tool-role result is a short pointer so the LLM knows
        the image is coming on the next round."""
        if ctx.image_processor is None:
            return ToolResult(
                result_text="Error: image processing helper is not wired up — cannot read image.",
                error="no image_processor",
            )
        try:
            raw = await asyncio.to_thread(_read_bytes, abs_path)
        except OSError as e:
            return ToolResult(result_text=f"Error: read failed: {e}", error=str(e))

        # Hand off to Pillow on a worker thread — keeps the chat loop responsive
        # while big WEBPs / HEICs decode.
        jpeg_bytes, w, h, err = await asyncio.to_thread(
            ctx.image_processor, raw,
            ctx.vision_max_edge_pixels, ctx.vision_jpeg_quality,
        )
        if err or not jpeg_bytes:
            return ToolResult(
                result_text=f"Error: image processing failed: {err or '(empty result)'}",
                error=err or "empty",
            )

        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        # Hand the image to the orchestrator via ToolResult.pending_attachments —
        # `_drain_pending_attachments` consumes that list and appends each entry
        # to session.history right after the tool_result row. The shape is a
        # wire-format dict (content as a list of OpenAI-style blocks) that
        # `ChatMessage.from_wire` translates back into a ChatMessage with
        # content_blocks set, so the next request body carries the image natively.
        attachment = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image returned by Read:"},
                {"type": "text", "text": f"Path: {abs_path} ({w}x{h})"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ],
        }

        # Freshness stamp so a later Edit/Write on this path passes the
        # must-read-first gate.
        ctx.read_state.set(abs_path, ReadStateEntry(
            content=None,
            timestamp=ctx.read_state.mtime_ticks(abs_path),
            offset=1,
            limit=None,
        ))
        return ToolResult(
            result_text=f"Loaded image: {abs_path} ({w}x{h}). The next message contains the image.",
            pending_attachments=[attachment],
        )

    async def _read_image_ocr(self, abs_path: str, ctx: ToolContext) -> ToolResult:
        """Text-only backend fallback: run RapidOCR and return the recognized
        text inline. Slower than the vision path (engine load on first call +
        ~1-3s per image), but works without a vision-capable LLM."""
        if ctx.ocr_processor is None:
            return ToolResult(
                result_text="Error: OCR helper is not wired up — cannot read image without a vision-capable backend.",
                error="no ocr_processor",
            )
        try:
            raw = await asyncio.to_thread(_read_bytes, abs_path)
        except OSError as e:
            return ToolResult(result_text=f"Error: read failed: {e}", error=str(e))
        text, err = await asyncio.to_thread(ctx.ocr_processor, raw)
        ctx.read_state.set(abs_path, ReadStateEntry(
            content=None,
            timestamp=ctx.read_state.mtime_ticks(abs_path),
            offset=1,
            limit=None,
        ))
        if err:
            return ToolResult(result_text=f"Error: OCR failed for {abs_path}: {err}", error=err)
        if not text.strip():
            return ToolResult(result_text=f"OCR result for {abs_path}:\n(no text recognized)")
        return ToolResult(result_text=f"OCR result for {abs_path}:\n{text}")

    async def _read_text(self, abs_path: str, arguments: dict, ctx: ToolContext) -> ToolResult:
        offset = _read_int(arguments, "offset", 1)
        if offset < 0:
            offset = 0
        limit_opt = _read_int_opt(arguments, "limit")
        limit = limit_opt if limit_opt is not None else _MAX_LINES_TO_READ
        if limit < 1:
            limit = 1

        mtime = ctx.read_state.mtime_ticks(abs_path)
        prior = ctx.read_state.get(abs_path)
        if (
            prior is not None
            and prior.offset is not None
            and prior.offset == offset
            and prior.limit == limit_opt
            and prior.timestamp == mtime
        ):
            return ToolResult(result_text=_FILE_UNCHANGED_STUB)

        try:
            file_length = os.path.getsize(abs_path)
        except OSError as e:
            return ToolResult(result_text=f"Error: could not stat file: {e}", error=str(e))

        byte_cap = ctx.workspace.max_read_file_bytes if ctx.workspace else 5 * 1024 * 1024
        size_truncated = file_length > byte_cap
        line_offset = 0 if offset == 0 else offset - 1

        out_lines: list[str] = []
        full_content: list[str] = []
        current_line = 0
        emitted = 0
        total_lines = 0
        reached_end = True
        bytes_read = 0

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace", newline="") as f:
                # Mimic StreamReader.ReadLine: strip the line terminator from the read line.
                buf = ""
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        if buf:
                            current_line += 1
                            total_lines = current_line
                            line = buf
                            buf = ""
                            if current_line > line_offset:
                                self._emit_line(line, current_line, out_lines, full_content)
                                emitted += 1
                                if emitted >= limit:
                                    reached_end = True
                                    break
                        break
                    buf += chunk
                    while True:
                        nl_idx = -1
                        for c in ("\r\n", "\n", "\r"):
                            j = buf.find(c)
                            if j != -1 and (nl_idx == -1 or j < nl_idx):
                                nl_idx = j
                                nl_len = len(c)
                        if nl_idx == -1:
                            break
                        line = buf[:nl_idx]
                        buf = buf[nl_idx + nl_len:]
                        current_line += 1
                        total_lines = current_line
                        bytes_read += len(line.encode("utf-8", errors="replace")) + nl_len
                        if current_line <= line_offset:
                            continue
                        self._emit_line(line, current_line, out_lines, full_content)
                        emitted += 1
                        if emitted >= limit:
                            reached_end = (not buf) and not f.read(1)
                            if not reached_end and limit_opt is not None:
                                # Caller specified a limit — keep counting total lines
                                # so we can tell them how many they didn't see.
                                rest = buf + f.read()
                                for c in rest.splitlines():
                                    total_lines += 1
                                reached_end = True
                            break
                        if size_truncated and bytes_read >= byte_cap:
                            reached_end = False
                            break
                    if emitted >= limit:
                        break
        except OSError as e:
            return ToolResult(result_text=f"Error: failed to read file: {e}", error=str(e))

        if total_lines == 0:
            ctx.read_state.set(abs_path, ReadStateEntry(
                content="", timestamp=mtime, offset=offset, limit=limit_opt,
            ))
            return ToolResult(
                result_text="<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>",
            )
        if emitted == 0:
            return ToolResult(
                result_text=(
                    f"<system-reminder>Warning: the file exists but is shorter than the "
                    f"provided offset ({offset}). The file has {total_lines} lines.</system-reminder>"
                ),
            )
        if (not reached_end) and limit_opt is None:
            out_lines.append(
                f"…[truncated: file exceeded byte cap. Call Read again with offset={current_line + 1} to continue]\n"
            )

        is_full_read = offset <= 1 and limit_opt is None and reached_end
        ctx.read_state.set(abs_path, ReadStateEntry(
            content=("".join(full_content) if is_full_read else None),
            timestamp=mtime,
            offset=offset,
            limit=limit_opt,
        ))
        return ToolResult(result_text="".join(out_lines))

    def _emit_line(self, line: str, current_line: int, out: list[str], full: list[str]) -> None:
        emit = line
        if len(emit) > self.max_line_length:
            emit = emit[: self.max_line_length] + " …[line truncated]"
        out.append(f"{current_line}\t{emit}\n")
        full.append(line + "\n")


def _read_int(args: dict, key: str, fallback: int) -> int:
    v = args.get(key)
    if v is None:
        return fallback
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _read_int_opt(args: dict, key: str) -> int | None:
    v = args.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
