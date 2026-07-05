"""HTTP client for an OpenAI-compatible chat completions endpoint (DeepSeek, OpenAI,
local, etc). Port of Assets/Scripts/Chat/Backend/LlmClient.cs.

Both blocking and streaming entry points are provided. The streaming path is what the
orchestrator uses in practice — it parses SSE incrementally and surfaces:

  * content deltas               -> on_token(delta)
  * "content has ended, tool calls are starting" -> on_content_ended()
  * "first tool name was seen"   -> on_tool_preparing(name)
  * the final assembled message  -> return value

Cancellation: every streaming call accepts an optional `asyncio.Event` that, when set,
causes the SSE loop to break early. The orchestrator hooks Chat.Stop to set it. We treat
cancellation the same way the C# version does — fall through and synthesize a "natural"
completion from whatever the SSE parser has so far, so the orchestrator's normal
round-done path commits the partial assistant message.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable

import httpx

from .models import ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolSchema, history_to_wire


# ---------- config ----------

@dataclass
class LlmConfig:
    """Plain settings struct loaded from llm.config.json (see chat.config). Mirrors the
    fields BackendConfig used to carry."""
    api_url: str = "https://api.deepseek.com/chat/completions"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 1.0
    request_timeout_seconds: int = 30
    thinking: str = "unset"  # "unset" | "disabled" | "enabled"
    send_system_prompt_as_user: bool = False
    supports_vision: bool = False
    # How many of the most-recent image-bearing messages PER ORIGIN ("user" composer attachments
    # vs "tool" Read/screenshot outputs) keep their images on the wire. Older ones are dropped to
    # bound context/cost. 0 = send no images at all. Default 10.
    vision_max_images: int = 10


# ---------- public client ----------

class LlmClient:
    """Owns the httpx.AsyncClient + the request body assembly. Stateless across calls
    except for the underlying connection pool."""

    def __init__(self, cfg: LlmConfig):
        self.cfg = cfg
        # Optional `image_resolver(url) -> str | None`, set by ChatManager, that expands
        # `saved-image:` refs (externalized image bytes) into base64 data URLs at request-build
        # time — so images are only read from disk when actually sent. None = no resolution.
        self.image_resolver = None
        # One client for the lifetime of the controller. httpx handles connection reuse.
        # Disable the default 5s connect timeout in favor of the per-request timeout
        # we set ourselves — slow first-token latency on a remote model is normal.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def close(self) -> None:
        await self._client.aclose()

    # ---------- one-shot (non-streaming) ----------

    async def chat_completion(
        self,
        history: list[ChatMessage],
        tools: list[ToolSchema] | None,
    ) -> LlmResponse:
        """Sends one chat completion request and returns the assistant message.
        Tool calls (if any) are inside the returned message's `tool_calls` field; caller
        is responsible for running them and looping."""
        body = self._build_body(history, tools, stream=False)
        timeout = float(self.cfg.request_timeout_seconds) if self.cfg.request_timeout_seconds > 0 else None
        try:
            resp = await self._client.post(
                self.cfg.api_url,
                headers=self._headers(),
                json=body,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise LlmClientError(f"LLM request failed: {e}") from e

        if resp.status_code != 200:
            raise LlmClientError(f"LLM request failed: HTTP {resp.status_code}\n{resp.text}")

        try:
            obj = resp.json()
        except json.JSONDecodeError as e:
            raise LlmClientError(f"LLM response parse error: {e}\nRaw: {resp.text}") from e

        return _parse_response_object(obj)

    # ---------- streaming ----------

    async def stream_chat_completion(
        self,
        history: list[ChatMessage],
        tools: list[ToolSchema] | None,
        on_token: Callable[[str], Awaitable[None] | None] | None = None,
        on_content_ended: Callable[[], Awaitable[None] | None] | None = None,
        on_tool_preparing: Callable[[str], Awaitable[None] | None] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> LlmResponse:
        """Streaming chat completion. Calls callbacks as the SSE stream fires; returns the
        assembled message once the stream ends (or once `cancel_event` is set)."""
        body = self._build_body(history, tools, stream=True)
        # Stream responses can outlast the normal chat-completion timeout; double it as a
        # buffer, matching the C# behavior.
        timeout = (float(self.cfg.request_timeout_seconds) * 2) if self.cfg.request_timeout_seconds > 0 else None
        parser = _SseParser(on_token=on_token, on_content_ended=on_content_ended,
                            on_tool_preparing=on_tool_preparing)

        try:
            async with self._client.stream(
                "POST",
                self.cfg.api_url,
                headers={**self._headers(), "Accept": "text/event-stream"},
                json=body,
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    body_text = raw.decode("utf-8", errors="replace") if raw else ""
                    raise LlmClientError(
                        f"LLM streaming request failed: HTTP {resp.status_code}\n{body_text}"
                    )
                async for line in resp.aiter_lines():
                    if cancel_event is not None and cancel_event.is_set():
                        # Treat cancellation as natural end-of-stream; the orchestrator
                        # commits the partial assistant message via the normal path.
                        break
                    await parser.feed_line(line)
        except httpx.HTTPError as e:
            # If the SSE handler already saw some content, prefer surfacing that as a
            # partial (mirrors C# behavior for aborted requests). For pure connection
            # errors with nothing buffered, surface the error.
            if not parser.has_any_output():
                raise LlmClientError(f"LLM streaming request failed: {e}") from e

        return parser.build_final_message()

    # ---------- internals ----------

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }

    def _build_body(
        self,
        history: list[ChatMessage],
        tools: list[ToolSchema] | None,
        *,
        stream: bool,
    ) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "messages": history_to_wire(
                history,
                system_as_user=self.cfg.send_system_prompt_as_user,
                image_resolver=self.image_resolver,
                max_images_per_origin=self.cfg.vision_max_images,
            ),
            "temperature": self.cfg.temperature,
        }
        if stream:
            body["stream"] = True
        if tools:
            body["tools"] = [t.to_wire() for t in tools]
            body["tool_choice"] = "auto"
        # Thinking-mode passthrough. "unset" omits the field entirely so the model uses
        # its default. The other two values match what the C# version sent.
        if self.cfg.thinking == "disabled":
            body["thinking"] = {"type": "disabled"}
        elif self.cfg.thinking == "enabled":
            body["thinking"] = {"type": "enabled"}
        return body


class LlmClientError(Exception):
    """All transport / parse / status failures raise as this. The orchestrator catches
    and surfaces via Chat.Error."""


# ---------- response parsing ----------

def _parse_response_object(root: dict) -> LlmResponse:
    """OpenAI-shaped response: `{choices: [{message: {...}}]}`. We read choices[0] only."""
    choices = root.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmClientError("LLM response: no 'choices'")
    msg_dict = choices[0].get("message")
    if not isinstance(msg_dict, dict):
        raise LlmClientError("LLM response: no 'message' in choice[0]")
    msg = ChatMessage.from_wire(msg_dict)
    # Some endpoints omit role; force it.
    if not msg.role:
        msg.role = "assistant"
    return LlmResponse(message=msg)


# ---------- SSE parsing ----------

class _SseParser:
    """Incremental SSE -> ChatMessage builder. Mirrors LlmClient.cs:SseStreamHandler.

    Tool calls stream by index — the first chunk for an index carries id/name, later
    chunks add argument-string fragments. We accumulate per-index, then flatten when the
    stream ends.

    Lines passed in via `feed_line` should be one SSE line each (already \\n-split).
    httpx's `aiter_lines()` does that for us, and also strips the trailing newline.
    """

    def __init__(
        self,
        on_token: Callable[[str], Awaitable[None] | None] | None,
        on_content_ended: Callable[[], Awaitable[None] | None] | None,
        on_tool_preparing: Callable[[str], Awaitable[None] | None] | None,
    ):
        self._on_token = on_token
        self._on_content_ended = on_content_ended
        self._on_tool_preparing = on_tool_preparing
        self._content_acc: list[str] = []
        self._reasoning_acc: list[str] = []
        self._tool_calls_by_index: dict[int, _AccumulatedToolCall] = {}
        self._tool_call_seen = False
        self._tool_preparing_fired = False
        self._done = False

    def has_any_output(self) -> bool:
        return bool(self._content_acc) or bool(self._tool_calls_by_index) or bool(self._reasoning_acc)

    async def feed_line(self, line: str) -> None:
        if self._done:
            return
        if not line:
            return
        if line.startswith(":"):
            return  # SSE comment / keepalive
        if not line.startswith("data:"):
            return
        payload = line[5:].lstrip()
        if payload == "[DONE]":
            self._done = True
            return
        if not payload:
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError as e:
            print(f"[LlmClient] SSE chunk parse failed: {e}; payload={payload!r}", file=sys.stderr)
            return
        await self._handle_obj(obj)

    async def _handle_obj(self, obj: dict) -> None:
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return

        # --- content delta ---
        ct = delta.get("content")
        if isinstance(ct, str) and ct:
            self._content_acc.append(ct)
            await _maybe_await(self._on_token, ct)

        # --- reasoning_content delta (DeepSeek reasoner) ---
        rc = delta.get("reasoning_content")
        if isinstance(rc, str) and rc:
            self._reasoning_acc.append(rc)

        # --- tool_calls delta ---
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list) and tcs:
            if not self._tool_call_seen:
                self._tool_call_seen = True
                await _maybe_await(self._on_content_ended)
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                idx = tc.get("index") or 0
                if not isinstance(idx, int):
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        idx = 0
                acc = self._tool_calls_by_index.get(idx)
                if acc is None:
                    acc = _AccumulatedToolCall()
                    self._tool_calls_by_index[idx] = acc
                if (tcid := tc.get("id")):
                    acc.id = str(tcid)
                if (ttype := tc.get("type")):
                    acc.type = str(ttype)
                fn = tc.get("function")
                if isinstance(fn, dict):
                    if (fn_name := fn.get("name")):
                        acc.name = str(fn_name)
                        # First tool-name in the stream — surface the activity label immediately
                        # rather than waiting for SSE end + dispatch.
                        if not self._tool_preparing_fired:
                            self._tool_preparing_fired = True
                            await _maybe_await(self._on_tool_preparing, acc.name)
                    if (args := fn.get("arguments")):
                        acc.args_parts.append(str(args))

    def build_final_message(self) -> LlmResponse:
        msg = ChatMessage(role="assistant", content="".join(self._content_acc))
        if self._reasoning_acc:
            msg.reasoning_content = "".join(self._reasoning_acc)
        if self._tool_calls_by_index:
            tool_calls = []
            for idx in sorted(self._tool_calls_by_index.keys()):
                acc = self._tool_calls_by_index[idx]
                tool_calls.append(ToolCall(
                    id=acc.id,
                    type=acc.type or "function",
                    function=FunctionCall(name=acc.name, arguments="".join(acc.args_parts)),
                ))
            msg.tool_calls = tool_calls
            # OpenAI convention: when tool_calls are present, content is typically null.
            if not msg.content:
                msg.content = None
        return LlmResponse(message=msg)


@dataclass
class _AccumulatedToolCall:
    id: str = ""
    type: str = ""
    name: str = ""
    args_parts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.args_parts is None:
            self.args_parts = []


async def _maybe_await(cb: Callable | None, *args) -> None:
    """Callbacks can be sync or async; either works. Errors are swallowed with a stderr
    log so a buggy callback doesn't tank the stream."""
    if cb is None:
        return
    try:
        result = cb(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception as e:
        print(f"[LlmClient] callback raised: {e}", file=sys.stderr)
