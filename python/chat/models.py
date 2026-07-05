"""Wire-format DTOs for the chat orchestrator. Mirrors
Assets/Scripts/Chat/Backend/ChatModels.cs — same field names, same JSON shape on the
OpenAI / DeepSeek wire so we can talk to the endpoint without an SDK.

`ChatMessage` is the one with the tricky shape: it carries either a plain `content` string
OR a list of OpenAI-style content blocks (for vision messages with image_url entries),
mutually exclusive. The `to_wire` / `from_wire` helpers do the multiplexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# ---------- chat history ----------

@dataclass
class ChatMessage:
    """A single entry in the LLM history. OpenAI chat-completions shape:
    `{role, content, tool_calls?, tool_call_id?, reasoning_content?}`.

    `content` and `content_blocks` are mutually exclusive — most messages carry a plain
    `content` string; image-bearing user messages (built by the read_image tool) set
    `content_blocks` instead. `to_wire` emits exactly one of the two.
    """
    role: str = ""
    content: str | None = None
    content_blocks: list[Any] | None = None
    # DeepSeek reasoner returns chain-of-thought in this field; the API requires we echo it
    # back on the next request when the assistant message also has tool_calls.
    reasoning_content: str | None = None
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None
    # Origin of any image blocks on this message ("user" = composer attachment, "tool" = produced
    # by a tool like Read). Drives the per-origin "keep only the latest image" strip policy. Saved
    # (so the policy survives reload) but stripped from the wire before the LLM sees it — it's not
    # part of the OpenAI message shape. See history_to_wire + orchestrator._strip_image_blocks.
    image_source: str = ""
    # Set on the hidden content-blocks user row injected for a KK Aibu touch (the AibuColliderKind
    # name). Marks the row so the renderer's history rebuild shows it as a rewindable tool-activity
    # event (not a bubble, not skipped like a plain hidden row). Persisted; stripped from the wire.
    touch_zone: str = ""
    # Set on the hidden content-blocks user row injected when the USER changes the avatar's outfit
    # from the app's outfit dialog (holds the new outfit's name). Same mechanics as touch_zone: the
    # history rebuild shows the row as a rewindable user_action event. Persisted; stripped from the wire.
    outfit_change: str = ""
    # Set on the hidden content-blocks user row injected when a background UwU helper reports back
    # (holds the ready-made row label, e.g. `UwU helper reported back: "Compare GPU prices"`). Same
    # mechanics as touch_zone / outfit_change. Persisted; stripped from the wire.
    task_done: str = ""

    def to_wire(self) -> dict:
        out: dict = {"role": self.role}
        if self.content_blocks:
            out["content"] = list(self.content_blocks)
        elif self.content is not None:
            out["content"] = self.content
        if self.reasoning_content is not None:
            out["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            out["tool_calls"] = [tc.to_wire() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.image_source:
            out["_image_source"] = self.image_source
        if self.touch_zone:
            out["_touch_zone"] = self.touch_zone
        if self.outfit_change:
            out["_outfit_change"] = self.outfit_change
        if self.task_done:
            out["_task_done"] = self.task_done
        return out

    @classmethod
    def from_wire(cls, data: dict) -> "ChatMessage":
        if not isinstance(data, dict):
            raise TypeError(f"ChatMessage.from_wire expected dict, got {type(data).__name__}")
        msg = cls(role=str(data.get("role") or ""))
        raw_content = data.get("content")
        if isinstance(raw_content, list):
            msg.content_blocks = raw_content
        elif raw_content is not None:
            msg.content = str(raw_content)
        if (rc := data.get("reasoning_content")) is not None:
            msg.reasoning_content = str(rc)
        if (raw_tc := data.get("tool_calls")):
            msg.tool_calls = [ToolCall.from_wire(t) for t in raw_tc if isinstance(t, dict)]
        if (tcid := data.get("tool_call_id")) is not None:
            msg.tool_call_id = str(tcid)
        msg.image_source = str(data.get("_image_source") or "")
        msg.touch_zone = str(data.get("_touch_zone") or "")
        msg.outfit_change = str(data.get("_outfit_change") or "")
        msg.task_done = str(data.get("_task_done") or "")
        return msg

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class FunctionCall:
    name: str = ""
    arguments: str = ""  # JSON-encoded string per OpenAI spec — caller parses

    def to_wire(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_wire(cls, data: dict) -> "FunctionCall":
        return cls(name=str(data.get("name") or ""), arguments=str(data.get("arguments") or ""))


@dataclass
class ToolCall:
    """OpenAI-style tool call emitted by the LLM."""
    id: str = ""
    type: str = "function"
    function: FunctionCall = field(default_factory=FunctionCall)

    def to_wire(self) -> dict:
        return {"id": self.id, "type": self.type, "function": self.function.to_wire()}

    @classmethod
    def from_wire(cls, data: dict) -> "ToolCall":
        fn = data.get("function") or {}
        return cls(
            id=str(data.get("id") or ""),
            type=str(data.get("type") or "function"),
            function=FunctionCall.from_wire(fn if isinstance(fn, dict) else {}),
        )


# ---------- tool schema (advertised to the LLM) ----------

@dataclass
class ToolSchema:
    """Schema sent in the request's `tools` parameter so the LLM knows a tool exists.
    Shape: `{type: "function", function: {name, description, parameters}}`."""
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------- LLM response wrappers ----------

@dataclass
class LlmResponse:
    """Single LLM round's result. Either text reply, or tool calls the client must execute
    (or both — the model can emit content alongside tool_calls)."""
    message: ChatMessage = field(default_factory=ChatMessage)

    @property
    def has_tool_calls(self) -> bool:
        return self.message.has_tool_calls


@dataclass
class EmotionEntry:
    """One emotion label. Matches the C# class so save files round-trip 1:1."""
    label: str = ""


@dataclass
class ExecutedToolCall:
    """Record of a tool call that actually ran during a turn."""
    name: str
    arguments: dict


@dataclass
class StructuredReply:
    """ChatManager-facing summary of a finished turn — final spoken text, the emotions
    in effect at the end, and the tools that ran. The C# version unwraps a JSON `reply`
    field for backwards compatibility; this port does the same in the orchestrator."""
    reply: str = ""
    emotions: list[EmotionEntry] = field(default_factory=list)
    executed_tool_calls: list[ExecutedToolCall] = field(default_factory=list)


# ---------- helpers ----------

# URL prefix marking an image_url block whose bytes live in a side file on disk (see
# save_load.SaveLoadManager). Resolved back to a base64 data URL at LLM-send time.
SAVED_IMAGE_URL_PREFIX = "saved-image:"

# Legacy: older saves replaced evicted images with this exact text in stored history. We now
# evict at send time instead, so filter any leftover instances out of the wire.
_LEGACY_IMAGE_PLACEHOLDER = "[image previously shown — removed to save context]"


def _is_image_block(block) -> bool:
    return isinstance(block, dict) and block.get("type") == "image_url"


def history_to_wire(
    history: Iterable[ChatMessage],
    *,
    system_as_user: bool = False,
    image_resolver: "Any" = None,
    max_images_per_origin: int = 1,
) -> list[dict]:
    """Serialize a chat history for the OpenAI request body. Skips None entries
    defensively — the orchestrator shouldn't be inserting them, but a corrupted save
    shouldn't crash the request build.

    `system_as_user=True` translates every `role: "system"` to `role: "user"` on
    the wire — for backends that don't honor the system role (some local models,
    legacy DeepSeek modes). Storage always keeps "system" so renderer-side history
    rebuilds + rollback logic can identify and skip those entries; this flag only
    affects what the LLM endpoint sees.

    Image policy (applied here, NOT in stored history): per image origin
    (`ChatMessage.image_source`, e.g. "user" vs "tool") only the `max_images_per_origin`
    MOST-RECENT image-bearing messages keep their images on the wire; older ones have their image
    blocks dropped (no placeholder text left behind), to bound context/cost. `max_images_per_origin`
    <= 0 drops all images. `image_resolver(url) -> str | None` expands `saved-image:` refs into
    base64 data URLs, so image bytes are only read from disk for the messages actually sent.
    """
    msgs = [m for m in history if m is not None]
    # Per origin, the indices of image-bearing messages; only the last N per origin keep their images.
    by_origin: dict[str, list[int]] = {}
    for i, m in enumerate(msgs):
        if m.content_blocks and any(_is_image_block(b) for b in m.content_blocks):
            by_origin.setdefault(m.image_source or "", []).append(i)
    keep_idx: set[int] = set()
    n = max(0, int(max_images_per_origin))
    if n:
        for idxs in by_origin.values():
            keep_idx.update(idxs[-n:])

    out: list[dict] = []
    for i, m in enumerate(msgs):
        wire = m.to_wire()
        if system_as_user and wire.get("role") == "system":
            wire["role"] = "user"
        # Strip our persistence-only markers so they never reach the LLM endpoint.
        wire.pop("_image_source", None)
        wire.pop("_touch_zone", None)
        wire.pop("_outfit_change", None)
        wire.pop("_task_done", None)
        _transform_wire_images(wire, image_resolver, keep_images=i in keep_idx)
        out.append(wire)
    return _group_tool_results(out)


def _group_tool_results(wire_msgs: list[dict]) -> list[dict]:
    """Ensure every assistant `tool_calls` message is immediately followed by ALL of its
    `tool` (tool_result) messages, with any image-bearing `user` rows (read_image/screenshot
    results) moved to AFTER the tool block.

    Anthropic-translating backends reject a request where a `tool_use` block isn't immediately
    followed by its `tool_result` ("tool_use ids were found without tool_result blocks immediately
    after"). Image tool results are appended as separate `user` content-blocks messages, so with
    PARALLEL tool calls they can land between two tool results and break that adjacency. The
    producer now defers them, but older saves already hold the interleaved order — this heals it at
    send time (wire-only; stored history is untouched).

    Within a tool round (the run after an assistant-with-tool_calls, up to the next assistant /
    system message) the only rows are tool results and their image attachments, so a stable
    partition — tool rows first, everything else after — is safe."""
    out: list[dict] = []
    i = 0
    n = len(wire_msgs)
    while i < n:
        msg = wire_msgs[i]
        out.append(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            j = i + 1
            run: list[dict] = []
            while j < n and wire_msgs[j].get("role") not in ("assistant", "system"):
                run.append(wire_msgs[j])
                j += 1
            out.extend(m for m in run if m.get("role") == "tool")
            out.extend(m for m in run if m.get("role") != "tool")
            i = j
            continue
        i += 1
    return out


def _transform_wire_images(wire: dict, resolver, keep_images: bool) -> None:
    """Rewrite a wire message's content blocks for the LLM request (copy-on-write so the
    in-memory history is untouched):
      * keep_images=True  → resolve `saved-image:` refs to base64 (drop the block if the file
        is gone);
      * keep_images=False → drop image blocks entirely (this is an older image-bearing message
        of its origin — only the latest keeps its images).
    Also filters out legacy "[image previously shown …]" placeholder text blocks."""
    content = wire.get("content")
    if not isinstance(content, list):
        return
    new_content: list = []
    changed = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" \
                and block.get("text") == _LEGACY_IMAGE_PLACEHOLDER:
            changed = True
            continue  # drop legacy placeholder text
        if _is_image_block(block):
            if not keep_images:
                changed = True
                continue  # older image of this origin — omit the bytes (no placeholder)
            iu = block.get("image_url")
            url = iu.get("url") if isinstance(iu, dict) else ""
            if isinstance(url, str) and url.startswith(SAVED_IMAGE_URL_PREFIX):
                resolved = resolver(url) if resolver is not None else None
                changed = True
                if resolved:
                    new_content.append({"type": "image_url", "image_url": {"url": resolved}})
                # file missing → drop the block silently
                continue
        new_content.append(block)
    if changed:
        wire["content"] = new_content
