"""Base types for the Python tool system.

Port of Assets/Scripts/ChatTools/IClientToolExecutor.cs + IAsyncClientToolExecutor.cs.
Every tool implements `ToolExecutor`; the `ToolManager` discovers them at startup,
builds JSON schemas for the LLM, and dispatches calls coming back from the orchestrator.

Differences from the C# version:
  * No Unity references. Tools receive a `ToolContext` carrying the live session,
    workspace config, approval gate, and a ChatManager handle (for WriteReport /
    TodoWrite that need to register UI widgets).
  * Execute is async by default. Sync-only tools simply don't await anything.
  * The `Permission` levels match the old ToolPermission enum; the approval gate
    decides whether each call goes through.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..orchestrator import ChatSession
    from .approval import ApprovalGate
    from .read_state import ReadFileStateTracker
    from .workspace import WorkspaceConfig
    # Avoid a hard import to dodge cycles; ChatManager is duck-typed at runtime.
    from ..manager import ChatManager  # noqa: F401


class ToolPermission(str, Enum):
    """Mirror of Assets/Scripts/ChatTools/Workspace/ToolPermission.cs."""
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    DANGER_FULL_ACCESS = "danger_full_access"


@dataclass
class ToolResult:
    """What a tool returns. Matches the shape ChatOrchestrator's tool_runner expects.

    `result_text` is the string fed back into the LLM as the tool message body.
    `session_mutations` carries currentOutfit / currentStatus changes that the
    orchestrator merges into the live session. `pending_attachments` is the
    read-image image_url block queue (only Read uses it today)."""
    result_text: str
    session_mutations: dict | None = None
    pending_attachments: list[dict] | None = None
    error: str | None = None


@dataclass
class ToolContext:
    """Everything a tool needs at execute time. Constructed per call by ToolManager.

    Tools read but generally don't mutate `session` directly — they return
    session_mutations on the ToolResult and the orchestrator merges them. The
    exception is `pending_attachments`, which a tool may append to session
    directly when it has image bytes to ride alongside the next LLM round
    (Read's vision branch does this).
    """
    session: "ChatSession"
    workspace: "WorkspaceConfig"
    read_state: "ReadFileStateTracker"
    approval: "ApprovalGate"
    chat_manager: Any  # ChatManager — duck-typed to avoid a hard import cycle
    # Async modal helpers. `ask_modal` spawns a task window and awaits its reply
    # via an asyncio.Future. app.py provides this; headless tests can pass
    # an async stub that returns a fixed dict.
    ask_modal: Any = None  # Callable[[dict], Awaitable[dict]]
    # Image / OCR helpers used by Read's image branch. `image_processor` decodes,
    # downscales and re-encodes as JPEG (for vision-capable backends — bytes ride
    # the next LLM round as an image_url block). `ocr_processor` runs RapidOCR
    # and returns recognized text (for text-only backends). Both are sync helpers
    # that the tool wraps in asyncio.to_thread.
    image_processor: Any = None  # Callable[[bytes, int, int], tuple[bytes, int, int, str | None]]
    ocr_processor: Any = None    # Callable[[bytes], tuple[str, str | None]]
    # LLM-side vision capability. When True, the Read tool encodes the image and
    # rides it on pending_attachments; when False, it falls back to OCR text.
    supports_vision: bool = False
    vision_max_edge_pixels: int = 2000
    vision_jpeg_quality: int = 85
    # Sub-agent support (chat.subagents). `tool_manager` is the owning ToolManager,
    # so the Agent tool can run a nested tool loop through the same registry.
    # `agent_depth` is 0 for the main character's calls and 1 inside a sub-agent —
    # the Agent tool refuses to nest (depth >= 1).
    tool_manager: Any = None
    agent_depth: int = 0


class ToolExecutor(ABC):
    """Contract every tool implements. Mirrors IClientToolExecutor + IAsyncClientToolExecutor."""

    name: str = ""
    description: str = ""
    permission: ToolPermission = ToolPermission.READ_ONLY
    activity_label: str = "Working…"
    defer_until_speech_caught_up: bool = False
    # Safe to run concurrently with other concurrency-safe calls of the same round
    # (no session mutations, no speech deferral, no ordering side effects). The
    # orchestrator parallelizes consecutive runs of these. Only UwUAgent today.
    concurrency_safe: bool = False

    @abstractmethod
    def build_schema(self, session: "ChatSession") -> dict:
        """Returns the `parameters` JSON-Schema dict the LLM gets for this tool.
        May read session state — e.g., ChangeOutfit enumerates the active character's
        outfits."""

    @abstractmethod
    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        """Run the tool. Must never raise — wrap failures in a ToolResult with an
        `Error:` prefix so the LLM can read what went wrong."""


@dataclass
class ToolSchemaEntry:
    """Wire-format schema entry the orchestrator hands to the LLM."""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)


def _error(message: str) -> ToolResult:
    """Convenience used by tools that want to bail with a single error string."""
    return ToolResult(result_text=f"Error: {message}", error=message)
