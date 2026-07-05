"""Tool registry + dispatcher.

Port of Assets/Scripts/ChatTools/ClientToolManager.cs. Holds the table of
registered tool executors, builds the per-session schemas the LLM gets, and
dispatches calls coming back from the orchestrator.

The orchestrator interacts with this via `dispatch(name, args, tool_call_id)`,
which is what `ChatOrchestrator.tool_runner` is set to.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .base import ToolContext, ToolExecutor, ToolResult, ToolSchemaEntry, ToolPermission
from .read_state import ReadFileStateTracker

if TYPE_CHECKING:
    from ..orchestrator import ChatSession
    from .approval import ApprovalGate
    from .workspace import WorkspaceConfig


class ToolManager:
    """One per ChatManager. Tools are registered at construction time."""

    def __init__(
        self,
        workspace: "WorkspaceConfig",
        approval: "ApprovalGate",
        chat_manager,
        ask_modal=None,
        image_processor=None,
        ocr_processor=None,
        supports_vision: bool = False,
        vision_max_edge_pixels: int = 2000,
        vision_jpeg_quality: int = 85,
    ):
        self.workspace = workspace
        self.approval = approval
        self.chat_manager = chat_manager
        self.ask_modal = ask_modal
        self.image_processor = image_processor
        self.ocr_processor = ocr_processor
        # Mutable so reload_config can flip vision on/off without rebuilding the
        # whole tool manager when the user toggles supports_vision in settings.
        self.supports_vision = supports_vision
        self.vision_max_edge_pixels = vision_max_edge_pixels
        self.vision_jpeg_quality = vision_jpeg_quality
        self.read_state = ReadFileStateTracker()
        self._tools: dict[str, ToolExecutor] = {}

    def register(self, tool: ToolExecutor) -> None:
        if not tool or not tool.name:
            return
        if tool.name in self._tools:
            print(f"[ToolManager] Duplicate tool '{tool.name}', overwriting.", file=sys.stderr)
        self._tools[tool.name] = tool

    def register_all(self, tools: list[ToolExecutor]) -> None:
        for t in tools:
            self.register(t)

    def has_tool(self, name: str) -> bool:
        return bool(name) and name in self._tools

    def get(self, name: str) -> ToolExecutor | None:
        return self._tools.get(name)

    def activity_label(self, name: str | None) -> str:
        if not name:
            return "Working…"
        t = self._tools.get(name)
        if t and t.activity_label:
            return t.activity_label
        return name + "…"

    def deferred_tools(self) -> set[str]:
        """Returns the set of tool names that should wait for speech to catch up
        before executing. The orchestrator reads this on every round."""
        return {n for n, t in self._tools.items() if t.defer_until_speech_caught_up}

    def concurrency_safe_tools(self) -> set[str]:
        """Names of tools the orchestrator may run in parallel when the LLM emits
        several of them consecutively in one round (see ToolExecutor.concurrency_safe)."""
        return {n for n, t in self._tools.items() if t.concurrency_safe}

    def build_schemas(
        self, session: "ChatSession", allowed: "frozenset[str] | set[str] | None" = None,
    ) -> list[ToolSchemaEntry]:
        """Build the schema list to advertise to the LLM this turn. Tools that
        depend on session state (ChangeOutfit reads outfit names off the active
        character) regenerate every turn. `allowed` restricts the list to those
        names — the sub-agent runner passes its agent's allow-list."""
        out: list[ToolSchemaEntry] = []
        for name, tool in self._tools.items():
            if allowed is not None and name not in allowed:
                continue
            try:
                params = tool.build_schema(session)
            except Exception as e:
                print(f"[ToolManager] Tool '{name}' threw building schema: {e}", file=sys.stderr)
                continue
            if params is None:
                continue
            out.append(ToolSchemaEntry(
                name=tool.name,
                description=tool.description or "",
                parameters=params,
            ))
        return out

    async def dispatch(
        self, name: str, arguments: dict, session: "ChatSession",
        agent_depth: int = 0,
        allowed: "frozenset[str] | set[str] | None" = None,
    ) -> ToolResult:
        """Execute a single tool call. Never raises — turns executor exceptions into
        an Error-prefixed ToolResult so the LLM sees consistent prose.

        `agent_depth` / `allowed` are set by the sub-agent runner (chat.subagents):
        depth rides the ToolContext so the Agent tool can refuse to nest, and the
        allow-list rejects calls outside the sub-agent's tool set (its schema list
        is already filtered, but a model can still hallucinate a name)."""
        if not name:
            return ToolResult(result_text="Error: tool name is empty.", error="empty name")
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                result_text=f"Error: tool '{name}' is not registered on the client.",
                error=f"unknown tool: {name}",
            )
        if allowed is not None and name not in allowed:
            return ToolResult(
                result_text=f"Error: tool '{name}' is not available to this sub-agent.",
                error=f"tool not allowed: {name}",
            )

        ctx = ToolContext(
            session=session,
            workspace=self.workspace,
            read_state=self.read_state,
            approval=self.approval,
            chat_manager=self.chat_manager,
            ask_modal=self.ask_modal,
            image_processor=self.image_processor,
            ocr_processor=self.ocr_processor,
            supports_vision=self.supports_vision,
            vision_max_edge_pixels=self.vision_max_edge_pixels,
            vision_jpeg_quality=self.vision_jpeg_quality,
            tool_manager=self,
            agent_depth=agent_depth,
        )
        try:
            result = await tool.execute(arguments or {}, ctx)
        except Exception as e:
            print(f"[ToolManager] Tool '{name}' raised: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return ToolResult(
                result_text=f"Error: tool '{name}' failed: {e}",
                error=str(e),
            )

        if result is None:
            return ToolResult(result_text="(tool returned no result)")
        return result


__all__ = [
    "ToolManager",
    "ToolPermission",
    "ToolExecutor",
    "ToolResult",
    "ToolContext",
    "ToolSchemaEntry",
]
