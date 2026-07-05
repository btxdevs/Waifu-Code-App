"""Python port of Assets/Scripts/ChatTools/.

Construction helper `build_tool_manager()` wires every executor into a single
ToolManager. ChatManager calls this once per session at startup; the orchestrator
then drives the manager via `tool_runner`.
"""
from __future__ import annotations

from .approval import ApprovalGate, ApprovalRequest, ApprovalDecision
from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult, ToolSchemaEntry
from .manager import ToolManager
from .read_state import ReadFileStateTracker, ReadStateEntry
from .workspace import WorkspaceConfig, load_workspace_config, resolve_path, default_root
from .web_cache import WebPageReaderCache


def build_tool_manager(
    chat_manager,
    ask_modal=None,
    workspace: WorkspaceConfig | None = None,
    image_processor=None,
    ocr_processor=None,
    supports_vision: bool = False,
    vision_max_edge_pixels: int = 2000,
    vision_jpeg_quality: int = 85,
) -> ToolManager:
    """Constructs a ToolManager with every executor registered.

    `chat_manager` is the live ChatManager instance — duck-typed so we don't
    need a hard import. `ask_modal` is the awaitable callable for spawning task
    windows that resolve a Python future (app.py supplies it). `workspace`
    is the sandbox config; when None we load from app.config.json.
    `image_processor` / `ocr_processor` are the in-process helpers Read's image
    branch uses; the vision settings tell it which branch to take per call.
    """
    from .impl_agent import UwUAgentTool, CheckUwUHelpersTool, DismissUwUHelperTool
    from .impl_ask_question import AskUserQuestionTool
    # from .impl_bash import BashTool  # disabled: the app is Windows-only for now, PowerShell covers shell needs
    from .impl_change_outfit import ChangeOutfitTool
    from .impl_remember import RememberTool, ForgetTool, RecallMemoryTool
    from .impl_edit import EditTool
    from .impl_glob import GlobTool
    from .impl_grep import GrepTool
    from .impl_look_at_self import LookAtYourselfTool
    from .impl_screenshot import ScreenshotTool
    from .impl_open import OpenTool
    from .impl_powershell import PowerShellTool
    from .impl_read import ReadTool
    from .impl_todo_write import TodoWriteTool
    from .impl_web_fetch import WebFetchTool
    from .impl_web_page_outline import WebPageOutlineTool
    from .impl_web_page_read import WebPageReadTool
    from .impl_web_search import WebSearchTool
    from .impl_write import WriteTool
    from .impl_report_write import ReportWriteTool

    ws = workspace if workspace is not None else load_workspace_config()
    # Drop tool-result spill files older than the retention window (see tools.spill).
    from .spill import cleanup_spill_dir
    cleanup_spill_dir()
    # Full-permission mode pairs the gate (no prompts) with the workspace (no path sandbox).
    approval = ApprovalGate(ask_modal=ask_modal, full_access=ws.full_access)
    mgr = ToolManager(
        workspace=ws, approval=approval, chat_manager=chat_manager,
        ask_modal=ask_modal,
        image_processor=image_processor,
        ocr_processor=ocr_processor,
        supports_vision=supports_vision,
        vision_max_edge_pixels=vision_max_edge_pixels,
        vision_jpeg_quality=vision_jpeg_quality,
    )
    web_cache = WebPageReaderCache()
    # These return images as vision attachments — useless on a text-only backend, so only register
    # them when the model can actually see images.
    vision_tools = [LookAtYourselfTool(), ScreenshotTool()] if supports_vision else []
    mgr.register_all([
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
        OpenTool(),
        # BashTool(),  # disabled: Windows-only app — see the import above
        PowerShellTool(),
        TodoWriteTool(),
        ReportWriteTool(),
        # Self-registers only when the character has 2+ outfits (build_schema returns None otherwise),
        # so it's silently absent for single-outfit / VRM characters.
        ChangeOutfitTool(),
        # Long-term memory (character + project scopes).
        RememberTool(),
        ForgetTool(),
        RecallMemoryTool(),
        *vision_tools,
        # Sub-agent delegation ("UwU helpers", chat.subagents). The worker loop
        # dispatches back through this same manager with an allow-list + agent_depth=1.
        # Check/Dismiss manage the background variety (chat.manager._tasks).
        UwUAgentTool(),
        CheckUwUHelpersTool(),
        DismissUwUHelperTool(),
        AskUserQuestionTool(),
        WebFetchTool(web_cache),
        WebPageOutlineTool(web_cache),
        WebPageReadTool(web_cache),
        WebSearchTool(),
    ])
    return mgr


__all__ = [
    "ApprovalGate", "ApprovalRequest", "ApprovalDecision",
    "ReadFileStateTracker", "ReadStateEntry",
    "ToolContext", "ToolExecutor", "ToolManager", "ToolPermission",
    "ToolResult", "ToolSchemaEntry",
    "WorkspaceConfig", "load_workspace_config", "resolve_path", "default_root",
    "build_tool_manager",
]
