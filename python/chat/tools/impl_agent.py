"""UwUAgent tool — summon a silent little helper to do a multi-step task.

Thin wrapper over chat.subagents.run_subagent: resolves the requested helper
type from BUILT_IN_AGENTS, runs the worker's own LLM+tool loop to completion
(foreground — the turn waits, like any long tool call), and returns the
worker's final report as the tool result. The worker's tool calls go through
the same ToolManager/ApprovalGate with an allow-list + agent_depth=1, so it
can't summon further helpers or reach character-facing tools.

concurrency_safe=True: when the character summons several helpers in one
round, the orchestrator runs them in parallel (see
ChatOrchestrator._execute_parallel_group).
"""
from __future__ import annotations

import sys

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult
from ..subagents import BUILT_IN_AGENTS, run_subagent


class UwUAgentTool(ToolExecutor):
    name = "UwUAgent"
    # The wrapper itself touches nothing; each of the worker's tool calls re-checks
    # its own permission through the shared ApprovalGate at dispatch time.
    permission = ToolPermission.READ_ONLY
    activity_label = "Summoning an UwU helper…"
    concurrency_safe = True
    description = (
        "Summon an UwU helper — a diligent little worker spirit with its own tools that "
        "silently runs a whole job (many searches / file reads / commands) and hands you "
        "back one final report, then vanishes. Use it when a task needs lots of tool calls "
        "or would flood you with detail: deep web research, digging through folders of "
        "files, a multi-step job. You can summon SEVERAL helpers in one message (one per "
        "independent sub-task) and they all work at the same time. You wait while they "
        "work (the user sees their progress), then relay the findings in your own words."
    )

    def build_schema(self, session) -> dict:
        types = "; ".join(f"'{d.agent_type}' = {d.summary}" for d in BUILT_IN_AGENTS.values())
        return {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "enum": list(BUILT_IN_AGENTS.keys()),
                    "description": f"Which helper to summon: {types}.",
                },
                "description": {
                    "type": "string",
                    "description": ("Very short label of the task (3-6 words), shown to the "
                                    "user while the helper works (e.g. \"Comparing GPU prices\")."),
                },
                "task": {
                    "type": "string",
                    "description": ("Complete, standalone instructions for the helper: the goal, "
                                    "every bit of context it needs (paths, URLs, names — it "
                                    "cannot see this conversation), and exactly what to report "
                                    "back. It cannot ask follow-up questions."),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": ("false (default): wait for the report now. true: the helper "
                                    "works in the background while you keep chatting — its report "
                                    "arrives later as a [System Message] you react to then. Use "
                                    "for long jobs, or when the user shouldn't have to wait."),
                },
            },
            "required": ["agent_type", "description", "task"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        if ctx.agent_depth >= 1:
            return ToolResult(
                result_text="Error: UwU helpers cannot summon more UwU helpers (no nesting).",
                error="nested agent",
            )
        agent_type = str(arguments.get("agent_type") or "").strip()
        defn = BUILT_IN_AGENTS.get(agent_type)
        if defn is None:
            avail = ", ".join(BUILT_IN_AGENTS.keys())
            return ToolResult(
                result_text=f"Error: unknown agent_type '{agent_type}'. Available: {avail}.",
                error="unknown agent_type",
            )
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return ToolResult(result_text="Error: 'task' is required.", error="bad args")
        description = str(arguments.get("description") or "").strip() or defn.agent_type

        llm = getattr(ctx.chat_manager, "llm", None)
        if llm is None or ctx.tool_manager is None:
            return ToolResult(
                result_text="Error: the helper runner is not available (no LLM client / tool manager).",
                error="missing runner deps",
            )

        # Background summon: register the task with the ChatManager and return
        # immediately — the report comes back later as a hidden notification turn
        # (see chat.manager._tasks).
        if bool(arguments.get("run_in_background", False)):
            start = getattr(ctx.chat_manager, "bg_helper_start", None)
            if start is None:
                return ToolResult(
                    result_text="Error: background helpers are not available here.",
                    error="no bg runner",
                )
            tid = start(defn, task.strip(), description)
            return ToolResult(result_text=(
                f"UwU helper '{agent_type}' [{tid}] summoned in the BACKGROUND for "
                f"\"{description}\". It works while you keep chatting; when it finishes, its "
                f"report arrives as a [System Message] and you react to it then. For now, just "
                f"tell the user you've sent a helper off. (CheckUwUHelpers shows its progress; "
                f"DismissUwUHelper cancels it.)"))

        # Live label override: while the helper works (possibly minutes), keep the
        # renderer's activity line on what it's actually doing instead of a frozen
        # "Summoning…". With several helpers in parallel the labels interleave —
        # latest writer wins, which reads as the busiest helper's status. The
        # orchestrator clears the label when the group returns.
        push_activity = getattr(ctx.chat_manager, "push_tool_activity", None)

        def _on_activity(label: str) -> None:
            if push_activity is not None:
                push_activity(f"UwU helper · {description}: {label}")

        result = await run_subagent(
            defn, task.strip(),
            llm=llm,
            tool_manager=ctx.tool_manager,
            session=ctx.session,
            workspace_root=(ctx.workspace.allowed_roots[0]
                            if ctx.workspace is not None and ctx.workspace.allowed_roots else None),
            on_activity=_on_activity,
            verbose=bool(getattr(ctx.chat_manager, "verbose", False)),
        )

        if result.error:
            return ToolResult(
                result_text=f"Error: UwU helper '{agent_type}' failed: {result.error}",
                error=result.error,
            )
        if not result.text:
            return ToolResult(
                result_text=(f"Error: UwU helper '{agent_type}' came back with no final report "
                             f"({result.rounds} rounds, {result.tool_calls} tool calls)."),
                error="empty report",
            )
        header = (f"[UwU helper '{agent_type}' report — {result.rounds} round(s), "
                  f"{result.tool_calls} tool call(s)]")
        return ToolResult(result_text=f"{header}\n{result.text}")


class CheckUwUHelpersTool(ToolExecutor):
    name = "CheckUwUHelpers"
    permission = ToolPermission.READ_ONLY
    activity_label = "Checking on the helpers…"
    description = (
        "Check on your background tasks — UwU helpers AND background commands: which are "
        "still working (and for how long), which finished, and whether their results were "
        "delivered yet. Use when the user asks how a background task is going."
    )

    def build_schema(self, session) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        lines_fn = getattr(ctx.chat_manager, "bg_helper_lines", None)
        if lines_fn is None:
            return ToolResult(result_text="Error: helper status is not available here.",
                              error="no bg runner")
        lines = lines_fn()
        if not lines:
            return ToolResult(result_text="No UwU helpers have been summoned in this session.")
        return ToolResult(result_text="Your UwU helpers (newest first):\n" + "\n".join(lines))


class DismissUwUHelperTool(ToolExecutor):
    name = "DismissUwUHelper"
    permission = ToolPermission.READ_ONLY
    activity_label = "Dismissing a helper…"
    description = (
        "Dismiss (cancel) a background task that is still running — an UwU helper or a "
        "background command — e.g. the user changed their mind, or it's taking too long. "
        "Takes the task id shown when it was started (also listed by CheckUwUHelpers). "
        "A dismissed task never reports back; a dismissed command is killed."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "The helper's task id (e.g. \"a1b2c3d4\")."},
            },
            "required": ["task_id"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        dismiss = getattr(ctx.chat_manager, "bg_helper_dismiss", None)
        if dismiss is None:
            return ToolResult(result_text="Error: helper dismissal is not available here.",
                              error="no bg runner")
        ok, msg = dismiss(str(arguments.get("task_id") or "").strip())
        if not ok:
            return ToolResult(result_text=f"Error: {msg}", error="dismiss failed")
        return ToolResult(result_text=msg)
