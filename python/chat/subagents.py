"""Sub-agent definitions + runner (the engine behind the UwUAgent tool).

The UwUAgent tool (chat.tools.impl_agent) lets the main character delegate a long,
multi-step task to a focused worker: a fresh LLM loop with its own disposable
history and a restricted tool set. The worker runs silently — no speech, no
emotion tags, no avatar/renderer side effects — and only its FINAL message
comes back to the main conversation as the tool result, so the character's
context stays clean of the intermediate tool noise.

`run_subagent` mirrors the shape of ChatOrchestrator.submit_message's LLM+tool
loop, minus everything session/persona-related. The LlmClient is shared with
the main chat (it's stateless across calls — safe to reuse concurrently), and
tools are dispatched through the same ToolManager with an allow-list plus
agent_depth=1, so a sub-agent can't spawn further sub-agents or reach the
character-facing tools (outfits, memory, reports, todos, modals). Privileged
tool calls (Write / Edit / PowerShell) still hit the shared ApprovalGate, so
the user approves a sub-agent's writes exactly like the character's own.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from .llm_client import LlmClientError
from .models import ChatMessage, ToolSchema
from .orchestrator import _has_image_blocks, build_environment_line

if TYPE_CHECKING:
    from .llm_client import LlmClient
    from .orchestrator import ChatSession
    from .tools.manager import ToolManager


# ----------------------------------------------------------------------------
# Agent definitions
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentDefinition:
    """One summonable helper type. `summary` feeds the UwUAgent tool's schema enum
    docs (how the LLM picks a type); `role` is the worker's own marching orders,
    injected into _BASE_PROMPT. `allowed_tools` are ToolExecutor.name strings — both
    the schema list the worker sees and a dispatch-time allow-list."""
    agent_type: str
    summary: str
    role: str
    allowed_tools: frozenset[str]
    max_rounds: int = 24


# The gimmick: workers are framed as summoned "UwU helper" spirits — but the flavor is
# confined to the framing and ONE optional sign-off line, so report substance stays
# clean and factual (the main character quotes it to the user).
_BASE_PROMPT = """\
You are "{agent_type}", a diligent little UwU helper — a worker spirit summoned by an AI companion to carry out one task autonomously, then vanish.

{role}

Rules:
- Work autonomously — nobody can answer follow-up questions mid-task, so make sensible assumptions and note them.
- Use your tools purposefully; stop as soon as you can answer.
- Your FINAL message is the ONLY thing handed back to the companion that summoned you. Make it a complete, self-contained answer: the findings/results with concrete details (paths, names, numbers, URLs), plus anything the companion must relay to the user. Plain prose/markdown — no greetings, no questions, no offers of further help.
- Keep the report body clear and factual — no cutesy speak in the substance. You MAY end with exactly one short playful sign-off line (e.g. "— your helper, uwu ♡").

Environment: {environment}
Current time: {time}"""


_RESEARCHER = AgentDefinition(
    agent_type="researcher",
    summary="web research — cross-checks several sources and reports back with the key facts + source URLs",
    role=("You research questions using the web. Search with varied queries, open the most "
          "promising results, and cross-check claims across independent sources (prefer primary "
          "ones). Your final report states the answer first, then the key supporting facts, "
          "then the source URLs you actually used."),
    allowed_tools=frozenset({"WebSearch", "WebFetch", "WebPageOutline", "WebPageRead"}),
    max_rounds=20,
)

_EXPLORER = AgentDefinition(
    agent_type="explorer",
    summary="read-only sweep of local files/folders — finds, reads, and summarizes; never modifies anything",
    role=("You explore local files and folders to answer questions about their contents and "
          "structure. You are strictly read-only: never modify anything. Cast a wide net with "
          "Glob/Grep, then read what matters. Report concrete paths and the relevant excerpts "
          "or facts — not just file lists."),
    allowed_tools=frozenset({"Glob", "Grep", "Read"}),
    max_rounds=20,
)

_GENERAL = AgentDefinition(
    agent_type="general",
    summary="full worker for multi-step jobs — files, shell commands, and web; can create and edit files",
    role=("You carry out multi-step tasks end to end using files, PowerShell, and the web. "
          "Verify your work before reporting (re-read what you wrote, check command output). "
          "Your final report says what you did, what you produced (exact paths), and anything "
          "that didn't work."),
    allowed_tools=frozenset({
        "Read", "Write", "Edit", "Glob", "Grep", "PowerShell",
        "WebSearch", "WebFetch", "WebPageOutline", "WebPageRead",
    }),
    max_rounds=40,
)

# Registry the UwUAgent tool resolves `agent_type` against. Insertion order is the
# order the types are listed in the tool schema.
BUILT_IN_AGENTS: dict[str, AgentDefinition] = {
    d.agent_type: d for d in (_RESEARCHER, _EXPLORER, _GENERAL)
}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

@dataclass
class SubagentResult:
    """What run_subagent hands back to the UwUAgent tool. `text` is the worker's final
    assistant message ("" when it never produced one); `error` is set on transport
    failures (LlmClientError) — tool-level errors stay inside the loop as tool rows."""
    text: str = ""
    rounds: int = 0
    tool_calls: int = 0
    error: str | None = None


async def run_subagent(
    defn: AgentDefinition,
    task: str,
    *,
    llm: "LlmClient",
    tool_manager: "ToolManager",
    session: "ChatSession",
    workspace_root: str | None = None,
    on_activity: Callable[[str], None] | None = None,
    verbose: bool = False,
) -> SubagentResult:
    """Run one sub-agent to completion and return its final report.

    A stripped-down clone of ChatOrchestrator.submit_message: fresh two-message
    history, stream + tool-call loop, image-bearing tool results appended after the
    round's tool rows (same ordering rule — an Anthropic-translating backend 400s
    on anything wedged between tool_use and its tool_results). No speech pipeline,
    no emotion canonicalization, no session mutations (a worker changing the
    character's outfit/status would be nonsense — none of its tools report any).

    `on_activity` gets coarse progress strings ("Thinking…", each tool's activity
    label) so the caller can surface what the worker is doing; errors in the
    callback are swallowed. Cancellation (user Stop → task.cancel()) propagates
    out as CancelledError — the caller's normal turn-abort path handles it.
    """
    history: list[ChatMessage] = [
        ChatMessage(role="system", content=_BASE_PROMPT.format(
            agent_type=defn.agent_type,
            role=defn.role,
            environment=build_environment_line(workspace_root),
            time=_dt.datetime.now().strftime("%A, %Y-%m-%d %H:%M"),
        )),
        ChatMessage(role="user", content=task),
    ]
    schemas = [
        ToolSchema(name=e.name, description=e.description, parameters=e.parameters)
        for e in tool_manager.build_schemas(session, allowed=defn.allowed_tools)
    ]

    def _activity(label: str) -> None:
        if on_activity is None:
            return
        try:
            on_activity(label)
        except Exception as e:
            print(f"[UwUAgent] on_activity raised: {e}", file=sys.stderr)

    print(f"[UwUAgent] '{defn.agent_type}' started: {task[:120]!r}", file=sys.stderr)
    result = SubagentResult()
    last_text = ""

    while result.rounds < defn.max_rounds:
        result.rounds += 1
        last_round = result.rounds == defn.max_rounds
        if last_round:
            # Out of budget after this call — force a text-only answer so the caller
            # gets *something* rather than a dangling tool request.
            history.append(ChatMessage(
                role="user",
                content="Tool budget exhausted — write your final report NOW from what you have.",
            ))

        _activity("Thinking…")
        try:
            response = await llm.stream_chat_completion(
                history=history,
                tools=None if last_round else schemas,
            )
        except LlmClientError as e:
            result.error = str(e)
            print(f"[UwUAgent] '{defn.agent_type}' LLM error: {e}", file=sys.stderr)
            return result

        history.append(response.message)
        if not response.has_tool_calls:
            last_text = response.message.content or ""
            break

        # Image-bearing tool results collect here and land only after every tool row
        # of the round (mirrors the orchestrator's round_attachments rule).
        round_attachments: list[dict] = []
        for tool_call in response.message.tool_calls or []:
            name = tool_call.function.name
            args_text = tool_call.function.arguments or ""
            try:
                args_obj = json.loads(args_text) if args_text else {}
            except json.JSONDecodeError as e:
                print(f"[UwUAgent] tool arg parse failed for '{name}': {e}", file=sys.stderr)
                history.append(ChatMessage(
                    role="tool", content="Error: invalid arguments.",
                    tool_call_id=tool_call.id,
                ))
                continue

            result.tool_calls += 1
            _activity(tool_manager.activity_label(name))
            if verbose:
                print(f"[UwUAgent] '{defn.agent_type}' round {result.rounds}: {name}({args_text[:200]})",
                      file=sys.stderr)
            tr = await tool_manager.dispatch(
                name, args_obj, session,
                agent_depth=1, allowed=defn.allowed_tools,
            )
            history.append(ChatMessage(
                role="tool",
                content=tr.result_text or "(tool returned no result)",
                tool_call_id=tool_call.id,
            ))
            if tr.pending_attachments:
                round_attachments.extend(tr.pending_attachments)

        for raw in round_attachments:
            if not isinstance(raw, dict):
                continue
            try:
                msg = ChatMessage.from_wire(raw)
            except Exception as e:
                print(f"[UwUAgent] failed to parse pending attachment: {e}", file=sys.stderr)
                continue
            if _has_image_blocks(msg):
                # Same origin tag the orchestrator applies — history_to_wire's
                # keep-only-the-latest-images policy needs it.
                msg.image_source = msg.image_source or "tool"
            history.append(msg)

    result.text = last_text.strip()
    print(f"[UwUAgent] '{defn.agent_type}' finished: rounds={result.rounds} "
          f"tool_calls={result.tool_calls} chars={len(result.text)}", file=sys.stderr)
    return result
