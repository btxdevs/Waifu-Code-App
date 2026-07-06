"""Port of Assets/Scripts/Chat/Backend/ChatOrchestrator.cs.

Owns the chat session, builds the system prompt, runs the LLM + tool loop, and emits
typed events ChatManager subscribes to. Pure logic — no IO beyond what's injected:

  * `llm_client`  — does the actual streaming HTTP
  * `tool_runner` — async function that executes a tool by name and returns the result
  * `speech`      — SentenceSpeechPipeline (optional; voice_mode controls whether it runs)
  * `events`      — OrchestratorEvents callback bag (typing/tool-activity/etc.)

The class is constructed once per process. `begin_session` resets per-session state.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .config import ChatBackendConfig
from .llm_client import LlmClient, LlmClientError
from .models import ChatMessage, EmotionEntry, ExecutedToolCall, LlmResponse, StructuredReply, ToolSchema
from .speech import SentenceSpeechPipeline
from .text import rewrite_tags_for_history


# ----------------------------------------------------------------------------
# Character DTOs (built from the app-owned character store)
# ----------------------------------------------------------------------------

@dataclass
class OutfitInfo:
    outfit_name: str
    description: str = ""
    is_default: bool = False
    # KK coordinate id (the NN in "Outfit NN" / the KK_Coordinates.json index). Sent to Unity so the
    # KKOutfitController switches to the matching outfit. -1 = unknown (non-KK / no coordinate).
    index: int = -1


@dataclass
class CharacterInfo:
    """The character fields the orchestrator uses for the system prompt. Built from a
    CharacterRecord in the app-owned character store."""
    # Stable character id (the chat-save key). character_name is just a display label.
    id: str = ""
    character_name: str = ""
    display_name: str = ""
    character_definition: str = ""
    initial_scenario: str = ""
    initial_assistant_message: str = "Hello!"
    system_prompt_template: str = ""
    initial_emotion_label: str = "Neutral"
    # Absolute path to the character's .vrm; forwarded to Unity as Session.Begin.modelPath so it
    # loads + binds the model at session start. Empty for editor-bound / already-present models.
    model_path: str = ""
    default_outfit_name: str = ""
    # Stable coordinate index of the outfit a NEW chat starts in (user-chosen in the
    # editor). -1 = unset → default_outfit_name, then the first outfit.
    default_outfit_index: int = -1
    outfits: list[OutfitInfo] = field(default_factory=list)
    # The character's emotion vocabulary, captured from its model at creation. Authoritative —
    # the LLM is constrained to exactly these labels (no default fallback).
    available_emotions: list[str] = field(default_factory=list)
    # Per-provider TTS voice (see CharacterRecord.voices). Carried on the active session so the
    # TTS path can resolve the right voice for the active provider without re-reading the store.
    voices: dict = field(default_factory=dict)

    def get_outfit(self, name: str) -> OutfitInfo | None:
        if not name:
            return None
        lowered = name.casefold()
        for o in self.outfits:
            if o.outfit_name.casefold() == lowered:
                return o
        return None

    def get_outfit_by_index(self, index: int) -> OutfitInfo | None:
        """Look up an outfit by its stable KK coordinate id (OutfitInfo.index). Preferred over
        get_outfit for restoring saved state, since the outfit's name is user-editable but its
        index isn't. Returns None for index < 0 or no match."""
        if index is None or index < 0:
            return None
        for o in self.outfits:
            if o.index == index:
                return o
        return None

    def get_default_outfit(self) -> OutfitInfo | None:
        # Prefer the stable index (outfit names are user-editable and needn't be unique).
        o = self.get_outfit_by_index(self.default_outfit_index)
        if o is not None:
            return o
        if self.default_outfit_name:
            o = self.get_outfit(self.default_outfit_name)
            if o is not None:
                return o
        return self.outfits[0] if self.outfits else None


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------

@dataclass
class ChatSession:
    """Python mirror of Assets/Scripts/Chat/Backend/ChatSession.cs."""
    character: CharacterInfo = field(default_factory=CharacterInfo)
    user_name: str = "User"
    history: list[ChatMessage] = field(default_factory=list)
    current_outfit: OutfitInfo | None = None
    current_status: str = "Nothing in particular."
    current_emotions: list[EmotionEntry] = field(default_factory=list)
    # Queue of user-role messages waiting to be appended right after the next tool result
    # (used by read_image to ride images alongside the tool round). Each entry is already a
    # complete ChatMessage with role="user" and content_blocks populated.
    pending_attachments: list[ChatMessage] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.character.display_name or self.character.character_name or "Assistant"


# ----------------------------------------------------------------------------
# Save shape (mirrors Unity's ChatSaveData)
# ----------------------------------------------------------------------------

@dataclass
class ChatSaveData:
    character_name: str
    user_name: str
    history: list[ChatMessage]
    current_outfit_name: str
    current_status: str
    emotion_labels: list[str]
    # Stable KK coordinate id of the saved outfit. Preferred over the name on restore (names are
    # user-editable); -1 = unknown (legacy save / non-KK), then the name is used.
    current_outfit_index: int = -1

    def to_dict(self) -> dict:
        return {
            "characterName": self.character_name,
            "userName": self.user_name,
            "history": [m.to_wire() for m in self.history if m is not None],
            "currentOutfit": self.current_outfit_name,
            "currentOutfitIndex": self.current_outfit_index,
            "currentStatus": self.current_status,
            "emotionLabels": list(self.emotion_labels),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSaveData":
        hist_raw = data.get("history") or []
        history = [ChatMessage.from_wire(h) for h in hist_raw if isinstance(h, dict)]
        idx = data.get("currentOutfitIndex")
        return cls(
            character_name=str(data.get("characterName") or ""),
            user_name=str(data.get("userName") or "User"),
            history=history,
            current_outfit_name=str(data.get("currentOutfit") or ""),
            current_outfit_index=int(idx) if isinstance(idx, (int, float)) else -1,
            current_status=str(data.get("currentStatus") or "Nothing in particular."),
            emotion_labels=[str(e) for e in (data.get("emotionLabels") or []) if isinstance(e, str)],
        )


# ----------------------------------------------------------------------------
# Tool RPC contract
# ----------------------------------------------------------------------------

@dataclass
class ToolExecutionResult:
    """What `tool_runner` returns. Mirrors the Tool.Result envelope shape."""
    result_text: str
    session_mutations: dict | None = None  # may carry currentOutfit / currentStatus
    pending_attachments: list[dict] | None = None  # raw wire dicts; orchestrator builds ChatMessages
    error: str | None = None


ToolRunner = Callable[[str, dict, str], Awaitable[ToolExecutionResult]]


# ----------------------------------------------------------------------------
# Event sink
# ----------------------------------------------------------------------------

@dataclass
class OrchestratorEvents:
    """Callbacks ChatManager wires up to push state changes to the renderer / avatar.
    All optional; None means "ignore this event". Errors raised by callbacks are
    swallowed with a stderr log so a buggy subscriber doesn't tank the turn."""
    on_token: Callable[[str], None] | None = None                # streamed content tokens
    on_emotion_changed: Callable[[str], None] | None = None      # canonical label; from EmotionStreamFilter
    on_tool_activity: Callable[[str | None], None] | None = None # "Thinking…" / "Writing file…" / None
    on_executed_tool: Callable[[ExecutedToolCall], None] | None = None  # post-execution; for ChangeOutfit etc.
    on_turn_complete: Callable[[StructuredReply], None] | None = None  # full reply + emotions + tools
    on_error: Callable[[str], None] | None = None


# ----------------------------------------------------------------------------
# The orchestrator
# ----------------------------------------------------------------------------

_OUTFIT_LINE_TEMPLATE = (
    '{{char}} is currently wearing the "{{outfit_name}}" outfit. {{outfit_desc}}'
)

# {{placeholder}} token — double braces, name matched case-insensitively ({{char}} == {{CHAR}}
# == {{Char}}), optional inner whitespace ({{ user }}).
_PLACEHOLDER_REGEX = re.compile(r"\{\{\s*([a-z0-9_]+)\s*\}\}", re.IGNORECASE)


def _apply_placeholders(text: str | None, values: dict[str, str]) -> str:
    """Substitute {{name}} tokens in `text` from `values` (keys lowercase). Names are
    matched case-insensitively; unknown names are left untouched so stray braces in
    persona text don't get eaten. Substituted values are NOT re-scanned, so a user
    name like "{{char}}" can't recurse."""
    if not text:
        return text or ""
    return _PLACEHOLDER_REGEX.sub(
        lambda m: values.get(m.group(1).lower(), m.group(0)), text
    )

# The system prompt template lives in an editable text file (CompanionApp/system_prompt.txt)
# so it can be tweaked without touching code. Read fresh on every call so edits apply on the
# next turn without a restart.
# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
from .app_paths import APP_ROOT
_SYSTEM_PROMPT_PATH = APP_ROOT / "system_prompt.txt"


def _load_system_prompt_template() -> str:
    """Read the default template from CompanionApp/system_prompt.txt. Logs and
    returns an empty string if the file is missing / unreadable — an empty system
    prompt is an obvious signal something's wrong without crashing the turn."""
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip("\n")
    except (OSError, UnicodeDecodeError) as e:
        print(f"[ChatOrchestrator] could not read {_SYSTEM_PROMPT_PATH}: {e}", file=sys.stderr)
        return ""

# Replaces any run of 2+ consecutive line breaks (handling \r\n, \n, \r) with a single \n.
_BLANK_LINE_REGEX = re.compile(r"(?:\r?\n){2,}")

# One inline [LABEL] tag (no brackets inside). Used to pull the final emotion out of a
# canonicalized assistant message so session.current_emotions tracks where the turn ended.
_EMOTION_TAG_REGEX = re.compile(r"\[([^\[\]]*)\]")


class ChatOrchestrator:
    """Owns the chat session, runs the LLM + tool loop."""

    def __init__(
        self,
        config: ChatBackendConfig,
        llm_client: LlmClient,
        tool_runner: ToolRunner,
        tool_schemas: list[ToolSchema] | None = None,
        speech: SentenceSpeechPipeline | None = None,
        workspace_root_getter: Callable[[], str | None] | None = None,
        memory_getter: Callable[["ChatSession", str], str] | None = None,
        activity_label_resolver: Callable[[str | None], str] | None = None,
        deferred_tools_getter: Callable[[], set] | None = None,
        concurrency_safe_getter: Callable[[], set] | None = None,
    ):
        self.config = config
        self.llm = llm_client
        self.tool_runner = tool_runner
        self.tool_schemas = list(tool_schemas) if tool_schemas else []
        self.speech = speech
        # Maps a tool name → the "Writing report…" style label shown live while it runs.
        # Backed by the ToolManager (each tool's own `activity_label`) so there's a single
        # source of truth; falls back to a generic "<name>…" when no resolver is wired.
        self._resolve_activity_label = activity_label_resolver or _activity_label_for
        # Live views of the ToolManager's per-tool flags: defer-until-speech-caught-up
        # (ChangeOutfit / ReportWrite / AskUserQuestion wait for the narration) and
        # concurrency-safe (consecutive UwUAgent calls run in parallel). Getters so the
        # tools' own attributes stay the single source of truth; the module constants
        # below are the no-getter fallbacks.
        self._deferred_tools_getter = deferred_tools_getter
        self._concurrency_safe_getter = concurrency_safe_getter
        # Called when building the system prompt's {{environment}} line — lets us
        # show the workspace root (the LLM's actual working surface) instead of
        # the Python process cwd. Optional; falls back to os.getcwd().
        self._workspace_root_getter = workspace_root_getter
        # Renders the {{memory}} section (character + project long-term memory). Takes the live session
        # and a relevance query (the current user message) → the prompt block. Optional.
        self._memory_getter = memory_getter
        # The relevance query the memory getter scores against — set to the user's message at the top
        # of each turn so the surfaced memory bodies match what they just said.
        self._memory_query = ""
        self.events = OrchestratorEvents()
        self.session: ChatSession | None = None
        # Set during a turn; the orchestrator monitors it to short-circuit on Chat.Stop.
        self._cancel_event: asyncio.Event | None = None

    # ------------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------------

    def begin_session(self, character: CharacterInfo, user_name: str) -> ChatSession:
        """Start a fresh session for the given character. Builds the initial history
        (system prompt + scenario) in memory. No LLM call yet, and the greeting is NOT
        seeded here — it's delivered as the assistant's first response once the chat is
        ready (see ChatManager._deliver_greeting), which appends it to history then."""
        session = ChatSession(
            character=character,
            user_name=user_name or "User",
            current_outfit=character.get_default_outfit(),
            current_status="Nothing in particular.",
            current_emotions=[EmotionEntry(label=character.initial_emotion_label or "Neutral")],
        )
        session.history.append(ChatMessage(role="system", content=self._build_system_prompt(session)))
        if character.initial_scenario:
            session.history.append(ChatMessage(
                role="system",
                content="Initial scene context: " + self._substitute_persona(character.initial_scenario, session),
            ))
        self.session = session
        return session

    def make_greeting_message(self, session: ChatSession) -> ChatMessage | None:
        """Build the character's opening line as an assistant ChatMessage ("[Label]\\ngreeting", with
        {{char}}/{{user}} substituted), or None when the character has no greeting. NOT appended to history —
        the caller delivers it as a fresh response and appends it then, so it behaves like any other
        assistant turn rather than a pre-seeded bubble."""
        greeting = self._substitute_persona(session.character.initial_assistant_message or "", session)
        if not greeting.strip():
            return None
        label = session.current_emotions[0].label if session.current_emotions else "Neutral"
        return ChatMessage(role="assistant", content=f"[{label}]\n{greeting}")

    def resume_session(self, character: CharacterInfo, saved: ChatSaveData) -> ChatSession:
        """Restore a session from a save. Falls back to the character's default outfit when
        the saved outfit can't be resolved."""
        # Prefer the stable index (outfit names are user-editable); fall back to name, then default.
        outfit = (character.get_outfit_by_index(saved.current_outfit_index)
                  or character.get_outfit(saved.current_outfit_name)
                  or character.get_default_outfit())
        emotions: list[EmotionEntry] = [EmotionEntry(label=lbl) for lbl in saved.emotion_labels if lbl]
        if not emotions:
            emotions.append(EmotionEntry(label=character.initial_emotion_label or "Neutral"))
        session = ChatSession(
            character=character,
            user_name=saved.user_name or "User",
            current_outfit=outfit,
            current_status=saved.current_status or "Nothing in particular.",
            current_emotions=emotions,
            history=list(saved.history),
        )
        _migrate_legacy_system_as_user(session.history)
        if not session.history:
            # Corrupt save — fall back to a fresh seed so the next LLM call has a system prompt.
            session.history.append(ChatMessage(role="system", content=self._build_system_prompt(session)))
            if character.initial_scenario:
                session.history.append(ChatMessage(
                    role="system",
                    content="Initial scene context: " + self._substitute_persona(character.initial_scenario, session),
                ))
        self.session = session
        return session

    def restart_session(self) -> None:
        """Clear conversation history; reseed system prompt + scenario. The greeting is delivered
        afterwards as the assistant's first response (see ChatManager._on_restart), same as a new
        chat — not pre-seeded here."""
        if self.session is None:
            return
        s = self.session
        s.history.clear()
        s.history.append(ChatMessage(role="system", content=self._build_system_prompt(s)))
        if s.character.initial_scenario:
            s.history.append(ChatMessage(
                role="system",
                content="Initial scene context: " + self._substitute_persona(s.character.initial_scenario, s),
            ))

    def end_session(self) -> None:
        """Tear down the active session, leaving the orchestrator with no character loaded.
        The next begin_session / resume_session starts fresh. Used when the user deletes the
        chat they're currently in."""
        self.session = None

    def cancel(self) -> None:
        """Abort the current LLM stream (if any). The submit-message loop falls through
        with whatever partial content has accumulated, committing it as the assistant
        message in history — matches the C# Cancel + natural-completion behavior."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    # ------------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------------

    async def submit_message(
        self,
        user_message: str,
        attachment_messages: list[ChatMessage] | None = None,
        hidden: bool = False,
        touch_zone: str = "",
        outfit_change: str = "",
        task_done: str = "",
    ) -> StructuredReply | None:
        """Run one turn message through the LLM + tool loop and return the assembled
        reply. Streams content tokens via `events.on_token` as they arrive. Returns
        None if the orchestrator failed to produce a final reply (an error is
        surfaced via `events.on_error` first).

        `attachment_messages` are extra content-blocks user rows (e.g. attached images)
        appended right after the turn message so the LLM sees them inline. They carry no
        plain `content`, so the history view skips them (same shape as Read's image rows).

        `hidden=True` injects an INVISIBLE turn (e.g. a touch stage-direction): the message is
        stored as a content-blocks user row — a single text block, no plain `content` — so the
        renderer's history rebuild skips it (same shape as Read's image rows) while the LLM still
        sees it. It stays a `user` row on the wire (a mid-conversation `system` message is rejected
        by backends like Gemini), so the turn runs and replies exactly like a typed message.

        `touch_zone` (with `hidden=True`) tags the injected row as a KK Aibu touch on that zone, so the
        renderer shows it as a rewindable tool-activity event instead of skipping it.

        `outfit_change` (with `hidden=True`) tags the injected row as a user-initiated outfit change
        (the new outfit's name) — same rendering mechanics as touch_zone.

        `task_done` (with `hidden=True`) tags the injected row as a background UwU helper's report
        landing (carries the ready-made row label) — same rendering mechanics as touch_zone.
        """
        if self.session is None:
            self._fire_error("No active session. Call begin_session first.")
            return None
        if not user_message or not user_message.strip():
            self._fire_error("Turn message is empty.")
            return None

        # Score memory relevance against this turn's message (used by _build_system_prompt below).
        self._memory_query = user_message
        self._refresh_system_prompt()
        if hidden:
            self.session.history.append(ChatMessage(
                role="user",
                content_blocks=[{"type": "text", "text": user_message}],
                touch_zone=touch_zone,
                outfit_change=outfit_change,
                task_done=task_done,
            ))
        else:
            self.session.history.append(ChatMessage(role="user", content=user_message))
        # Append this turn's attached images (already tagged image_source="user"). Eviction of
        # older images happens at send time (history_to_wire), so nothing is rewritten here.
        for am in attachment_messages or []:
            self.session.history.append(am)

        self._cancel_event = asyncio.Event()
        executed_tools: list[ExecutedToolCall] = []
        accumulated: list[str] = []
        final_message: ChatMessage | None = None
        max_rounds = self.config.max_tool_call_rounds or 0
        round_idx = 0

        while True:
            if max_rounds > 0 and round_idx >= max_rounds:
                break
            round_idx += 1

            # Surface "Thinking…" while we wait for the first delta (the on_token /
            # on_tool_preparing callbacks below clear it as soon as data lands).
            self._fire_tool_activity("Thinking…")
            first_token_fired = False
            tool_preparing_fired = False
            content_before_round = sum(len(s) for s in accumulated)

            def _on_token(delta: str) -> None:
                nonlocal first_token_fired
                if not first_token_fired:
                    first_token_fired = True
                    self._fire_tool_activity(None)
                # Inject a space between previous-round and this-round content so they
                # don't run together ("Let me check.The answer is…"). Mirrors the C# path.
                if content_before_round > 0 and sum(len(s) for s in accumulated) == content_before_round:
                    self._track_token(" ", accumulated)
                self._track_token(delta, accumulated)

            def _on_tool_preparing(tool_name: str) -> None:
                nonlocal tool_preparing_fired
                tool_preparing_fired = True
                self._fire_tool_activity(self._resolve_activity_label(tool_name))

            def _on_content_ended() -> None:
                # The LLM transitioned content → tool_calls. Flush any partial sentence
                # into the speech pipeline so TTS doesn't stall mid-narration while the
                # tool runs.
                if self.speech is not None:
                    self.speech.flush_text_buffer()

            try:
                response = await self.llm.stream_chat_completion(
                    history=self.session.history,
                    tools=self.tool_schemas,
                    on_token=_on_token,
                    on_content_ended=_on_content_ended,
                    on_tool_preparing=_on_tool_preparing,
                    cancel_event=self._cancel_event,
                )
            except LlmClientError as e:
                self._fire_tool_activity(None)
                self._fire_error(str(e))
                return None
            finally:
                if not first_token_fired and not tool_preparing_fired:
                    # SSE ended without ever firing — clear the still-showing "Thinking…".
                    self._fire_tool_activity(None)

            self._canonicalize_emotion_tags(response.message)
            self.session.history.append(response.message)

            if not response.has_tool_calls:
                final_message = response.message
                break

            # Flush partial sentence into speech BEFORE running tools so audio doesn't
            # stall during tool execution (no trailing whitespace after the LLM's last
            # punctuation — SentenceSplitter would otherwise leave it buffered).
            if self.speech is not None:
                self.speech.flush_text_buffer()

            # Image-bearing tool results (read_image, screenshot) come back as separate `user`
            # content-blocks messages. They MUST be appended only AFTER every tool_result for this
            # round, never interleaved between them: an Anthropic-translating backend requires all
            # `tool_result` blocks to immediately follow the assistant's `tool_use` blocks, so a
            # user image message wedged between two parallel tool results triggers a 400
            # ("tool_use ids ... without tool_result blocks immediately after"). Collect here, drain
            # once the loop is done.
            round_attachments: list = []
            # Consecutive concurrency-safe calls (UwUAgent workers) run in parallel;
            # everything else runs alone, in call order. Tool rows land in call
            # order either way.
            for group in self._partition_tool_calls(response.message.tool_calls or []):
                if len(group) == 1:
                    await self._execute_tool_call(group[0], executed_tools, round_attachments)
                else:
                    await self._execute_parallel_group(group, executed_tools, round_attachments)

            # All tool_result rows for this round are now in place — append any image attachments
            # after them (see round_attachments note above).
            self._drain_pending_attachments(round_attachments)

        if final_message is None:
            limit_desc = f"{max_rounds} tool rounds" if max_rounds > 0 else "the tool round limit"
            self._fire_error(f"LLM did not produce a final text reply within {limit_desc}.")
            return None

        # Record where this turn ended so a later rollback restores this emotion (not the
        # stale initial label). Must run before _build_final_reply, which snapshots
        # current_emotions into the StructuredReply.
        self._update_current_emotion_from_message(final_message)

        reply = self._build_final_reply("".join(accumulated))
        reply.executed_tool_calls = executed_tools
        self._fire_turn_complete(reply)
        return reply

    # ------------------------------------------------------------------------
    # Tool-call execution (one round's calls, sequential + parallel groups)
    # ------------------------------------------------------------------------

    def _deferred_tool_names(self) -> set:
        """Names of tools that wait for the speech pipeline to catch up before
        executing (their visible side effect should land on the narration — outfit
        swap, report panel, question modal)."""
        if self._deferred_tools_getter is not None:
            try:
                return set(self._deferred_tools_getter() or ())
            except Exception as e:
                print(f"[ChatOrchestrator] deferred_tools_getter raised: {e}", file=sys.stderr)
        return _DEFERRED_TOOLS

    def _concurrency_safe_names(self) -> set:
        """Names of tools that may run in parallel with each other (see
        ToolExecutor.concurrency_safe)."""
        if self._concurrency_safe_getter is not None:
            try:
                return set(self._concurrency_safe_getter() or ())
            except Exception as e:
                print(f"[ChatOrchestrator] concurrency_safe_getter raised: {e}", file=sys.stderr)
        return _CONCURRENCY_SAFE_TOOLS

    def _partition_tool_calls(self, tool_calls: list) -> list[list]:
        """Split one round's tool calls into ordered execution groups: a consecutive
        run of concurrency-safe calls becomes one parallel group; every other call is
        its own single-entry group."""
        safe = self._concurrency_safe_names()
        groups: list[list] = []
        for tc in tool_calls:
            if (groups
                    and tc.function.name in safe
                    and groups[-1][0].function.name in safe):
                groups[-1].append(tc)
            else:
                groups.append([tc])
        return groups

    def _parse_tool_args(self, tool_call) -> dict | None:
        """Parse a call's JSON arguments. None = unparseable — the caller appends the
        error row itself so it lands in call order."""
        args_text = tool_call.function.arguments or ""
        try:
            return json.loads(args_text) if args_text else {}
        except json.JSONDecodeError as e:
            print(f"[ChatOrchestrator] Tool arg parse failed for "
                  f"'{tool_call.function.name}': {e}", file=sys.stderr)
            return None

    def _append_bad_args_row(self, tool_call) -> None:
        self.session.history.append(ChatMessage(
            role="tool", content="Error: invalid arguments.",
            tool_call_id=tool_call.id,
        ))

    async def _run_tool(self, name: str, args_obj: dict, call_id: str) -> ToolExecutionResult:
        """tool_runner with the never-raise guarantee (CancelledError excepted, so a
        user Stop still aborts the turn)."""
        try:
            return await self.tool_runner(name, args_obj, call_id)
        except Exception as e:
            print(f"[ChatOrchestrator] tool_runner raised for '{name}': {e}", file=sys.stderr)
            return ToolExecutionResult(
                result_text=f"Error: tool '{name}' failed: {e}",
                error=str(e),
            )

    def _commit_tool_result(self, tool_call, args_obj: dict, result: ToolExecutionResult,
                            executed_tools: list, round_attachments: list) -> None:
        """Post-execution bookkeeping for one call: session mutations, system-prompt
        refresh, the tool row, attachment collection, executed-tool event."""
        name = tool_call.function.name
        executed_tools.append(ExecutedToolCall(name=name, arguments=args_obj))
        self._apply_session_mutations(result.session_mutations)
        self._refresh_system_prompt()
        self.session.history.append(ChatMessage(
            role="tool",
            content=result.result_text or "(tool returned no result)",
            tool_call_id=tool_call.id,
        ))
        if result.pending_attachments:
            round_attachments.extend(result.pending_attachments)
        self._fire_executed_tool(executed_tools[-1])

    async def _execute_tool_call(self, tool_call, executed_tools: list,
                                 round_attachments: list) -> None:
        """Run one tool call inline (the sequential path)."""
        name = tool_call.function.name
        args_obj = self._parse_tool_args(tool_call)
        if args_obj is None:
            self._append_bad_args_row(tool_call)
            return

        # Defer-until-speech-caught-up: tools whose visible side effect should land
        # on the narration wait for the playback buffer to drain what's already
        # been written.
        if self.speech is not None and name in self._deferred_tool_names():
            await self.speech.wait_until_done()

        self._fire_tool_activity(self._resolve_activity_label(name))
        try:
            result = await self._run_tool(name, args_obj, tool_call.id)
        finally:
            self._fire_tool_activity(None)
        self._commit_tool_result(tool_call, args_obj, result, executed_tools, round_attachments)

    async def _execute_parallel_group(self, group: list, executed_tools: list,
                                      round_attachments: list) -> None:
        """Run a consecutive group of concurrency-safe calls (UwUAgent workers)
        concurrently, then commit the results in call order — history rows and
        executed-tool events come out exactly like the sequential path. These tools
        never defer for speech and never mutate the session, so the between-call
        ceremony the sequential path does isn't needed mid-group."""
        prepared = [(tc, self._parse_tool_args(tc)) for tc in group]
        tasks = {
            i: asyncio.create_task(self._run_tool(tc.function.name, args, tc.id))
            for i, (tc, args) in enumerate(prepared) if args is not None
        }
        self._fire_tool_activity(self._resolve_activity_label(group[0].function.name))
        try:
            if tasks:
                # A user Stop cancels the awaiting turn; gather propagates that into
                # every worker task, so the whole group dies together.
                await asyncio.gather(*tasks.values())
        finally:
            self._fire_tool_activity(None)
        for i, (tc, args) in enumerate(prepared):
            if args is None:
                self._append_bad_args_row(tc)
                continue
            self._commit_tool_result(tc, args, tasks[i].result(), executed_tools, round_attachments)

    # ------------------------------------------------------------------------
    # System-prompt assembly
    # ------------------------------------------------------------------------

    def _refresh_system_prompt(self) -> None:
        if self.session is None or not self.session.history:
            return
        self.session.history[0] = ChatMessage(
            role="system",
            content=self._build_system_prompt(self.session),
        )

    def _build_system_prompt(self, session: ChatSession) -> str:
        c = session.character
        template = c.system_prompt_template or _load_system_prompt_template()
        outfit_section = self._build_outfit_section(session)
        memory_section = ""
        if self._memory_getter is not None:
            try:
                memory_section = self._memory_getter(session, self._memory_query) or ""
            except Exception as e:  # noqa: BLE001 — memory must never break prompt assembly
                print(f"[ChatOrchestrator] memory_getter failed: {e}", file=sys.stderr)
        # Values injected into a template (char_def, outfit line, …) are persona-substituted
        # BEFORE injection — _apply_placeholders doesn't re-scan replacements, so {{char}} /
        # {{user}} inside character details must already be resolved when they land here.
        return _apply_placeholders(template, {
            "memory": memory_section,
            "char": c.character_name or "",
            "char_def": self._substitute_persona(c.character_definition, session),
            "user": session.user_name,
            "current_status": session.current_status or "",
            "outfit_section": outfit_section,
            "current_emotion": self._build_emotion_line(session),
            "current_time": self._build_time_line(),
            "environment": self._build_environment_line(),
            "allowed_emotions": self._build_allowed_emotions_str(session),
            "available_tools": self._build_available_tools_str(),
        })

    def _build_outfit_section(self, session: ChatSession) -> str:
        char = session.character
        outfit = session.current_outfit
        available = [o for o in (char.outfits or []) if o.outfit_name]
        # The section owns its "Clothing status:" header (the template just holds
        # {{outfit_section}}), so a character with no outfit data at all — e.g. a VRM model
        # without outfit metadata — renders nothing, header included. Mirrors {{memory}}.
        if not available and (outfit is None or not outfit.outfit_name):
            return ""
        if outfit is None or not outfit.outfit_name:
            current = "You are not wearing any specific outfit."
        else:
            current = _apply_placeholders(_OUTFIT_LINE_TEMPLATE, {
                "char": char.character_name or "",
                "user": session.user_name,
                "outfit_name": outfit.outfit_name,
                # Outfit descriptions are character details too — resolve {{char}}/{{user}} in them.
                "outfit_desc": self._substitute_persona(outfit.description, session),
            }).strip()
        lines = ["Clothing status:", current]
        # When the character has more than one outfit, list the alternatives (with descriptions) so the
        # model knows what it can switch into via the ChangeOutfit tool. With one (or none), the tool
        # isn't offered, so there's nothing to list.
        if len(available) > 1:
            lines += ["", "Your available outfits (switch with the ChangeOutfit tool):"]
            for o in available:
                marker = " (currently worn)" if outfit is not None and o.outfit_name == outfit.outfit_name else ""
                desc = self._substitute_persona(o.description, session).strip()
                lines.append(f'  - "{o.outfit_name}"{marker}' + (f": {desc}" if desc else ""))
        return "\n".join(lines)

    def _build_emotion_line(self, session: ChatSession) -> str:
        if not session.current_emotions:
            return "Neutral"
        labels = [e.label for e in session.current_emotions if e.label]
        return ", ".join(labels) if labels else "Neutral"

    def _build_allowed_emotions_str(self, session: ChatSession) -> str:
        labels = self.effective_emotion_labels(session)
        if not labels:
            return '"Neutral"'
        return ", ".join(f'"{l}"' for l in labels)

    def _build_available_tools_str(self) -> str:
        if not self.tool_schemas:
            return "(none)"
        lines: list[str] = []
        for s in self.tool_schemas:
            if not s or not s.name:
                continue
            line = f"  - {s.name}"
            if s.description:
                line += f": {s.description}"
            lines.append(line)
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _build_time_line() -> str:
        return _dt.datetime.now().strftime("%A, %Y-%m-%d %H:%M")

    def _build_environment_line(self) -> str:
        workspace_root: str | None = None
        if self._workspace_root_getter is not None:
            try:
                workspace_root = self._workspace_root_getter()
            except Exception as e:
                print(f"[ChatOrchestrator] workspace_root_getter raised: {e}", file=sys.stderr)
        return build_environment_line(workspace_root)

    def effective_emotion_labels(self, session: ChatSession | None = None) -> list[str]:
        # The character's emotion vocabulary is authoritative (captured from its model at creation).
        # No default fallback — an empty list means the LLM is told it has no emotion tags to emit.
        s = session or self.session
        return list(s.character.available_emotions) if s is not None else []

    def _substitute_persona(self, text: str | None, session: ChatSession) -> str:
        """Resolve the persona placeholders — {{char}} (character name) and {{user}} (user
        name), case-insensitive — allowed in every character detail field (definition,
        scenario, greeting, outfit descriptions, custom system prompt template)."""
        return _apply_placeholders(text, {
            "char": session.character.character_name or "",
            "user": session.user_name,
        })

    # ------------------------------------------------------------------------
    # Post-round processing
    # ------------------------------------------------------------------------

    def _canonicalize_emotion_tags(self, msg: ChatMessage) -> None:
        """Rewrite [LABEL] tags to canonical form and collapse blank-line runs. Same
        cleanup the C# orchestrator does — keeps the LLM's vocabulary aligned across
        turns and prevents double-newline accumulation."""
        if msg is None or not msg.content:
            return
        labels = self.effective_emotion_labels()
        if labels:
            msg.content = rewrite_tags_for_history(
                msg.content,
                labels,
                on_correction=lambda raw, canonical: print(
                    f"[ChatOrchestrator] Emotion-tag autocorrect: [{raw}] → [{canonical}]", file=sys.stderr
                ),
                on_removed=lambda raw: print(
                    f"[ChatOrchestrator] Removed unknown emotion tag from history: [{raw}]", file=sys.stderr
                ),
            )
        msg.content = _BLANK_LINE_REGEX.sub("\n", msg.content)

    def _update_current_emotion_from_message(self, msg: ChatMessage) -> None:
        """Set session.current_emotions to the LAST [LABEL] tag in a finished assistant
        message. The per-sentence playback markers drive the avatar live, but they never
        write back to the session — so without this, current_emotions stays at the
        initial label for the whole session, and a rollback restores that stale label
        (e.g. "Neutral") instead of where the previous turn actually ended. The tags here
        are already canonical (msg ran through _canonicalize_emotion_tags first)."""
        if self.session is None or msg is None or not msg.content:
            return
        labels = _EMOTION_TAG_REGEX.findall(msg.content)
        last = next((lbl.strip() for lbl in reversed(labels) if lbl.strip()), "")
        if last:
            self.session.current_emotions = [EmotionEntry(label=last)]

    def _apply_session_mutations(self, mutations: dict | None) -> None:
        """Merge tool-reported session changes into the live session. Currently only
        currentOutfit (name → OutfitInfo lookup) and currentStatus."""
        if self.session is None or not mutations:
            return
        if (outfit_name := mutations.get("currentOutfit")):
            new_outfit = self.session.character.get_outfit(str(outfit_name))
            if new_outfit is not None:
                self.session.current_outfit = new_outfit
        if (status := mutations.get("currentStatus")) is not None:
            self.session.current_status = str(status)

    def _drain_pending_attachments(self, attachments: list[dict] | None) -> None:
        """Append any user-role messages the tool queued (image blocks from read_image
        etc.). Mirrors the C# DrainPendingAttachments behavior, including the "at most
        one image-bearing message keeps its bytes" policy: when a new image-bearing
        attachment lands, every prior image_url block in history is rewritten to a
        text placeholder. Saves both in-memory cost (every turn re-sends history) and
        on-disk save size."""
        if self.session is None or not attachments:
            return
        for raw in attachments:
            if not isinstance(raw, dict):
                continue
            try:
                msg = ChatMessage.from_wire(raw)
            except Exception as e:
                print(f"[ChatOrchestrator] failed to parse pending attachment: {e}", file=sys.stderr)
                continue
            if _has_image_blocks(msg):
                # Tag the origin so the send-time policy (history_to_wire) keeps only the latest
                # image per origin — user attachments and tool images are counted separately.
                msg.image_source = msg.image_source or "tool"
            self.session.history.append(msg)

    def _build_final_reply(self, content: str) -> StructuredReply:
        """Pulls the final spoken text + a snapshot of current emotions into a
        StructuredReply. Falls back to unwrapping a JSON object's `reply` field if the
        model (against the prompt) emitted JSON."""
        text = content or ""
        trimmed = text.lstrip()
        if trimmed.startswith("{"):
            try:
                obj = json.loads(trimmed)
                if isinstance(obj, dict) and isinstance(obj.get("reply"), str):
                    text = obj["reply"]
            except json.JSONDecodeError:
                pass

        if self.session is not None and self.session.current_emotions:
            emotions = [EmotionEntry(label=e.label) for e in self.session.current_emotions]
        else:
            emotions = [EmotionEntry(label="Neutral")]

        return StructuredReply(reply=text, emotions=emotions)

    # ------------------------------------------------------------------------
    # Event firing
    # ------------------------------------------------------------------------

    def _track_token(self, token: str, accumulated: list[str]) -> None:
        if not token:
            return
        accumulated.append(token)
        cb = self.events.on_token
        if cb is None:
            return
        try:
            cb(token)
        except Exception as e:
            print(f"[ChatOrchestrator] on_token raised: {e}", file=sys.stderr)

    def _fire_tool_activity(self, label: str | None) -> None:
        cb = self.events.on_tool_activity
        if cb is None:
            return
        try:
            cb(label)
        except Exception as e:
            print(f"[ChatOrchestrator] on_tool_activity raised: {e}", file=sys.stderr)

    def _fire_executed_tool(self, etc: ExecutedToolCall) -> None:
        cb = self.events.on_executed_tool
        if cb is None:
            return
        try:
            cb(etc)
        except Exception as e:
            print(f"[ChatOrchestrator] on_executed_tool raised: {e}", file=sys.stderr)

    def _fire_turn_complete(self, reply: StructuredReply) -> None:
        cb = self.events.on_turn_complete
        if cb is None:
            return
        try:
            cb(reply)
        except Exception as e:
            print(f"[ChatOrchestrator] on_turn_complete raised: {e}", file=sys.stderr)

    def _fire_error(self, message: str) -> None:
        print(f"[ChatOrchestrator] {message}", file=sys.stderr)
        cb = self.events.on_error
        if cb is None:
            return
        try:
            cb(message)
        except Exception as e:
            print(f"[ChatOrchestrator] on_error raised: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def build_environment_line(workspace_root: str | None = None) -> str:
    """Single-line summary of the host the orchestrator (and the file/shell tools) is
    running on. Lets the LLM pick sensible defaults — Bash vs PowerShell, path
    separators, etc. — without guessing from cwd alone. The cwd we surface is
    the workspace root (the LLM's actual working surface for file/shell tools),
    NOT the Python process cwd — those are usually different and the workspace
    is what the LLM should reason about. Module-level so the sub-agent runner
    (chat.subagents) can build the same line for its own system prompt."""
    import os
    import platform
    sys_name = platform.system()
    if sys_name == "Windows":
        # platform.release() returns "10" on Windows 11 because the kernel version
        # is still 10.0.x — you can only tell them apart by build number. Build
        # numbers >= 22000 are Windows 11.
        ver_parts = (platform.version() or "0.0.0").split(".")
        try:
            build = int(ver_parts[-1])
        except (ValueError, IndexError):
            build = 0
        label = f"Windows {'11' if build >= 22000 else (platform.release() or '?')} (build {platform.version()})"
    elif sys_name == "Darwin":
        label = f"macOS {platform.mac_ver()[0] or platform.release()}"
    else:
        label = f"{sys_name} {platform.release()}"
    cwd = workspace_root or os.getcwd()
    return f"{label} ({platform.machine() or 'unknown arch'}); cwd: {cwd}"

# Fallback tool-flag sets, used ONLY when no getter is wired into the orchestrator
# (ChatManager always wires both from the ToolManager, where each tool's own
# `defer_until_speech_caught_up` / `concurrency_safe` attribute is the single source
# of truth — keep these mirrored). _DEFERRED_TOOLS: the visible side effect should
# land on the narration, so the call waits for the speech pipeline to catch up.
# _CONCURRENCY_SAFE_TOOLS: consecutive calls in one round run in parallel.
_DEFERRED_TOOLS = {"ChangeOutfit", "ReportWrite", "AskUserQuestion"}
_CONCURRENCY_SAFE_TOOLS = {"UwUAgent"}


def _activity_label_for(tool_name: str | None) -> str:
    """Generic fallback label when no per-tool resolver is wired (see
    ChatOrchestrator.activity_label_resolver). The real labels come from each tool's
    own `activity_label` via the ToolManager."""
    if not tool_name:
        return "Working…"
    return tool_name + "…"


def _is_image_block(block) -> bool:
    return isinstance(block, dict) and block.get("type") == "image_url"


def _has_image_blocks(msg: ChatMessage) -> bool:
    return bool(msg.content_blocks) and any(_is_image_block(b) for b in msg.content_blocks)


def _migrate_legacy_system_as_user(history: list[ChatMessage]) -> None:
    """Saves created while `send_system_prompt_as_user=True` was the storage shape
    have the system prompt + scenario stored with `role="user"`. Fix them on load
    so the renderer's history rebuild + rollback counter don't see them as real
    user turns. The wire-time translation in `history_to_wire` re-emits them as
    user when the flag is set, so the LLM still sees what it expects.

    Detection: a leading run of "user" rows at indices 0+ until we hit the first
    "assistant" row. The first such row is the system prompt (long, no leading
    "Initial scene context:" prefix); the second (if present) is the scenario,
    prefixed with "Initial scene context: ". Anything that doesn't fit that
    shape is left alone — it's a real user message in an old vision-style save.
    """
    if not history:
        return
    for i, m in enumerate(history):
        if m is None:
            continue
        if m.role == "assistant":
            break  # reached the seeded greeting — stop migration
        if m.role != "user":
            continue
        if not isinstance(m.content, str) or not m.content:
            continue
        # The scenario line is unambiguous.
        if m.content.startswith("Initial scene context: "):
            m.role = "system"
            continue
        # The system prompt is the very first row and is large + carries the
        # [CONTEXT] / [CONSTRAINTS & FORMATTING] markers the default template emits.
        if i == 0 and ("[CONTEXT]" in m.content or "[CONSTRAINTS & FORMATTING]" in m.content):
            m.role = "system"
            continue
