"""Background tasks (UwU helpers + shell commands): registry + notification queue.

A helper summoned with run_in_background=true runs its whole run_subagent loop
detached from the turn that summoned it (asyncio.create_task on the chat loop);
a shell command run with run_in_background=true does the same with run_command
(kind="shell" — its "report" is the captured output). The summoning turn returns
immediately; when the task finishes, its report is folded back into the
conversation as a NOTIFICATION TURN — a hidden stage-direction turn
(submit_message(hidden=True, task_done=…)) that the character answers by
relaying the result, exactly like the touch / outfit-change flows.

The one invariant everything here protects: session.history is single-writer.
Only the WORK runs concurrently; the notification turn that appends to history
is serialized behind the same guards every other turn uses (_touch_busy +
_current_turn_task). A helper that finishes mid-turn queues its report; the
queue drains when the in-flight turn's finally block (or the next idle moment)
calls _drain_bg_notifications. Reports queued together are coalesced into ONE
notification turn so three helpers landing at once produce one reaction, not
three monologues.

Tasks are per-session and NOT persisted: _reset_session_bookkeeping (create /
load / restart / end) dismisses everything still running. Chat.Stop cancels
the current TURN only — background helpers keep working through it.
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field

from ..subagents import AgentDefinition, run_subagent, SubagentResult
from .protocol import *  # noqa: F401,F403
from .view_models import HistoryEntry


# Terminal records kept around (for the CheckUwUHelpers tool) after this many
# newer ones exist. Running tasks are never pruned.
_MAX_FINISHED_KEPT = 8


@dataclass
class BackgroundTask:
    """One background helper. `status`: running | completed | failed | dismissed.
    `announced` flips once its report has been folded into a notification turn
    (test-and-set — a task is announced exactly once)."""
    id: str
    kind: str            # agent_type ("researcher" / ...) or "shell"
    label: str           # short user-facing description ("Compare GPU prices")
    task: asyncio.Task
    status: str = "running"
    result: SubagentResult | None = None
    error: str = ""      # transport/crash error when status == "failed"
    announced: bool = False
    started_at: float = field(default_factory=time.monotonic)
    # Wall-clock start (epoch seconds) — sent to the renderer so the status-bar
    # chip can tick a live elapsed time. monotonic stays for runtime_seconds.
    started_wall: float = field(default_factory=time.time)

    @property
    def running(self) -> bool:
        return self.status == "running"

    def runtime_seconds(self) -> int:
        return int(time.monotonic() - self.started_at)


class BgTaskMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    # ------------------------------------------------------------------------
    # Summon / dismiss / inspect (called by the UwUAgent tool family)
    # ------------------------------------------------------------------------

    def bg_helper_start(self, defn: AgentDefinition, task_text: str, label: str) -> str:
        """Launch a background helper and return its task id. Called by the UwUAgent
        tool (run_in_background=true) from the chat loop, so create_task binds to the
        right loop and the done-callback runs on it too."""
        tid = uuid.uuid4().hex[:8]
        runner = run_subagent(
            defn, task_text,
            llm=self.llm,
            tool_manager=self._tool_manager,
            session=self.orchestrator.session,
            workspace_root=self._current_workspace_root(),
            # No live activity labels from background helpers — they'd fight the
            # foreground turn's own activity line. Progress shows in the pill.
            on_activity=None,
            verbose=self.verbose,
        )
        bt = BackgroundTask(id=tid, kind=defn.agent_type, label=label,
                            task=asyncio.ensure_future(runner))
        self._bg_tasks[tid] = bt
        bt.task.add_done_callback(lambda t, _tid=tid: self._on_bg_task_done(_tid, t))
        self._prune_bg_tasks()
        self._push_bg_tasks()
        print(f"[UwUAgent] background helper '{defn.agent_type}' [{tid}] summoned: {label!r}",
              file=sys.stderr)
        return tid

    def bg_shell_start(self, argv: list[str], cwd: str | None, timeout_seconds: int,
                       label: str) -> str:
        """Launch a shell command as a background task (kind="shell") and return its
        task id. Same registry/notification machinery as the helpers — the captured
        output is the "report" the character reacts to. Called by the PowerShell
        tool (run_in_background=true) on the chat loop."""
        # Lazy import — keeps chat.tools out of manager module-load (same convention
        # as _ensure_orchestrator's build_tool_manager import).
        from ..tools.shell import run_command, format_output

        async def _runner() -> SubagentResult:
            res = await run_command(argv, cwd=cwd, timeout_seconds=timeout_seconds)
            if res.error and not res.stdout and not res.stderr:
                # Spawn failure — surfaces as a FAILED announcement.
                return SubagentResult(text="", error=res.error)
            # Non-zero exits / timeouts stay "completed": the output carries the
            # [exit N] / timed-out marker and the character relays what happened.
            return SubagentResult(text=format_output(res))

        tid = uuid.uuid4().hex[:8]
        bt = BackgroundTask(id=tid, kind="shell", label=label,
                            task=asyncio.ensure_future(_runner()))
        self._bg_tasks[tid] = bt
        bt.task.add_done_callback(lambda t, _tid=tid: self._on_bg_task_done(_tid, t))
        self._prune_bg_tasks()
        self._push_bg_tasks()
        print(f"[BgTask] shell command [{tid}] started: {label!r}", file=sys.stderr)
        return tid

    def bg_helper_dismiss(self, task_id: str) -> tuple[bool, str]:
        """Cancel a running helper (the DismissUwUHelper tool). (ok, message)."""
        bt = self._bg_tasks.get(task_id or "")
        if bt is None:
            return False, f"No UwU helper with id '{task_id}'."
        if not bt.running:
            return False, (f"UwU helper [{bt.id}] ('{bt.label}') already finished "
                           f"({bt.status}); nothing to dismiss.")
        bt.status = "dismissed"
        bt.task.cancel()
        self._push_bg_tasks()
        return True, f"Dismissed UwU helper [{bt.id}] ('{bt.label}'). It won't report back."

    def bg_helper_lines(self) -> list[str]:
        """One status line per known helper, newest first (the CheckUwUHelpers tool)."""
        lines: list[str] = []
        for bt in reversed(list(self._bg_tasks.values())):
            if bt.running:
                state = f"running for {bt.runtime_seconds()}s"
            elif bt.status in ("completed", "failed"):
                delivered = "report delivered" if bt.announced else "report pending delivery"
                state = f"{bt.status} ({delivered})"
            else:
                state = bt.status
            lines.append(f"- [{bt.id}] {bt.kind} \"{bt.label}\" — {state}")
        return lines

    def bg_tasks_running(self) -> int:
        return sum(1 for bt in self._bg_tasks.values() if bt.running)

    # ------------------------------------------------------------------------
    # Completion → queue → notification turn
    # ------------------------------------------------------------------------

    def _on_bg_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """Done-callback (runs on the chat loop). Classify the outcome, queue the
        announcement for completed/failed, and try to deliver right away if idle."""
        bt = self._bg_tasks.get(task_id)
        if bt is None:
            return
        if task.cancelled():
            # Dismissed (status already set) or torn down with the session.
            if bt.status == "running":
                bt.status = "dismissed"
        else:
            exc = task.exception()
            if exc is not None:
                bt.status = "failed"
                bt.error = f"{type(exc).__name__}: {exc}"
                print(f"[UwUAgent] background helper [{bt.id}] crashed: {bt.error}",
                      file=sys.stderr)
            else:
                res: SubagentResult = task.result()
                bt.result = res
                bt.status = "failed" if (res.error or not res.text) else "completed"
                if res.error:
                    bt.error = res.error
        self._push_bg_tasks()
        if bt.status in ("completed", "failed"):
            self._bg_notify_queue.append(task_id)
            self._drain_bg_notifications()

    def _drain_bg_notifications(self) -> None:
        """Kick off a notification turn if reports are waiting and nothing else is
        running. Called from the done-callback, every turn's finally, and rollback.
        Safe to call anytime — re-checks the guards; double-scheduling is harmless
        (the second run finds an empty queue and returns)."""
        if not self._bg_notify_queue:
            return
        if self.orchestrator is None or self.orchestrator.session is None:
            self._bg_notify_queue.clear()
            return
        if self._touch_busy:
            return
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return
        self._schedule(self._run_bg_notification_turn())

    def _bg_row_label(self, tasks: list[BackgroundTask]) -> str:
        if len(tasks) == 1:
            bt = tasks[0]
            if bt.kind == "shell":
                return f'Background command finished: "{bt.label}"'
            return f'UwU helper reported back: "{bt.label}"'
        if all(t.kind == "shell" for t in tasks):
            return f"{len(tasks)} background commands finished"
        if all(t.kind != "shell" for t in tasks):
            return f"{len(tasks)} UwU helpers reported back"
        return f"{len(tasks)} background tasks reported back"

    def _bg_note_body(self, tasks: list[BackgroundTask]) -> str:
        """The hidden stage-direction the character answers. Carries each task's
        full report/output (that's the payload — the character relays the substance)."""
        sections: list[str] = []
        for bt in tasks:
            if bt.kind == "shell":
                if bt.status == "completed" and bt.result is not None:
                    sections.append(
                        f'Your background command ("{bt.label}") has finished. Its output:\n\n'
                        f"{bt.result.text}")
                else:
                    reason = bt.error or "it produced no output"
                    sections.append(
                        f'Your background command ("{bt.label}") FAILED to run — {reason}. '
                        f"Let the user know it didn't work out.")
                continue
            if bt.status == "completed" and bt.result is not None:
                sections.append(
                    f'Your background UwU helper \'{bt.kind}\' ("{bt.label}") has finished. '
                    f"Its report:\n\n{bt.result.text}")
            else:
                reason = bt.error or "it produced no report"
                sections.append(
                    f'Your background UwU helper \'{bt.kind}\' ("{bt.label}") FAILED — {reason}. '
                    f"Let the user know it didn't work out.")
        joined = "\n\n---\n\n".join(sections)
        return (f"{joined}\n\nReact to this: share the relevant findings with the user "
                f"naturally, in your own words (they may be mid-conversation or doing "
                f"something else, so lead in accordingly).")

    async def _run_bg_notification_turn(self) -> None:
        """Fold every queued helper report into ONE hidden reaction turn. A clone of
        the outfit-change turn mechanics: guard, snapshot (rewindable), TTS/lip-sync
        bracketing, a user_action row, submit_message(hidden=True, task_done=…),
        persist. Bails (leaving the queue intact) if something started running
        between scheduling and now — the next drain hook retries."""
        if self.orchestrator is None or self.orchestrator.session is None:
            self._bg_notify_queue.clear()
            return
        if self._touch_busy:
            return
        if self._current_turn_task is not None and not self._current_turn_task.done():
            return
        # Claim the queue only once we're committed to running.
        ids, self._bg_notify_queue = list(dict.fromkeys(self._bg_notify_queue)), []
        tasks = [self._bg_tasks[i] for i in ids
                 if i in self._bg_tasks and not self._bg_tasks[i].announced]
        if not tasks:
            return
        for bt in tasks:
            bt.announced = True
        # Announced tasks leave the status bar ("waiting to report" chips clear now).
        self._push_bg_tasks()

        self._touch_busy = True
        try:
            s = self.orchestrator.session
            row_label = self._bg_row_label(tasks)

            # NO turn snapshot: a background report isn't a user-initiated turn, and
            # rewinding it can't un-run the work — so it gets no rewind handle and
            # consumes no rollback turn_index (the history rebuild skips task_done
            # rows in its turn counting to match). Rewinding an EARLIER turn still
            # truncates these rows away like everything else after that point.

            # Same playback bracketing as a typed message so voice / text lip-sync /
            # emotion all run normally.
            tts_gen: int | None = None
            if self.voice_enabled and self.speech is not None:
                tts_gen = self._begin_tts_stream()
            elif not self.voice_enabled:
                self._send_envelope(T_LIPSYNC_TEXT_BEGIN, {})
            self._reset_streaming_state()
            self._needs_new_assistant_entry = True
            self._pending_report_ids_this_turn.clear()
            self._pending_todo_snapshot_id_this_turn = None

            # Compact event row (not rewindable — see the snapshot note above). The
            # hidden note is the very next thing appended to history (by
            # submit_message), so its index is the current length.
            self._push_entry(HistoryEntry(
                role="user_action",
                speaker=s.display_name,
                text=row_label,
                can_rollback=False,
                turn_index=-1,
                reports=[], todos=None,
                tool_name="UwUAgent",
                history_index=len(s.history),
            ))
            self._push_envelope(T_CHAT_TYPING, {"active": True})

            self._current_turn_task = asyncio.current_task()
            try:
                await self.orchestrator.submit_message(
                    self._system_message(self._bg_note_body(tasks)),
                    hidden=True, task_done=row_label)
            except asyncio.CancelledError:
                pass
            finally:
                self._current_turn_task = None
                self._push_envelope(T_CHAT_TYPING, {"active": False})
                if self.speech is not None:
                    self.speech.end_llm_stream()
                    try:
                        await self.speech.wait_until_done()
                    except Exception as e:
                        print(f"[ChatManager] bg-notify wait_until_done raised: {e}", file=sys.stderr)
                    if tts_gen is not None:
                        self._end_tts_stream(tts_gen)
                if not self.voice_enabled:
                    self._send_envelope(T_LIPSYNC_TEXT_END, {})

            self._persist()
        finally:
            self._touch_busy = False
        # More helpers may have finished while we were reacting.
        self._drain_bg_notifications()

    # ------------------------------------------------------------------------
    # Lifecycle + renderer sync
    # ------------------------------------------------------------------------

    def _cancel_all_bg_tasks(self) -> None:
        """Dismiss every helper and drop all pending announcements. Wired into
        _reset_session_bookkeeping, so create/load/restart/end all tear down here."""
        for bt in self._bg_tasks.values():
            if bt.running:
                bt.status = "dismissed"
                bt.task.cancel()
        had_any = bool(self._bg_tasks)
        self._bg_tasks.clear()
        self._bg_notify_queue.clear()
        if had_any:
            self._push_bg_tasks()

    def _prune_bg_tasks(self) -> None:
        """Cap how many finished records linger for CheckUwUHelpers."""
        finished = [tid for tid, bt in self._bg_tasks.items() if not bt.running]
        for tid in finished[:-_MAX_FINISHED_KEPT] if len(finished) > _MAX_FINISHED_KEPT else []:
            del self._bg_tasks[tid]

    def _push_bg_tasks(self) -> None:
        self._push_envelope(T_CHAT_BG_TASKS, {
            "tasks": [
                {"id": bt.id, "kind": bt.kind, "label": bt.label, "status": bt.status,
                 "startedAt": int(bt.started_wall * 1000), "announced": bt.announced}
                for bt in self._bg_tasks.values()
            ],
        })

    async def _on_dismiss_bg_task(self, task_id: str) -> None:
        """Renderer-initiated dismiss (the ✕ on a status-bar chip). Scheduled onto
        the chat loop — task.cancel() isn't safe from the JS-bridge thread."""
        ok, msg = self.bg_helper_dismiss(task_id)
        if not ok:
            print(f"[BgTask] UI dismiss ignored: {msg}", file=sys.stderr)
