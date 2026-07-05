"""Tool approval gate.

Port of Assets/Scripts/ChatTools/Workspace/ToolApprovalGate.cs. Spawns a modal
through ChatManager's `ask_modal` callable and resolves the result into a
session-scoped decision. Mirrors Claude Code's per-tool approval flow:

  * "Allow once"        — this single invocation
  * "Allow this session" — sticky for the rest of the chat session
  * "Deny"              — refused; tool returns an error

The session-sticky decisions live on the gate as a set of "approved (tool, hash)"
keys, where `hash` is whatever the tool wants to scope its approval by — typically
the resolved absolute path for file tools and the normalized command line for
shell tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


# App's RequestPermission envelope is the one the renderer already understands.
TYPE_REQUEST_PERMISSION = "RequestPermission"


@dataclass
class ApprovalRequest:
    """What we send the modal. `tool_name`, `summary`, `details` are surfaced in
    the permission window; `risk_level` colors the call-to-action."""
    tool_name: str
    summary: str
    details: dict
    risk_level: str = "elevated"  # "elevated" | "danger"


@dataclass
class ApprovalDecision:
    allow: bool
    # "Once" | "Session"  — modal returns scope as a string. Default to "Once" on dismissal.
    scope: str = "Once"


@dataclass
class ApprovalGate:
    """Session-scoped approval state + modal driver.

    `ask_modal` is the awaitable callable app.py supplies — it spawns a
    pywebview window and returns the user's reply dict ({"allow": bool, "scope":
    "Once"|"Session"}). Headless tests can substitute a stub that returns a
    fixed decision."""
    ask_modal: Callable[[dict], Awaitable[dict]] | None = None
    # Full-permission mode: auto-approve every request without showing the modal. Paired with
    # WorkspaceConfig.full_access (which drops the path sandbox). Toggled from the settings panel.
    full_access: bool = False
    # (tool_name, scope_key) → True for the rest of the session.
    _session_allow: set[tuple[str, str]] = field(default_factory=set)

    def is_pre_approved(self, tool_name: str, scope_key: str) -> bool:
        if not tool_name or not scope_key:
            return False
        return (tool_name, scope_key) in self._session_allow

    def remember(self, tool_name: str, scope_key: str) -> None:
        if not tool_name or not scope_key:
            return
        self._session_allow.add((tool_name, scope_key))

    async def request(self, req: ApprovalRequest, scope_key: str | None = None) -> ApprovalDecision:
        """Ask the user. If the (tool, scope_key) pair was already approved this
        session, return immediately without showing the modal. On dismissal /
        missing ask_modal, denies safely."""
        # Full-permission mode: allow everything, never prompt.
        if self.full_access:
            return ApprovalDecision(allow=True, scope="Session")
        if scope_key and self.is_pre_approved(req.tool_name, scope_key):
            return ApprovalDecision(allow=True, scope="Session")
        if self.ask_modal is None:
            return ApprovalDecision(allow=False, scope="Once")

        env = {
            "id": _new_id(),
            "type": TYPE_REQUEST_PERMISSION,
            "payload": {
                "toolName": req.tool_name,
                # Renderer-expected fields. `tier` drives the session-allow button label
                # ("Allow shell this session" vs "Allow workspace writes this session");
                # `detail` is the pre-formatted block shown under the title.
                "tier": _tier_from_risk(req.risk_level),
                "detail": _format_detail(req.summary, req.details),
                # Structured fields kept alongside so a future renderer can pick them
                # up without a wire-format change.
                "summary": req.summary,
                "riskLevel": req.risk_level,
                "details": req.details,
            },
        }
        try:
            reply = await self.ask_modal(env)
        except Exception:
            return ApprovalDecision(allow=False, scope="Once")
        if not isinstance(reply, dict):
            return ApprovalDecision(allow=False, scope="Once")
        allow = bool(reply.get("allow", False))
        scope = str(reply.get("scope") or "Once")
        if allow and scope == "Session" and scope_key:
            self.remember(req.tool_name, scope_key)
        return ApprovalDecision(allow=allow, scope=scope if scope in ("Once", "Session") else "Once")


def _new_id() -> str:
    import uuid
    return "m_" + uuid.uuid4().hex


def _tier_from_risk(risk: str) -> str:
    """Map ApprovalRequest.risk_level to the wire-level tier string the renderer
    keys off for its button label and styling."""
    return "DangerFullAccess" if (risk or "").lower() == "danger" else "WorkspaceWrite"


def _format_detail(summary: str, details: dict | None) -> str:
    """Lay out the structured request details into a single pre-wrap string the
    renderer can drop into its `.detail` block. Goal is `git status` showing the
    actual command instead of just "Approve tool: Bash"."""
    lines: list[str] = []
    if summary:
        lines.append(summary)
    if not isinstance(details, dict) or not details:
        return "\n".join(lines)

    # Command-shaped tools surface the command itself prominently — that's what the
    # user actually needs to see before clicking allow.
    cmd = details.get("command")
    if isinstance(cmd, str) and cmd.strip():
        if lines:
            lines.append("")
        lines.append("$ " + cmd.strip())

    # Edit's old→new preview is the only nested-dict case in use today.
    preview = details.get("preview")
    if isinstance(preview, dict):
        old = str(preview.get("old") or "")
        new = str(preview.get("new") or "")
        if old or new:
            if lines:
                lines.append("")
            if old:
                lines.append("- " + old.replace("\n", "\n  "))
            if new:
                lines.append("+ " + new.replace("\n", "\n  "))
    elif isinstance(preview, str) and preview.strip():
        if lines:
            lines.append("")
        lines.append("Preview:")
        lines.append(preview)

    # Remaining scalar fields get a flat key: value layout. Skip the ones already
    # rendered above and any falsy values that wouldn't add information.
    skip = {"command", "preview"}
    rendered_any = False
    for key, value in details.items():
        if key in skip:
            continue
        if value is None or value == "" or value is False:
            continue
        if not rendered_any:
            if lines and lines[-1] != "":
                lines.append("")
            rendered_any = True
        lines.append(f"{_humanize_key(key)}: {value}")
    return "\n".join(lines)


def _humanize_key(k: str) -> str:
    """`timeoutSeconds` → `Timeout seconds`, `cwd` → `Cwd`, etc. Light touch — the
    keys are already pretty readable."""
    if not k:
        return k
    # Insert spaces before uppercase letters that follow a lowercase one (camelCase).
    out: list[str] = [k[0].upper()]
    for prev, ch in zip(k, k[1:]):
        if ch.isupper() and prev.islower():
            out.append(" ")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)
