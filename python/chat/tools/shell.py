"""Shared shell-execution helpers used by Bash + PowerShell tools.

Ports of:
  * Assets/Scripts/ChatTools/Workspace/ShellDanger.cs   — hard-block dangerous patterns
  * Assets/Scripts/ChatTools/Workspace/ShellPermission.cs — allow / deny prefixes
  * Assets/Scripts/ChatTools/Workspace/ShellResolver.cs  — find the interpreter
  * Assets/Scripts/ChatTools/Workspace/ShellExecutor.cs  — subprocess execution
  * Assets/Scripts/ChatTools/Workspace/ShellEdition.cs   — detect PS edition

Reduced to the parts the Python tool runner actually needs. The dangerous-pattern
list is the same as the C# side.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Hard-block patterns
# ----------------------------------------------------------------------------

_BASH_DANGEROUS = [
    re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-rf|--recursive\s+--force)\s+(/|/\*|~)\s*(\s|$)"),
    re.compile(r":\(\)\s*{\s*:\|:&"),  # fork bomb
    re.compile(r"\bdd\s+.*\bof=/dev/(sd|nvme|hd)"),
    re.compile(r"\bmkfs\."),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bcurl\s+[^|]*\|\s*(sh|bash|zsh)\b", re.IGNORECASE),
    re.compile(r"\bwget\s+[^|]*\|\s*(sh|bash|zsh)\b", re.IGNORECASE),
    re.compile(r"\b/dev/sd[a-z]"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"sudo\s+rm\s+-rf"),
]

_POWERSHELL_DANGEROUS = [
    re.compile(r"Invoke-Expression", re.IGNORECASE),
    re.compile(r"\biex\b", re.IGNORECASE),
    re.compile(r"Add-Type", re.IGNORECASE),
    re.compile(r"Start-Process[^\n]*[-/–]Verb\s+RunAs", re.IGNORECASE),
    re.compile(r"Remove-Item\s+[-/–][a-zA-Z]*Force[a-zA-Z]*\s+/", re.IGNORECASE),
    re.compile(r"Format-Volume", re.IGNORECASE),
    re.compile(r"Clear-Disk", re.IGNORECASE),
    re.compile(r"Restart-Computer", re.IGNORECASE),
    re.compile(r"Stop-Computer", re.IGNORECASE),
    re.compile(r"\biwr\s+[^|]*\|\s*iex", re.IGNORECASE),
    re.compile(r"Invoke-WebRequest\s+[^|]*\|\s*iex", re.IGNORECASE),
]


def is_dangerous_bash(command: str) -> str | None:
    """Returns the dangerous-pattern description if the command is hard-blocked,
    otherwise None."""
    if not command:
        return None
    for rx in _BASH_DANGEROUS:
        if rx.search(command):
            return rx.pattern
    return None


def is_dangerous_powershell(command: str) -> str | None:
    if not command:
        return None
    for rx in _POWERSHELL_DANGEROUS:
        if rx.search(command):
            return rx.pattern
    return None


# ----------------------------------------------------------------------------
# Allow / deny prefix classification
# ----------------------------------------------------------------------------

class ShellClassification:
    ALLOWED = "allowed"
    DENIED = "denied"
    NEEDS_APPROVAL = "needs_approval"


def classify_command(command: str, allowed_prefixes: list[str], denied_prefixes: list[str]) -> str:
    """Classify a (possibly compound) command against the allow/deny prefix lists.

    The command is split on unquoted `;`, `&&`, `||` and newlines, and EVERY
    segment is classified on its own: any denied segment denies the whole thing;
    auto-allow requires every segment to be allowlisted. Without the split,
    `git status; <anything>` would ride an allowlisted `git` prefix straight past
    the approval prompt."""
    segments = split_command_segments(command)
    if not segments:
        return ShellClassification.NEEDS_APPROVAL
    results = [_classify_single(s, allowed_prefixes, denied_prefixes) for s in segments]
    if ShellClassification.DENIED in results:
        return ShellClassification.DENIED
    if all(r == ShellClassification.ALLOWED for r in results):
        return ShellClassification.ALLOWED
    return ShellClassification.NEEDS_APPROVAL


def _classify_single(command: str, allowed_prefixes: list[str], denied_prefixes: list[str]) -> str:
    """One segment: whitespace-collapsed, case-insensitive prefix match; deny wins."""
    norm = _normalize_command(command)
    for p in (denied_prefixes or []):
        if norm.startswith(_normalize_command(p)):
            return ShellClassification.DENIED
    for p in (allowed_prefixes or []):
        if norm.startswith(_normalize_command(p)):
            return ShellClassification.ALLOWED
    return ShellClassification.NEEDS_APPROVAL


def split_command_segments(command: str) -> list[str]:
    """Split a compound command line on unquoted `;`, `&&`, `||` and newlines.
    Quote-aware (single + double quotes); no deeper shell parsing — good enough
    to stop chained commands from inheriting the first segment's allowlisting.
    Pipes are NOT split: the hard-block danger patterns cover `… | iex`-style
    abuse, and splitting on `|` would make ordinary pipelines unapprovable."""
    segments: list[str] = []
    cur: list[str] = []
    quote = ""
    i, n = 0, len(command or "")
    while i < n:
        ch = command[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
            i += 1
            continue
        if command[i:i + 2] in ("&&", "||"):
            segments.append("".join(cur))
            cur = []
            i += 2
            continue
        if ch in (";", "\n"):
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    segments.append("".join(cur))
    return [s.strip() for s in segments if s.strip()]


def _normalize_command(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


# ----------------------------------------------------------------------------
# Interpreter resolution
# ----------------------------------------------------------------------------

@dataclass
class ShellInterpreter:
    path: str
    edition: str = ""  # for PowerShell: "ps7" | "ps5" | ""


def resolve_bash() -> ShellInterpreter | None:
    """Find a bash binary. On Windows we fall back to Git Bash if PATH lookup misses."""
    p = shutil.which("bash")
    if p:
        return ShellInterpreter(path=p)
    if sys.platform.startswith("win"):
        for candidate in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
            r"C:\Windows\System32\bash.exe",  # WSL bash shim
        ):
            if os.path.isfile(candidate):
                return ShellInterpreter(path=candidate)
    return None


def resolve_powershell() -> ShellInterpreter | None:
    """Prefer pwsh (PS7+) over the legacy powershell.exe (5.1)."""
    p = shutil.which("pwsh")
    if p:
        return ShellInterpreter(path=p, edition="ps7")
    if sys.platform.startswith("win"):
        p = shutil.which("powershell")
        if p:
            return ShellInterpreter(path=p, edition="ps5")
    return None


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_OUTPUT_CHARS = 30_000


@dataclass
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_seconds: int = 0  # set on the timed_out path (for the output message)
    error: str | None = None


async def run_command(argv: list[str], cwd: str | None, timeout_seconds: int) -> ShellResult:
    """Run a subprocess with the given argv list and return its captured output.

    Caps output at MAX_OUTPUT_CHARS (combined, tail kept) and kills the whole
    process TREE if it exceeds the timeout. stdin is /dev/null so a command that
    tries to read input fails fast instead of hanging until the timeout."""
    timeout = max(1, min(timeout_seconds or DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS))
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # POSIX: own process group so the timeout kill can take the whole tree.
            # (Windows uses taskkill /T instead — see _kill_process_tree.)
            start_new_session=not sys.platform.startswith("win"),
        )
    except (OSError, FileNotFoundError) as e:
        return ShellResult(exit_code=-1, stdout="", stderr="", error=str(e))

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.CancelledError:
        # The awaiting task was cancelled (user Stop mid-turn, or a dismissed
        # background command). Without this the subprocess would keep running.
        _kill_process_tree(proc)
        raise
    except asyncio.TimeoutError:
        _kill_process_tree(proc)
        try:
            stdout_b, stderr_b = await proc.communicate()
        except Exception:
            stdout_b = stderr_b = b""
        return ShellResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=_decode_stream(stdout_b),
            stderr=_decode_stream(stderr_b),
            timed_out=True,
            timeout_seconds=timeout,
        )

    return ShellResult(
        exit_code=proc.returncode or 0,
        stdout=_decode_stream(stdout_b),
        stderr=_decode_stream(stderr_b),
    )


def _kill_process_tree(proc) -> None:
    """Kill the child AND its descendants. A bare proc.kill() only takes down the
    shell process — grandchildren (npm→node, dotnet, …) would keep running and
    holding ports/files after a timeout."""
    pid = proc.pid
    if pid:
        if sys.platform.startswith("win"):
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True, timeout=10,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                )
                return
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                return
            except (OSError, ProcessLookupError):
                pass
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass


def start_detached(argv: list[str], cwd: str | None) -> int | None:
    """Spawn a detached process and return its PID. No output is captured."""
    try:
        if sys.platform.startswith("win"):
            # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS to outlive parent.
            DETACHED = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=DETACHED | CREATE_NEW_PROCESS_GROUP,
                close_fds=True,
            )
        else:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        return proc.pid
    except OSError:
        return None


# Per-stream in-memory cap for the FULL capture (spill source). Streams beyond this
# lose their head silently — a runaway command shouldn't eat unbounded RAM.
_HARD_STREAM_CAP = 8 * 1024 * 1024


def _decode_stream(b: bytes | None) -> str:
    if not b:
        return ""
    text = b.decode("utf-8", errors="replace")
    return text[-_HARD_STREAM_CAP:] if len(text) > _HARD_STREAM_CAP else text


def format_output(result: ShellResult, spill_tool: str = "PowerShell") -> str:
    """Combine stdout + stderr into the canonical LLM-facing output.

    Oversized output keeps the TAIL (for builds/tests the errors and the summary
    come last) and the FULL text is spilled to a file whose path is included, so
    the model can Read the earlier part instead of losing it."""
    if result.error and not result.stdout and not result.stderr:
        return f"Error: {result.error}"
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout.rstrip("\n"))
    if result.stderr:
        # Mark stderr inline so the LLM can tell streams apart.
        parts.append("[stderr]\n" + result.stderr.rstrip("\n"))
    if result.timed_out:
        parts.append(f"[command timed out after {result.timeout_seconds}s and was killed]")
    if not parts:
        if result.exit_code == 0:
            return "(no output)"
        return f"(exit {result.exit_code}, no output)"
    # The timeout note already explains the abnormal exit — no [exit N] suffix then.
    suffix = "" if (result.exit_code == 0 or result.timed_out) else f"\n[exit {result.exit_code}]"
    combined = "\n".join(parts) + suffix
    if len(combined) <= MAX_OUTPUT_CHARS:
        return combined

    from .spill import spill_text
    path = spill_text(combined, spill_tool)
    dropped = combined[:-MAX_OUTPUT_CHARS]
    note = f"[{dropped.count(chr(10))} lines truncated from start"
    if path:
        note += f" — full output saved to: {path} (Read it if you need the earlier part)"
    note += "]\n"
    return note + combined[-MAX_OUTPUT_CHARS:]
