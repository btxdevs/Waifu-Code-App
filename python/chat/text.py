"""Pure text-processing helpers for the streaming LLM pipeline. Ports two C# files:

  EmotionStreamFilter — watches LLM output for inline [LABEL] tags, strips them from the
                        cleaned stream, fires a callback per tag. Buffer-safe across chunk
                        boundaries so a tag split mid-token doesn't leak.
  SentenceSplitter    — turns the cleaned stream into sentences for TTS. Handles
                        common abbreviations, ellipses, and chops over-long sentences on
                        commas/semicolons/colons.

No external state, no IO — these are equally usable from sync or async code.
"""
from __future__ import annotations

import re
import sys
from typing import Callable, Iterable


# ============================================================================
# EmotionStreamFilter
# ============================================================================

_OPEN_BRACKET = "["
_CLOSE_BRACKET = "]"

# Hard cap on tag body length (chars after the opening "["). If exceeded with no closing "]"
# we treat the buffered "[…" as plain text and resume. When the character's emotion labels are
# known (a whitelist is passed) the cap is derived from them — see __init__ — so a stray "[" in
# prose releases sooner; this is the fallback used when no whitelist is given.
_DEFAULT_MAX_TAG_BODY_LEN = 64
# Headroom added over the longest allowed label, to still catch loose/verbose tags the model
# might write (e.g. "[a bit of joy]" → "Joy") before declaring the "[" plain text.
_TAG_BODY_MARGIN = 16

# Matches one inline [LABEL] tag where LABEL has no brackets inside. Used by the history
# rewriter, not the streaming filter (which is char-by-char).
_TAG_REGEX = re.compile(r"\[([^\[\]]*)\]")


# ============================================================================
# Control markers
# ============================================================================
# Bracketed tokens that are NOT emotions but carry a side-effect signal the app acts on
# (e.g. [Reject] refuses an in-progress caress — see manager._turn). Like emotion tags they are
# ALWAYS hidden from the chat bubble and never sent to TTS. UNLIKE emotion tags they are kept
# VERBATIM in the stored history — but only when the turn's context makes them legitimate (a
# [Reject] is only valid on a touch-triggered turn). Emitted out of context they're stripped from
# history like a misused tag, so the save file only ever shows markers that actually fired.
# Extend this map to add new markers; keyed by the lowercased marker name.
#   canon   — the casing kept in history (the model is told to emit this form).
#   context — the turn-context key that must be active for the marker to be valid.
CONTROL_MARKERS: dict[str, dict[str, str]] = {
    "reject": {"canon": "Reject", "context": "touch"},
}


def is_control_marker(label: str) -> bool:
    """True if `label` (a tag body, no brackets) is a recognized control marker."""
    return label.strip().casefold() in CONTROL_MARKERS


def filter_control_markers(text: str, active_contexts: set[str] | None) -> str:
    """Resolve control-marker tags against the turn's active contexts: a marker whose context is
    active is kept (normalized to its canonical casing); one emitted out of context is removed.
    Emotion and other tags are left untouched. Runs on the message stored to history."""
    if not text:
        return text
    active = active_contexts or set()

    def _sub(m: re.Match) -> str:
        marker = CONTROL_MARKERS.get(m.group(1).strip().casefold())
        if marker is None:
            return m.group(0)  # not a control marker — leave emotion/other tags as-is
        if marker["context"] in active:
            return f"[{marker['canon']}]"  # valid for this turn → keep, normalized casing
        return ""  # emitted out of context → strip like a misused tag

    return _TAG_REGEX.sub(_sub, text)


class EmotionStreamFilter:
    """Stateful filter that you feed raw token chunks. `feed` returns the cleaned text
    safe to emit; `flush` returns whatever buffered bytes turned out NOT to be a tag.

    Callback signature: `on_emotion(label, position)` where `position` is the cumulative
    index into the cleaned output stream where the tag fired. Lets text-mode lip sync
    register a trigger at the exact char the tag was stripped from.
    """

    def __init__(
        self,
        allowed_labels: Iterable[str] | None = None,
        on_emotion: Callable[[str, int], None] | None = None,
    ):
        # None whitelist = accept any non-empty label. Otherwise case-insensitive.
        self._whitelist: set[str] | None
        if allowed_labels is None:
            self._whitelist = None
        else:
            self._whitelist = {s.strip() for s in allowed_labels if s and s.strip()}
            if not self._whitelist:
                self._whitelist = None
        # Cap the buffered tag-body length from the character's actual labels (longest label +
        # headroom for loose matches); fall back to the default when no labels are known.
        if self._whitelist:
            self._max_tag_body_len = max(len(w) for w in self._whitelist) + _TAG_BODY_MARGIN
        else:
            self._max_tag_body_len = _DEFAULT_MAX_TAG_BODY_LEN
        self._on_emotion = on_emotion
        self._pending: list[str] = []  # chars buffered as a possible-tag
        self._total_emitted = 0
        # When a tag is stripped, whitespace on both sides would otherwise survive as a double
        # space ("end. [Tag] Next" → "end.  Next"). After a strip we swallow leading whitespace
        # if the last emitted char was already whitespace, collapsing the seam to one space.
        self._collapse_ws = False
        self._last_char = ""

    def reset(self) -> None:
        self._pending.clear()
        self._total_emitted = 0
        self._collapse_ws = False
        self._last_char = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        out: list[str] = []
        for ch in chunk:
            self._process_char(ch, out)
        result = "".join(out)
        self._total_emitted += len(result)
        return result

    def flush(self) -> str:
        if not self._pending:
            return ""
        leftover = "".join(self._pending)
        self._pending.clear()
        self._total_emitted += len(leftover)
        return leftover

    def _emit_text(self, s: str, out: list[str]) -> None:
        """Append text to `out`, collapsing whitespace adjacent to a just-removed tag: while
        `_collapse_ws` is set, leading whitespace is dropped as long as the previous emitted char
        was whitespace (or nothing has been emitted), so a stripped tag never leaves a double
        space. Tracks the last emitted char across feed() calls."""
        for ch in s:
            if self._collapse_ws:
                if ch.isspace() and (self._last_char == "" or self._last_char.isspace()):
                    continue  # swallow — would be a double space / leading space at the seam
                self._collapse_ws = False
            out.append(ch)
            self._last_char = ch

    def _process_char(self, c: str, out: list[str]) -> None:
        if not self._pending:
            if c == _OPEN_BRACKET:
                self._pending.append(c)
            else:
                self._emit_text(c, out)
            return

        if c == _CLOSE_BRACKET:
            # _pending currently holds "[body" — body starts at index 1.
            label = "".join(self._pending[1:]).strip()
            position = self._total_emitted + len(out)
            self._try_fire_emotion(label, position)
            self._pending.clear()
            # Tag stripped → collapse any whitespace straddling the seam on the next emit.
            self._collapse_ws = True
            return

        if c == _OPEN_BRACKET:
            # New "[" inside an unfinished tag — the previous "[" wasn't a tag start.
            # Release the buffered chars except the new "[", keep scanning from there.
            self._emit_text("".join(self._pending), out)
            self._pending.clear()
            self._pending.append(_OPEN_BRACKET)
            return

        self._pending.append(c)

        # Body length = len(_pending) - 1 (subtracting the leading "[").
        if len(self._pending) - 1 > self._max_tag_body_len:
            self._emit_text("".join(self._pending), out)
            self._pending.clear()

    def _try_fire_emotion(self, label: str, position: int) -> bool:
        if not label:
            return False
        if label.casefold() in CONTROL_MARKERS:
            # A control marker (e.g. [Reject]) — swallow it (stripped from the TTS/display stream
            # like any tag) but never treat it as an emotion or log it as unknown.
            return True
        if self._whitelist is None:
            canonical = label
        else:
            canonical, exact = _resolve_against_whitelist(label, self._whitelist)
            if canonical is None:
                print(f"[EmotionStreamFilter] Unknown emotion label '{label}' — tag dropped.", file=sys.stderr)
                return False
            if not exact:
                print(f"[EmotionStreamFilter] Loose-matched '{label}' → '{canonical}'.", file=sys.stderr)

        if self._on_emotion is None:
            return True
        try:
            self._on_emotion(canonical, position)
        except Exception as e:
            print(f"[EmotionStreamFilter] on_emotion handler raised: {e}", file=sys.stderr)
        return True


def _resolve_against_whitelist(label: str, whitelist: set[str]) -> tuple[str | None, bool]:
    """Tries exact match first, then loose substring matching both directions:
      "guilt"        → "Guilt/Shame"          (whitelist entry contains the input)
      "extreme joy"  → "Joy"                  (input contains a whitelist entry)
    Returns (canonical, exact) or (None, False) when nothing matches.
    """
    lowered = label.casefold()
    # Pass 1: exact (case-insensitive)
    for w in whitelist:
        if w.casefold() == lowered:
            return w, True
    # Pass 2: whitelist entry contains the input
    for w in whitelist:
        if lowered in w.casefold():
            return w, False
    # Pass 3: input contains a whitelist entry
    for w in whitelist:
        if w.casefold() in lowered:
            return w, False
    return None, False


def rewrite_tags_for_history(
    text: str,
    allowed_labels: Iterable[str] | None,
    on_correction: Callable[[str, str], None] | None = None,
    on_removed: Callable[[str], None] | None = None,
) -> str:
    """Rewrites every [LABEL] tag in `text` to its canonical form against `allowed_labels`:
      * Exact match → kept as-is.
      * Loose match → replaced with [CANONICAL]; `on_correction(raw, canonical)` fires.
      * No match    → tag removed; `on_removed(raw)` fires.

    Used to rewrite assistant messages BEFORE they're stored in the LLM history so the
    model sees only canonical labels and self-corrects its vocabulary.
    """
    if not text:
        return text
    if allowed_labels is None:
        return text
    whitelist = {s.strip() for s in allowed_labels if s and s.strip()}
    if not whitelist:
        return text

    def _sub(m: re.Match) -> str:
        raw = m.group(1).strip()
        if not raw:
            if on_removed:
                on_removed(raw)
            return ""
        if raw.casefold() in CONTROL_MARKERS:
            return m.group(0)  # control marker, not an emotion — filter_control_markers judges it
        canonical, exact = _resolve_against_whitelist(raw, whitelist)
        if canonical is None:
            if on_removed:
                on_removed(raw)
            return ""
        if exact:
            return m.group(0)
        if on_correction:
            on_correction(raw, canonical)
        return f"[{canonical}]"

    return _TAG_REGEX.sub(_sub, text)


# ============================================================================
# SpokenTextSanitizer
# ============================================================================
# The system prompt bans markdown and *stage directions* in spoken replies, but no prompt
# wording gets 100% compliance — this filter enforces the format deterministically on the
# way out. It runs AFTER EmotionStreamFilter (so it never sees [LABEL] tags mid-parse;
# brackets pass through untouched) in front of every consumer of clean text: the renderer
# bubble, text lip-sync, and the TTS sentence splitter. `sanitize_spoken_text` applies the
# same rules batch-wise to the message stored in LLM history, so the model only ever sees
# compliant copies of its own replies and self-corrects — same trick as
# rewrite_tags_for_history.
#
# Rules:
#   *stage direction*      → removed entirely. Roleplay models use single-star spans for
#                            actions, not emphasis; spans up to _STAR_SPAN_CAP chars are
#                            treated as actions and dropped. An unclosed span releases as
#                            plain text (the '*' itself is still dropped).
#   **bold** / __bold__    → markers stripped, text kept.
#   `code`                 → backticks stripped, text kept.
#   ``` fence lines        → the marker line is dropped whole (fenced CONTENT lines still
#                            stream through as plain text).
#   line-start "# ", "> ", "- ", "+ ", "* ", "1. ", "1) " → marker stripped.
#   lone '*'               → dropped ('*' and '`' never reach the output; a math "2 * 3"
#                            degrades to "2 3", acceptable for speech).
# Whitespace next to a removed span is collapsed so no double spaces are left behind.

_STAR_SPAN_CAP = 64    # max chars a *span* may buffer before we decide it isn't a stage direction
_PAIR_SPAN_CAP = 128   # same for **bold** / __bold__
_TICK_SPAN_CAP = 128   # same for `code`


class SpokenTextSanitizer:
    """Stateful streaming markdown/stage-direction stripper. Feed raw chunks (already
    emotion-filtered), get cleaned text back; `flush` releases whatever buffered chars
    turned out not to be markup. Buffer-safe across chunk boundaries, like
    EmotionStreamFilter."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._mode = "normal"       # normal | star_open | star | pair | tick_open | tick | fence | linehead
        self._pend: list[str] = []  # buffered chars of the construct being decided
        self._marker = ""           # pair mode: '*' or '_'
        self._marker_seen = False   # pair mode: previous char was the marker (half of a close)
        self._ticks = 0             # tick_open mode: consecutive backticks seen
        self._at_line_start = True
        self._collapse_ws = False
        self._last_char = ""

    # ---- output helpers ----------------------------------------------------

    def _emit(self, s: str, out: list[str]) -> None:
        """Emit text, collapsing whitespace adjacent to a just-removed construct (same
        seam handling as EmotionStreamFilter._emit_text)."""
        for ch in s:
            if self._collapse_ws:
                if ch.isspace() and (self._last_char == "" or self._last_char.isspace()):
                    continue
                self._collapse_ws = False
            out.append(ch)
            self._last_char = ch
            self._at_line_start = ch == "\n"

    def _release(self, out: list[str], drop_pend: bool = False) -> None:
        """Leave the current construct mode, emitting the buffered chars as plain text
        (unless `drop_pend`)."""
        if self._pend and not drop_pend:
            pend = "".join(self._pend)
            self._pend = []
            self._emit(pend, out)
        else:
            self._pend = []
        self._mode = "normal"
        self._marker_seen = False

    # ---- public API ----------------------------------------------------------

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        out: list[str] = []
        for ch in chunk:
            self._step(ch, out)
        return "".join(out)

    def flush(self) -> str:
        out: list[str] = []
        # Unclosed spans release their inner text; bare markers (star_open / tick_open /
        # fence) just vanish.
        self._release(out, drop_pend=self._mode in ("star_open", "tick_open", "fence"))
        return "".join(out)

    # ---- state machine -------------------------------------------------------

    def _step(self, c: str, out: list[str]) -> None:
        mode = self._mode
        if mode == "normal":
            self._step_normal(c, out)
        elif mode == "star_open":
            self._step_star_open(c, out)
        elif mode == "star":
            self._step_star(c, out)
        elif mode == "pair":
            self._step_pair(c, out)
        elif mode == "tick_open":
            self._step_tick_open(c, out)
        elif mode == "tick":
            self._step_tick(c, out)
        elif mode == "fence":
            self._step_fence(c)
        elif mode == "linehead":
            self._step_linehead(c, out)

    def _step_normal(self, c: str, out: list[str]) -> None:
        if c == "*":
            self._mode = "star_open"
            return
        if c == "`":
            self._mode = "tick_open"
            self._ticks = 1
            return
        if c == "_":
            # Only '__' pairs are markup; a single '_' is prose (snake_case…). Reuse the
            # pair machinery via a tiny lookahead: buffer the '_' in linehead-style pend.
            self._mode = "linehead"
            self._pend = ["_"]
            return
        if self._at_line_start and (c in "#>-+" or c.isdigit()):
            self._mode = "linehead"
            self._pend = [c]
            return
        self._emit(c, out)

    def _step_star_open(self, c: str, out: list[str]) -> None:
        if c == "*":
            self._mode = "pair"
            self._marker = "*"
            self._marker_seen = False
            self._pend = []
            return
        if c == " " and self._at_line_start:
            # "* " at line start = bullet marker → drop marker and space.
            self._mode = "normal"
            self._at_line_start = False
            return
        if c.isspace():
            # Lone '*' ("2 * 3") — drop the star, keep the whitespace.
            self._mode = "normal"
            self._collapse_ws = True
            self._emit(c, out)
            return
        self._mode = "star"
        self._pend = [c]

    def _step_star(self, c: str, out: list[str]) -> None:
        if c == "*":
            # Closed single-star span → stage direction: drop it whole.
            self._pend = []
            self._mode = "normal"
            self._collapse_ws = True
            return
        if c == "\n" or len(self._pend) >= _STAR_SPAN_CAP:
            self._release(out)      # not a stage direction after all — plain text
            self._step(c, out)
            return
        self._pend.append(c)

    def _step_pair(self, c: str, out: list[str]) -> None:
        if c == self._marker:
            if self._marker_seen:   # full "**" / "__" close → emphasis: keep inner text
                self._release(out)
                return
            self._marker_seen = True
            return
        self._marker_seen = False   # lone marker inside the span — drop it
        if c == "\n" or len(self._pend) >= _PAIR_SPAN_CAP:
            self._release(out)
            self._step(c, out)
            return
        self._pend.append(c)

    def _step_tick_open(self, c: str, out: list[str]) -> None:
        if c == "`":
            self._ticks += 1
            if self._ticks >= 3:
                # "```" — a fence marker line: drop it whole (only ever line-start in
                # practice; a mid-line "```" degrades to the same handling).
                self._mode = "fence"
            return
        # 1 backtick → inline code span; 2 backticks ("``x``" style) → same, the second
        # pair of ticks at close just re-enters tick_open and vanishes.
        self._mode = "tick"
        self._pend = []
        self._step(c, out)

    def _step_tick(self, c: str, out: list[str]) -> None:
        if c == "`":
            self._release(out)      # close → keep inner text, ticks vanish
            return
        if c == "\n" or len(self._pend) >= _TICK_SPAN_CAP:
            self._release(out)
            self._step(c, out)
            return
        self._pend.append(c)

    def _step_fence(self, c: str) -> None:
        # Drop everything through end-of-line, newline included (the marker line
        # vanishes; the preceding char was already a line break).
        if c == "\n":
            self._mode = "normal"
            self._at_line_start = True

    def _step_linehead(self, c: str, out: list[str]) -> None:
        """Deciding whether the line's first chars are a markdown marker. `_pend` holds
        the candidate ('#'+, '>', '-', '+', digits[./)], or a lone '_' from anywhere)."""
        first = self._pend[0]
        if first == "_":
            if c == "_":            # "__" → emphasis pair
                self._mode = "pair"
                self._marker = "_"
                self._marker_seen = False
                self._pend = []
                return
            self._release(out)      # single '_' is prose
            self._step(c, out)
            return
        if first == "#":
            if c == "#" and len(self._pend) < 6:
                self._pend.append(c)
                return
            if c == " ":            # "# " header marker → drop it
                self._release(out, drop_pend=True)
                self._at_line_start = False
                return
            self._release(out)      # "#hashtag" etc. — plain text
            self._step(c, out)
            return
        if first in ">-+":
            if c == " " and len(self._pend) == 1:   # "> " / "- " / "+ " → drop marker
                self._release(out, drop_pend=True)
                self._at_line_start = False
                return
            self._release(out)
            self._step(c, out)
            return
        # Digits: numbered-list marker is digits + '.' or ')' + space.
        if c.isdigit() and self._pend[-1].isdigit() and len(self._pend) < 4:
            self._pend.append(c)
            return
        if c in ".)" and self._pend[-1].isdigit():
            self._pend.append(c)
            return
        if c == " " and self._pend[-1] in ".)":     # "1. " / "1) " → drop marker
            self._release(out, drop_pend=True)
            self._at_line_start = False
            return
        self._release(out)          # "3.14…", "42 things", … — plain text
        self._step(c, out)


def sanitize_spoken_text(text: str) -> str:
    """Batch form of SpokenTextSanitizer: strip markdown and *stage directions* from a
    complete piece of text. [LABEL] tags and control markers pass through untouched, so
    this is safe to run on history messages that still carry their emotion tags."""
    if not text:
        return text
    s = SpokenTextSanitizer()
    return s.feed(text) + s.flush()


# ============================================================================
# SentenceSplitter
# ============================================================================

MAX_CHARS_PER_CHUNK = 220

# Lowercase set so the check is `word.lower() in _ABBREVIATIONS`.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "no", "fig", "approx", "inc", "ltd", "co",
}


def try_consume(buffer: list[str]) -> str | None:
    """If `buffer` (a list of chars used as a mutable string) contains a completed
    sentence, returns it and removes the consumed chars (plus trailing whitespace).
    Returns None when no boundary is present yet.

    A list is used in place of StringBuilder so we can do O(1) leading-prefix slicing
    via `del buffer[:n]`. Tokens are appended with `buffer.extend(token)`.
    """
    n = len(buffer)
    if n == 0:
        return None

    i = 0
    while i < n:
        c = buffer[i]

        # Line breaks are hard sentence boundaries on their own — useful when the LLM
        # separates points with newlines instead of punctuation.
        if c == "\n" or c == "\r":
            line_sentence_end = i
            consume_to = i
            while consume_to < n and buffer[consume_to].isspace():
                consume_to += 1
            sentence = "".join(buffer[:line_sentence_end]).lstrip()
            del buffer[:consume_to]
            if not sentence:
                # Buffer started with a line break; recurse on what's left.
                return try_consume(buffer)
            return _split_if_too_long(sentence)

        if c not in (".", "!", "?"):
            i += 1
            continue

        # Collapse runs of terminal punctuation ("?!", "...") into one endpoint.
        punct_end = i
        while punct_end + 1 < n and _is_terminal_punct(buffer[punct_end + 1]):
            punct_end += 1

        # '!' / '?' are unambiguous prose terminators — commit even at end-of-buffer.
        has_strong_terminator = any(buffer[j] in ("!", "?") for j in range(i, punct_end + 1))

        at_buffer_end = punct_end + 1 >= n
        if at_buffer_end:
            if not has_strong_terminator:
                # Multi-dot run (ellipsis "..."): may still grow. Wait.
                if punct_end != i:
                    return None
                # Single '.': only commit if context strongly suggests a sentence end.
                if _is_abbreviation_terminated_at(buffer, i):
                    return None
                # Decimal / dotted number ("$5.", "v1."): waiting for ".99" / ".0".
                if i > 0 and buffer[i - 1].isdigit():
                    return None
                # No letter before the dot: probably opening punctuation (".5") or stray dot.
                if i == 0 or not buffer[i - 1].isalpha():
                    return None
        else:
            if not buffer[punct_end + 1].isspace():
                i = punct_end + 1  # skip past this punctuation run
                continue
            # Skip abbreviation boundaries ("Dr. "). Only '.' runs; "!" / "?" never abbreviate.
            if not has_strong_terminator and _is_abbreviation_terminated_at(buffer, i):
                i = punct_end + 1
                continue

        sentence_end = punct_end + 1  # exclusive index past punctuation
        consume_to = sentence_end
        while consume_to < n and buffer[consume_to].isspace():
            consume_to += 1
        sentence = "".join(buffer[:sentence_end]).lstrip()
        del buffer[:consume_to]
        return _split_if_too_long(sentence)

    return None


def flush_remaining(buffer: list[str]) -> str | None:
    """Returns whatever is left in the buffer as a final sentence (even without
    terminating punctuation). Clears the buffer. Returns None if only whitespace."""
    if not buffer:
        return None
    remaining = "".join(buffer).strip()
    buffer.clear()
    return _split_if_too_long(remaining) if remaining else None


def split_long_sentence(sentence: str) -> list[str]:
    """Splits a long sentence on commas/semicolons/colons (followed by whitespace) and
    greedily packs the pieces into chunks at most MAX_CHARS_PER_CHUNK long."""
    if not sentence:
        return []
    if len(sentence) <= MAX_CHARS_PER_CHUNK:
        return [sentence]

    # Find clause delimiters: comma/semicolon/colon followed by whitespace. Keep the
    # delimiter with the preceding piece.
    parts: list[str] = []
    last = 0
    n = len(sentence)
    for i in range(n - 1):
        c = sentence[i]
        if c in (",", ";", ":") and sentence[i + 1].isspace():
            parts.append(sentence[last:i + 1])
            j = i + 1
            while j < n and sentence[j].isspace():
                j += 1
            last = j
    if last < n:
        parts.append(sentence[last:])

    # Greedy packing.
    result: list[str] = []
    buf = ""
    for piece in parts:
        if not buf:
            buf = piece
        elif len(buf) + 1 + len(piece) <= MAX_CHARS_PER_CHUNK:
            buf = buf + " " + piece
        else:
            result.append(buf)
            buf = piece
    if buf:
        result.append(buf)
    return result


def _split_if_too_long(sentence: str) -> str:
    """Returns just the first chunk if the sentence is over MAX_CHARS_PER_CHUNK; otherwise
    the sentence unchanged. Callers that want every piece should use `split_long_sentence`
    directly."""
    if not sentence or len(sentence) <= MAX_CHARS_PER_CHUNK:
        return sentence
    pieces = split_long_sentence(sentence)
    return pieces[0] if pieces else sentence


def _is_terminal_punct(c: str) -> bool:
    return c in (".", "!", "?")


def _is_abbreviation_terminated_at(buffer: list[str], dot_idx: int) -> bool:
    if dot_idx < 0 or dot_idx >= len(buffer) or buffer[dot_idx] != ".":
        return False
    # Walk back to the start of the word — letters only, but allow internal dots so "e.g"
    # and "i.e" match as whole tokens.
    start = dot_idx
    while start > 0:
        prev = buffer[start - 1]
        if prev.isalpha() or prev == ".":
            start -= 1
        else:
            break
    if start == dot_idx:
        return False
    word = "".join(buffer[start:dot_idx]).lower()
    return word in _ABBREVIATIONS
