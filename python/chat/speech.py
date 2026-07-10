"""Sentence-level streaming TTS pipeline. Port of
Assets/Scripts/Chat/Backend/SentenceSpeechPipeline.cs — but much smaller, because all of
the AudioSource / StreamingAudioBuffer / first-audible-sample tracking stays in Unity.
Python's job is:

  1. Feed raw LLM tokens through `EmotionStreamFilter` (strips [LABEL] tags), then
     `SpokenTextSanitizer` (strips markdown / *stage directions* the prompt bans but the
     model emits anyway — otherwise TTS reads the markers aloud).
  2. Sentence-split the cleaned text.
  3. Push each sentence into a TTS queue.
  4. The pump pulls sentences in order, fires the per-sentence emotion (Avatar.ApplyEmotion),
     then awaits `tts_synthesize(text)` which streams audio chunks to Unity.
  5. Track speaking state via `notify_playback_started` / `notify_playback_ended` so the
     orchestrator and renderer know when speech ends.

The emotion-fires-slightly-ahead-of-audio behavior is approximated by firing the label
when its sentence enters synthesis — the TTS-to-Unity network round-trip + Unity's
buffer warmup gives the face a natural beat before the words land. The C# version used
`audioSource.time` to schedule precisely; Python has no equivalent without new
envelopes, and the approximation is close enough for the migration.
"""
from __future__ import annotations

import asyncio
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .text import EmotionStreamFilter, SpokenTextSanitizer, flush_remaining, try_consume


# Callback aliases — keep signatures explicit so readers don't have to chase types.
TtsSynthesizeFn = Callable[[str], Awaitable[None]]  # speak this text; returns when audio finishes streaming to Unity
EmotionFireFn = Callable[[str], None]               # apply this emotion label to the avatar (Avatar.ApplyEmotion)
SpeakingEdgeFn = Callable[[bool], None]              # (True at first audible, False at drain)


@dataclass
class _PendingSentence:
    text: str
    pre_emotion: str | None  # None if no emotion change at this boundary


class SentenceSpeechPipeline:
    """One instance per chat session. Lifecycle: `begin_session` → feed tokens via
    `feed_token` (multiple) → `end_llm_stream` → optional `wait_until_done` →
    `stop_session` (called by ChatManager on session reset or new turn)."""

    def __init__(
        self,
        tts_synthesize: TtsSynthesizeFn,
        on_apply_emotion: EmotionFireFn,
        on_speaking_edge: SpeakingEdgeFn | None = None,
        allowed_emotions: list[str] | None = None,
        verbose: bool = False,
    ):
        self._tts_synthesize = tts_synthesize
        self._on_apply_emotion = on_apply_emotion
        self._on_speaking_edge = on_speaking_edge
        self._allowed_emotions = list(allowed_emotions) if allowed_emotions else None
        self._verbose = verbose

        # Per-session state — reset on every begin_session.
        self._emotion_filter: EmotionStreamFilter | None = None
        self._sanitizer: SpokenTextSanitizer | None = None
        self._text_buffer: list[str] = []          # cleaned tokens awaiting sentence boundary
        self._queue: deque[_PendingSentence] = deque()
        self._next_sentence_emotion: str | None = None  # latched by EmotionStreamFilter callback
        self._llm_stream_ended = False
        self._session_active = False
        self._pump_task: asyncio.Task | None = None
        # Notifies the pump that there's new work to look at (sentences enqueued or
        # llm-stream-ended). Avoids a busy-wait while idle.
        self._wake = asyncio.Event()
        # Set once the pump's lifecycle monitor confirms everything has drained.
        self._all_done = asyncio.Event()
        # Tracks whether we've fired the True edge; the False edge is paired off this.
        self._speaking_edge_fired = False

    # ------------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------------

    def begin_session(self) -> None:
        """Start a fresh streaming session. Safe to call mid-session — it tears down
        the previous pump first."""
        if self._session_active:
            self.stop_session()

        self._emotion_filter = EmotionStreamFilter(
            allowed_labels=self._allowed_emotions,
            on_emotion=self._on_emotion_in_stream,
        )
        self._sanitizer = SpokenTextSanitizer()
        self._text_buffer.clear()
        self._queue.clear()
        self._next_sentence_emotion = None
        self._llm_stream_ended = False
        self._session_active = True
        self._wake = asyncio.Event()
        self._all_done = asyncio.Event()
        self._speaking_edge_fired = False

        self._pump_task = asyncio.create_task(self._pump(), name="speech-pump")
        if self._verbose:
            print("[SentenceSpeechPipeline] session started", file=sys.stderr)

    def feed_token(self, raw_token: str) -> None:
        """Push one token (or partial token) of raw LLM output through the emotion
        filter, append the cleaned text to the buffer, and drain any complete
        sentences into the TTS queue."""
        if not self._session_active or not raw_token:
            return
        cleaned = self._emotion_filter.feed(raw_token) if self._emotion_filter else raw_token
        if cleaned and self._sanitizer is not None:
            cleaned = self._sanitizer.feed(cleaned)
        if not cleaned:
            return
        self._text_buffer.extend(cleaned)
        self._drain_sentences()

    def flush_text_buffer(self) -> None:
        """Force any partial sentence currently buffered to be enqueued. Called by the
        orchestrator right before a tool runs (so the TTS doesn't stall on a
        half-spoken sentence while a tool blocks) and on end-of-stream."""
        if not self._session_active:
            return
        # Emotion filter may have a trailing "[" buffered — release it through the
        # sanitizer, then release the sanitizer's own buffered construct (if any).
        tail = self._emotion_filter.flush() if self._emotion_filter is not None else ""
        if self._sanitizer is not None:
            tail = self._sanitizer.feed(tail) + self._sanitizer.flush()
        if tail:
            self._text_buffer.extend(tail)
        self._drain_sentences()
        leftover = flush_remaining(self._text_buffer)
        if leftover:
            self._enqueue_sentence(leftover)

    def end_llm_stream(self) -> None:
        """Tell the pipeline no more tokens are coming. The pump will exit after the
        current queue drains."""
        if not self._session_active:
            return
        self.flush_text_buffer()
        self._llm_stream_ended = True
        self._wake.set()

    async def wait_until_done(self) -> None:
        """Yields until everything currently enqueued has been TTS'd. Used by the
        orchestrator to time tool side-effects (outfit swap should land just after the
        narration). Unlike the C# version this doesn't wait for the AudioSource clock to
        catch up — that distinction needs Unity-side help (a new envelope) we haven't
        wired up yet, so for now it's "sent to Unity" rather than "heard"."""
        if not self._session_active:
            return
        # Pump exits once queue is empty AND llm_stream_ended is true. Waiting on
        # _all_done would block until end_llm_stream is called; for the mid-stream
        # tool-flush case we want to wait for a quieter signal: queue empty + no
        # active synthesis.
        while self._session_active and (self._queue or self._tts_in_flight):
            await asyncio.sleep(0.02)

    def stop_session(self) -> None:
        """Tear down — cancel the pump, drop the queue. Fires the False speaking edge
        if it had previously fired True so subscribers don't end up stuck in the
        speaking state."""
        was_speaking = self._speaking_edge_fired
        self._session_active = False
        self._llm_stream_ended = True

        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
        self._pump_task = None
        self._queue.clear()
        self._text_buffer.clear()
        self._all_done.set()

        if was_speaking and self._on_speaking_edge is not None:
            try:
                self._on_speaking_edge(False)
            except Exception as e:
                print(f"[SentenceSpeechPipeline] speaking_edge False raised: {e}", file=sys.stderr)
        self._speaking_edge_fired = False

    # ------------------------------------------------------------------------
    # External edge notifications from Unity (Tts.PlaybackStarted / Tts.PlaybackEnded)
    # ------------------------------------------------------------------------

    def notify_playback_started(self) -> None:
        """Unity tells us its AudioSource just emitted the first audible sample. Forward
        as the True speaking-edge if we haven't already."""
        if self._speaking_edge_fired:
            return
        self._speaking_edge_fired = True
        if self._on_speaking_edge is not None:
            try:
                self._on_speaking_edge(True)
            except Exception as e:
                print(f"[SentenceSpeechPipeline] speaking_edge True raised: {e}", file=sys.stderr)

    def notify_playback_ended(self) -> None:
        """Unity tells us the StreamingAudioBuffer has drained. Fire the False edge."""
        if not self._speaking_edge_fired:
            return
        self._speaking_edge_fired = False
        if self._on_speaking_edge is not None:
            try:
                self._on_speaking_edge(False)
            except Exception as e:
                print(f"[SentenceSpeechPipeline] speaking_edge False raised: {e}", file=sys.stderr)

    # ------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------

    _tts_in_flight: int = 0

    def _on_emotion_in_stream(self, label: str, _position: int) -> None:
        """Latched by the next enqueued sentence — its `pre_emotion` becomes this label
        so the pump can fire Avatar.ApplyEmotion right before that sentence's TTS."""
        self._next_sentence_emotion = label

    def _drain_sentences(self) -> None:
        while True:
            sentence = try_consume(self._text_buffer)
            if sentence is None:
                return
            self._enqueue_sentence(sentence)

    def _enqueue_sentence(self, sentence: str) -> None:
        if not sentence.strip():
            return
        emo = self._next_sentence_emotion
        self._next_sentence_emotion = None
        self._queue.append(_PendingSentence(text=sentence, pre_emotion=emo))
        if self._verbose:
            print(f"[SentenceSpeechPipeline] queued sentence (pre_emotion={emo}): {sentence!r}", file=sys.stderr)
        self._wake.set()

    async def _pump(self) -> None:
        """Pulls sentences in order, fires the per-sentence emotion, then awaits TTS
        synthesis. Exits once the queue is empty AND `end_llm_stream` has been called."""
        try:
            while self._session_active:
                if not self._queue:
                    if self._llm_stream_ended:
                        break
                    # Wait for either new sentences or stream-end.
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass
                    continue
                item = self._queue.popleft()
                if item.pre_emotion and self._on_apply_emotion is not None:
                    try:
                        self._on_apply_emotion(item.pre_emotion)
                    except Exception as e:
                        print(f"[SentenceSpeechPipeline] on_apply_emotion raised: {e}", file=sys.stderr)
                self._tts_in_flight += 1
                try:
                    await self._tts_synthesize(item.text)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[SentenceSpeechPipeline] tts_synthesize raised: {e}", file=sys.stderr)
                finally:
                    self._tts_in_flight -= 1
        except asyncio.CancelledError:
            pass
        finally:
            self._all_done.set()
            if self._verbose:
                print("[SentenceSpeechPipeline] pump exited", file=sys.stderr)
