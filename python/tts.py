"""Text-to-speech for the app.

Two providers, picked by `tts.provider` in app.config.json:

  * `pocket`     — local pocket-tts-onnx (KevinAHM/pocket-tts-onnx on HF). All
                   non-24l language bundles (english_2026-04, german, italian,
                   portuguese, spanish) are eager-downloaded on first start into
                   CompanionApp/models/. Streaming is sentence-internal via the
                   library's `stream()` method.
  * `elevenlabs` — proxy to any ElevenLabs-compatible HTTP endpoint (e.g. the
                   chatterbox-tts-server). Requests pcm_24000 so we don't decode MP3.

`TtsController` listens for `Tts.Synthesize` envelopes off the WS, runs the chosen
provider in a worker thread, and streams `Tts.AudioChunk` envelopes back to Unity
(plus a terminating `Tts.AudioEnd` or `Tts.Error`).

Voices are per-character, not a global map: each character carries a `voices` entry
keyed by provider — an ElevenLabs voice id, or a pocket-tts embedding (encoded from a
reference clip at character-save time and stored in the character's own folder). The
caller resolves the active provider's entry via `resolve_character_voice` before
synthesizing.

Wire format for audio chunks: payload `{sampleRate, channels, samples}` where
`samples` is base64-encoded little-endian float32 PCM. Unity decodes directly into
its `StreamingAudioBuffer` without going through MP3 / WAV.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator


# ----------- envelope types we own -----------
TYPE_TTS_SYNTHESIZE = 'Tts.Synthesize'              # Unity → us
TYPE_TTS_AUDIO_CHUNK = 'Tts.AudioChunk'             # us → Unity (repeated)
TYPE_TTS_AUDIO_END = 'Tts.AudioEnd'                 # us → Unity (terminator)
TYPE_TTS_ERROR = 'Tts.Error'                        # us → Unity (failure)
TYPE_TTS_PLAYBACK_STARTED = 'Tts.PlaybackStarted'   # Unity → us (first audible sample)
TYPE_TTS_PLAYBACK_ENDED = 'Tts.PlaybackEnded'       # Unity → us (StreamingAudioBuffer drained)

TTS_TYPE_PREFIX = 'Tts.'

# ----------- pocket-tts model metadata -----------
POCKET_TTS_REPO_ID = 'KevinAHM/pocket-tts-onnx'

# English-only deployment. The repo ships non-24l bundles for German / Italian /
# Portuguese / Spanish too (and *_24l larger variants for each); we deliberately
# skip them to keep the on-disk model size down (~440 MB FP32 for english_2026-04
# vs ~2.2 GB if we pulled the full non-24l set). Add languages back here if a
# future feature needs them.
NON_24L_LANGUAGES = ['english_2026-04']

# Sample rate isn't strictly fixed (it comes from each bundle's metadata), but
# pocket-tts uses 24 kHz mono for all the current bundles. The renderer side
# matches and so does the existing StreamingAudioBuffer's expectation.
TTS_TARGET_SR = 24000


# ----------- helpers -----------

def _new_id() -> str:
    return 'm_' + uuid.uuid4().hex


def _env_or(name: str, default):
    v = os.environ.get(name)
    return v if v is not None and v != '' else default


def _coerce_bool(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


# Mirror of C# TtsClient.SanitizeForTts: dashed stutter prefixes like "C-can",
# "Th-thank", "wh-what" come out as a hard letter-by-letter read on most TTS
# voices, so we strip the prefix and leave the bare word. Same regex as the C#
# side (`\b\p{L}{1,2}-(\p{L}+)` in .NET, here approximated with \w letters).
_STUTTER_REGEX = re.compile(r'\b[A-Za-zÀ-ɏ]{1,2}-([A-Za-zÀ-ɏ]+)', re.IGNORECASE)

def sanitize_for_tts(text: str) -> str:
    if not text:
        return text
    return _STUTTER_REGEX.sub(r'\1', text)


# ----------- provider abstraction -----------

class TtsProvider:
    """Stream-synthesize text → (sample_rate, channels, float32 samples) chunks.
    `ensure_loaded()` is called once on a background thread before the first
    synthesize request is served; subclasses can defer heavy imports/downloads
    there. Subclasses must be thread-safe across concurrent `stream_synthesize`
    calls (or document that they aren't and we'll serialize them upstream)."""

    def ensure_loaded(self) -> None:
        pass

    def provider_key(self) -> str:
        """The key this provider's voice entry is stored under in a character's
        `voices` map (e.g. 'pocket', 'elevenlabs'). The controller uses it to pick
        the right per-character entry for the active provider."""
        raise NotImplementedError

    def voice_from_entry(self, entry: dict) -> str:
        """Convert a character's per-provider voice entry (the dict stored under
        `voices[provider_key()]`) into the voice string `stream_synthesize` expects.
        ElevenLabs → the opaque voice id; pocket → the saved embedding's path."""
        return ''

    def default_voice(self) -> str:
        """Returns a voice identifier the provider can synthesize with even if the
        caller's voice map has no entry for the requested character. Subclasses
        can override; the default returns empty string (forcing a "no voice"
        error) so providers without a sensible default fail loudly."""
        return ''

    def stream_synthesize(self, text: str, voice: str) -> Iterator[tuple[int, int, Any]]:
        raise NotImplementedError


def _install_voice_state_cache_patch(engine) -> None:
    """Extend pocket-tts's voice-state cache to cover reference-WAV voices.

    The shipped `PocketTTSOnnx.prepare_voice_state` only caches the conditioning
    state for *built-in* (`predefined_voices`) names — see pocket_tts_onnx.py:349.
    For reference-WAV paths it caches only the mimi_encoder embeddings, then
    re-runs `_condition_with_voice_embeddings` (one `flow_lm_main` forward pass
    on an empty sequence) every Synthesize. With many short sentences this adds
    up. We wrap `prepare_voice_state` so reference-WAV paths populate
    `_voice_state_cache` too; the original cache check on line 354 then
    short-circuits subsequent calls and clones the stored state."""
    import numpy as _np
    original = engine.prepare_voice_state

    def cached(voice):
        if isinstance(voice, _np.ndarray):
            # Pre-computed embedding arrays bypass — the caller controls them
            # and there's no stable key to cache against (object identity is
            # the wrong granularity).
            return original(voice)
        voice_str = str(voice)
        cache = engine._voice_state_cache
        if voice_str in cache:
            return engine._clone_state(cache[voice_str])
        state = original(voice)
        cache[voice_str] = engine._clone_state(state)
        return state

    engine.prepare_voice_state = cached


class PocketTtsProvider(TtsProvider):
    """Local pocket-tts-onnx. Downloads only the non-24l language bundles via
    huggingface_hub.snapshot_download (the bundle dirs are self-contained), then
    instantiates `PocketTTSOnnx` for the configured default language. Switching
    languages mid-session would require holding a per-language cache of the
    inference object — out of scope for v1."""

    def __init__(
        self,
        precision: str,
        language: str,
        lsd_steps: int,
        temperature: float,
    ):
        self._precision = precision
        self._language = language
        self._lsd_steps = lsd_steps
        self._temperature = temperature
        self._engine = None
        self._engine_lock = threading.Lock()
        # Populated by `ensure_loaded()` — caller (factory / TtsController) reads this
        # to resolve the default voice. Kyutai's built-in voice names (alba, etc.)
        # would otherwise trigger a fetch from the gated `kyutai/pocket-tts` repo;
        # pointing at the reference WAV that ships with the ONNX export sidesteps it.
        self.reference_wav_path: str | None = None
        # Paths of `.npy` embedding files we've already loaded + registered into the
        # engine's `_voice_cache`, so repeat synths skip the re-load.
        self._loaded_embeddings: set[str] = set()

    def provider_key(self) -> str:
        return 'pocket'

    def voice_from_entry(self, entry: dict) -> str:
        # Pocket voices are stored as a pre-computed mimi-encoder embedding on disk;
        # the `embeddingFile` path is what we synthesize from.
        return str((entry or {}).get('embeddingFile') or '')

    # Max reference-clip length fed to the mimi encoder. The encoder runs the WHOLE clip in one
    # shot, so a long file (e.g. a full song) balloons activations to tens of GB of RAM. Mirrors
    # KevinAHM/pocket-tts-web (onnx-streaming.js): SAMPLE_RATE * 10 → a 10s cap is plenty to clone.
    _MAX_REFERENCE_SECONDS = 10.0

    def encode_voice_to_file(self, clip_path: str, out_path: str) -> None:
        """Encode a reference voice clip into a mimi-encoder embedding and persist it
        as a `.npy` (the format `stream_synthesize` loads). Heavy (runs the encoder
        ONNX session); call off the event loop. Loads the engine on first use.

        The clip is capped to `_MAX_REFERENCE_SECONDS` first (reading only that span off disk) so a
        long file doesn't blow up encoder memory — the engine still mono-mixes + resamples it."""
        import numpy as np
        self.ensure_loaded()
        if self._engine is None:
            raise RuntimeError('pocket-tts engine unavailable for encoding')
        clip_to_encode, tmp_path = self._cap_reference_clip(clip_path)
        try:
            embeddings = self._engine.encode_voice(clip_to_encode)
            out = Path(out_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out), np.asarray(embeddings, dtype=np.float32))
        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _cap_reference_clip(self, clip_path: str) -> tuple[str, str | None]:
        """Return a clip no longer than `_MAX_REFERENCE_SECONDS` to encode, plus a temp file to
        delete afterwards (None when the original is short enough / can't be inspected). Reads only
        the capped span (`frames=`) so a huge file never loads whole; mono-mixed and written as
        PCM_16 so the engine's wav reader decodes it (it resamples to 24 kHz itself). On any failure
        the original clip is returned unchanged — correctness over the memory guard."""
        try:
            import soundfile as sf
        except Exception:
            return clip_path, None
        try:
            info = sf.info(clip_path)
        except Exception as e:
            print(f'[tts] reference clip info failed ({e}); encoding as-is', file=sys.stderr)
            return clip_path, None
        if info.samplerate <= 0 or info.frames <= 0:
            return clip_path, None
        max_frames = int(self._MAX_REFERENCE_SECONDS * info.samplerate)
        if info.frames <= max_frames:
            return clip_path, None  # already short enough
        try:
            import numpy as np
            import tempfile
            audio, sr = sf.read(clip_path, frames=max_frames, dtype='float32', always_2d=True)
            audio = np.asarray(audio, dtype=np.float32).mean(axis=1)  # stereo → mono
            fd, tmp_path = tempfile.mkstemp(prefix='pkt_ref_', suffix='.wav')
            os.close(fd)
            sf.write(tmp_path, audio, sr, subtype='PCM_16')
            print(f'[tts] reference clip capped {info.frames / info.samplerate:.1f}s → '
                  f'{self._MAX_REFERENCE_SECONDS:.0f}s for encoding', file=sys.stderr)
            return tmp_path, tmp_path
        except Exception as e:
            print(f'[tts] reference clip cap failed ({e}); encoding as-is', file=sys.stderr)
            return clip_path, None

    @staticmethod
    def _allow_patterns(precision: str) -> list[str]:
        # FP32 helpers (`mimi_encoder.onnx`, `text_conditioner.onnx`) are kept
        # FP32 regardless of precision — the README explicitly says so. The
        # int8 vs fp32 choice only swaps three files per bundle.
        # `reference_sample.wav` is the voice-cloning reference the repo ships
        # with — we default to it instead of a built-in name like "alba", which
        # would otherwise trigger a fetch from the gated `kyutai/pocket-tts` repo.
        # pocket_tts_onnx.py is vendored into the project now (see python/pocket_tts_onnx.py);
        # we only fetch the model weights + the reference voice from the snapshot.
        patterns: list[str] = ['reference_sample.wav']
        for lang in NON_24L_LANGUAGES:
            if precision == 'int8':
                patterns += [
                    f'onnx/{lang}/flow_lm_main_int8.onnx',
                    f'onnx/{lang}/flow_lm_flow_int8.onnx',
                    f'onnx/{lang}/mimi_decoder_int8.onnx',
                ]
            else:
                patterns += [
                    f'onnx/{lang}/flow_lm_main.onnx',
                    f'onnx/{lang}/flow_lm_flow.onnx',
                    f'onnx/{lang}/mimi_decoder.onnx',
                ]
            patterns += [
                f'onnx/{lang}/mimi_encoder.onnx',     # always FP32
                f'onnx/{lang}/text_conditioner.onnx', # always FP32
                f'onnx/{lang}/tokenizer.model',
                f'onnx/{lang}/bos_before_voice.npy',
                f'onnx/{lang}/bundle.json',
            ]
        return patterns

    def ensure_loaded(self) -> None:
        with self._engine_lock:
            if self._engine is not None:
                return
            print(f'[tts] downloading pocket-tts bundles (non-24l, {self._precision})…')
            from huggingface_hub import snapshot_download  # transitively loads heavy stack
            import download_progress
            repo_path = snapshot_download(
                repo_id=POCKET_TTS_REPO_ID,
                allow_patterns=self._allow_patterns(self._precision),
                tqdm_class=download_progress.tqdm_class(),
            )
            ref_wav = Path(repo_path) / 'reference_sample.wav'
            self.reference_wav_path = str(ref_wav) if ref_wav.exists() else None
            print(f'[tts] loading pocket-tts language={self._language} precision={self._precision}…')
            # Import the vendored engine (python/pocket_tts_onnx.py) — NOT the copy that used to
            # live in the snapshot — so its deps are bundled in the frozen build.
            from pocket_tts_onnx import PocketTTSOnnx
            engine = PocketTTSOnnx(
                models_dir=str(Path(repo_path) / 'onnx'),
                language=self._language,
                precision=self._precision,
                temperature=self._temperature,
                lsd_steps=self._lsd_steps,
            )
            _install_voice_state_cache_patch(engine)
            self._engine = engine
            print(f'[tts] pocket-tts ready ({engine.sample_rate} Hz)')

    def default_voice(self) -> str:
        # Path to `reference_sample.wav` if the snapshot includes it; otherwise
        # empty string and TtsController will error loudly for unmapped characters.
        return self.reference_wav_path or ''

    def _prepare_voice(self, voice: str) -> str:
        """For a `.npy` embedding path, load it once and register it under the
        engine's `_voice_cache` keyed by the path, then return the path as the
        voice key. `prepare_voice_state` then finds the embedding in `_voice_cache`
        and the voice-state cache patch caches the conditioned state by that key —
        so subsequent synths for the same character skip both the disk load and
        re-conditioning. Non-`.npy` voices (built-in names, WAV/safetensors paths)
        pass straight through."""
        if not voice or not voice.lower().endswith('.npy'):
            return voice
        if voice not in self._loaded_embeddings:
            import numpy as np
            embeddings = np.load(voice)
            # The engine treats anything in `_voice_cache` as a known voice keyed by
            # the value string (see PocketTTSOnnx.prepare_voice_state).
            self._engine._voice_cache[voice] = embeddings
            self._loaded_embeddings.add(voice)
        return voice

    def stream_synthesize(self, text: str, voice: str) -> Iterator[tuple[int, int, Any]]:
        if self._engine is None:
            raise RuntimeError('pocket-tts not loaded')
        sr = int(self._engine.sample_rate)
        for samples in self._engine.stream(text, self._prepare_voice(voice)):
            yield (sr, 1, samples)


class ElevenLabsProvider(TtsProvider):
    """Proxy to an ElevenLabs-compatible HTTP endpoint. Requests pcm_24000 so we
    can stream int16 LE bytes straight through without an MP3 decode. Uses httpx
    (already a transitive dep via huggingface_hub) for streaming I/O."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 stability: float, similarity_boost: float,
                 use_speaker_boost: bool, speed: float,
                 request_timeout_seconds: int):
        self._base_url = (base_url or '').rstrip('/')
        self._api_key = api_key or ''
        self._model = model
        self._stability = stability
        self._similarity_boost = similarity_boost
        self._use_speaker_boost = use_speaker_boost
        self._speed = speed
        self._timeout = float(request_timeout_seconds) if request_timeout_seconds else 30.0

    def provider_key(self) -> str:
        return 'elevenlabs'

    def voice_from_entry(self, entry: dict) -> str:
        return str((entry or {}).get('voiceId') or '')

    def stream_synthesize(self, text: str, voice_id: str) -> Iterator[tuple[int, int, Any]]:
        if not self._base_url:
            raise RuntimeError('ElevenLabs baseUrl is empty')
        if not voice_id:
            raise RuntimeError('ElevenLabs voice id is empty')
        import httpx
        import numpy as np
        url = f"{self._base_url}/{voice_id}/stream?output_format=pcm_{TTS_TARGET_SR}"
        body = {
            'text': text,
            'model_id': self._model,
            'voice_settings': {
                'stability': self._stability,
                'similarity_boost': self._similarity_boost,
                'use_speaker_boost': self._use_speaker_boost,
                'style': 0,
                'speed': self._speed,
            },
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': f'audio/L16;rate={TTS_TARGET_SR}',
        }
        if self._api_key:
            headers['xi-api-key'] = self._api_key

        # Carry-byte handling for chunks that end mid-sample. int16 LE = 2 bytes.
        carry = b''
        with httpx.stream('POST', url, json=body, headers=headers, timeout=self._timeout) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes(chunk_size=4096):
                if not chunk:
                    continue
                buf = carry + chunk
                aligned_len = (len(buf) // 2) * 2
                if aligned_len == 0:
                    carry = buf
                    continue
                pcm16 = np.frombuffer(buf[:aligned_len], dtype='<i2')
                carry = buf[aligned_len:]
                samples = pcm16.astype(np.float32) / 32768.0
                yield (TTS_TARGET_SR, 1, samples)


# ----------- TtsController: dispatch + WS responses -----------

class TtsController:
    """Owns the active TTS provider. Bridge delegates every `Tts.Synthesize`
    envelope here; the controller runs synthesis on a worker thread and streams
    `Tts.AudioChunk` envelopes back via the supplied `send_envelope` callable.
    Provider load happens once at startup on a background thread; synthesize
    requests that arrive earlier block on a threading.Event until ready (or fail
    if the load itself failed).

    Voices are no longer kept in a global map — each character carries its own
    per-provider `voices` entry, resolved to a voice string via
    `resolve_character_voice` before synthesis."""

    def __init__(
        self,
        send_envelope: Callable[[dict], None],
        provider: TtsProvider,
    ):
        self._send = send_envelope
        self._provider = provider
        self._ready = threading.Event()
        self._load_error: str | None = None
        # Serialize concurrent Synthesize requests: pocket-tts maintains internal
        # ORT session state during `stream()` and isn't safe to call from two
        # threads at once. ElevenLabs would tolerate concurrency, but pipelining
        # sentences for a single reply doesn't help — playback is sequential.
        self._synth_lock = threading.Lock()
        # Barge-in plumbing. `_active_id` is the request id currently being
        # synthesized (or None if idle); `_cancel_id` is the id of the request
        # the caller asked us to abort. The synth loop checks the latter every
        # chunk and breaks out early if they match — that's what lets the STT
        # path interrupt an in-flight assistant reply when the user starts
        # talking. State lives under `_active_lock`, separate from `_synth_lock`
        # so `is_active()` / `cancel_active()` can be called from the audio /
        # worker threads without blocking on the running synth.
        self._active_id: str | None = None
        self._cancel_id: str | None = None
        self._active_lock = threading.Lock()
        # `_speakers_busy` is the canonical "audio is currently audible" flag.
        # Set True when we send the first chunk of a request, cleared only when
        # Unity reports back with Tts.PlaybackEnded (its StreamingAudioBuffer
        # has finished draining). This is what the STT path uses to gate the
        # recognizer — no guessed tail, no clock-alignment math.
        self._speakers_busy = False

    # ----- public -----

    def start_load(self) -> None:
        threading.Thread(target=self._load_worker, name='tts-load', daemon=True).start()

    def is_ready(self) -> bool:
        """True once the active provider's load attempt has finished — set on SUCCESS and on
        FAILURE alike (the load worker sets `_ready` in its finally), so a terminal load error
        still counts as 'resolved'. A pending provider switch (`reload`) swaps in a fresh, unset
        event, so this correctly reports False again until the new provider finishes loading.
        Used by the chat-loading overlay gate."""
        return self._ready.is_set()

    def reload(self, provider: 'TtsProvider') -> None:
        """Hot-swap the active provider without recreating the controller — so the
        bound hooks the STT/chat paths captured at startup (is_active /
        cancel_active / synthesize_text) stay valid. Any in-flight synth is
        cancelled first, then we wait on `_synth_lock` so the swap can't race a
        running `stream_synthesize`. A fresh `_ready` Event re-gates requests until
        the new provider finishes loading on a background thread."""
        self.cancel_active('provider reload')
        with self._synth_lock:
            self._provider = provider
            # New gate: requests block until the swapped-in provider loads.
            self._ready = threading.Event()
            self._load_error = None
        print('[tts] provider reloaded; reloading model…')
        self.start_load()

    def provider_key(self) -> str:
        """The active provider's key (e.g. 'pocket' / 'elevenlabs') — the key its
        per-character voice entry is stored under."""
        return self._provider.provider_key()

    def pocket_provider(self) -> 'PocketTtsProvider | None':
        """The active provider if it's pocket-tts (so the embedding encoder can
        reuse its already-loaded engine), else None."""
        return self._provider if isinstance(self._provider, PocketTtsProvider) else None

    def resolve_character_voice(self, voices: dict) -> str:
        """Pick the active provider's entry from a character's `voices` map and turn
        it into the voice string `stream_synthesize` wants. Returns '' when the
        character has no voice for the active provider — the caller runs in
        no-voice mode rather than speaking in some default voice."""
        entry = (voices or {}).get(self._provider.provider_key())
        return self._provider.voice_from_entry(entry) if entry else ''

    def has_character_voice(self, voices: dict) -> bool:
        """Whether the given character `voices` map yields a usable voice for the
        ACTIVE provider. Drives whether voice mode can be enabled at all."""
        return bool(self.resolve_character_voice(voices))

    def is_tts_envelope_type(self, type_: str) -> bool:
        return isinstance(type_, str) and type_.startswith(TTS_TYPE_PREFIX)

    def handle_envelope(self, env: dict) -> None:
        """Bridge calls this for any envelope with type starting in 'Tts.'. Three
        types we act on:
          * Tts.Synthesize       — Unity asking us to speak.
          * Tts.PlaybackStarted  — Unity emitted the first audible TTS sample;
                                   flip _speakers_busy so STT mutes the recognizer.
          * Tts.PlaybackEnded    — Unity's StreamingAudioBuffer drained; flip
                                   _speakers_busy off so STT un-mutes.
        Everything else under `Tts.*` is outbound (us → Unity) and lands here only
        because Bridge prefix-routes the whole family through us."""
        t = env.get('type')
        if t == TYPE_TTS_SYNTHESIZE:
            # Dispatch to a worker so the WS reader thread isn't tied up for the
            # duration of TTS (multi-second per sentence).
            threading.Thread(
                target=self._handle_synthesize, args=(env,),
                name='tts-synth', daemon=True,
            ).start()
        elif t == TYPE_TTS_PLAYBACK_STARTED:
            with self._active_lock:
                self._speakers_busy = True
            print('[tts] playback started (unity emitted first sample)')
        elif t == TYPE_TTS_PLAYBACK_ENDED:
            with self._active_lock:
                self._speakers_busy = False
            print('[tts] playback ended (unity reported buffer drained)')

    def is_active(self) -> bool:
        """True iff a synthesize request is currently streaming chunks. Polled by
        the STT path to decide whether a fresh user-speech onset should trigger
        barge-in (no point cancelling silence)."""
        with self._active_lock:
            return self._active_id is not None

    def is_playing(self) -> bool:
        """True iff audio we sent is currently being played on Unity's side —
        either we're still streaming chunks, OR we've finished streaming but
        Unity hasn't reported the buffer drained yet. This is the canonical
        flag for echo suppression in the STT path."""
        with self._active_lock:
            return self._active_id is not None or self._speakers_busy

    def synthesize_text(self, text: str, voice: str = '', request_id: str | None = None) -> None:
        """Synchronous synthesize-and-stream. Same effect as receiving a Tts.Synthesize
        envelope, but inline so async callers (the Python orchestrator's speech pipeline)
        can `asyncio.to_thread(synthesize_text, ...)` and await completion. `voice` is the
        already-resolved provider voice string (see `resolve_character_voice`); empty falls
        back to the provider default. Blocks until the last Tts.AudioChunk + the terminating
        Tts.AudioEnd have been dispatched to Unity. Caller-supplied `request_id` lets the
        speech pipeline correlate barge-in cancels against a specific utterance; omit to
        auto-generate."""
        env = {
            'id': request_id or _new_id(),
            'type': TYPE_TTS_SYNTHESIZE,
            'payload': {'text': text or '', 'voice': voice or ''},
        }
        self._handle_synthesize(env)

    def cancel_active(self, reason: str = '') -> bool:
        """Mark the currently-active synth for cancellation. The synth loop
        checks this on every chunk and bails out cleanly, sending a Tts.AudioEnd
        to Unity so the playback buffer drains rather than holding the last
        chunk forever. Returns False if nothing was active (caller can skip the
        wire-side Tts.Cancel)."""
        with self._active_lock:
            if self._active_id is None:
                return False
            self._cancel_id = self._active_id
        if reason:
            print(f'[tts] cancel requested: {reason}')
        return True

    # ----- internals -----

    def _load_worker(self) -> None:
        try:
            self._provider.ensure_loaded()
            self._load_error = None
        except Exception as e:
            self._load_error = f'{type(e).__name__}: {e}'
            print(f'[tts] load failed: {self._load_error}', file=sys.stderr)
        finally:
            self._ready.set()

    def _handle_synthesize(self, env: dict) -> None:
        req_id = env.get('id') or ''
        payload = env.get('payload') or {}
        text = str(payload.get('text') or '').strip()
        # The voice is resolved by the caller (resolve_character_voice) and passed in
        # the payload; empty falls back to the provider default.
        voice = str(payload.get('voice') or '')
        if not text:
            self._send_error(req_id, 'TTS text is empty.')
            return

        # Block until the model has finished loading. A common pattern is the
        # first assistant message hitting this before the snapshot_download
        # completes; rather than dropping the sentence we just wait.
        self._ready.wait()
        if self._load_error is not None:
            self._send_error(req_id, f'TTS provider failed to load: {self._load_error}')
            return

        if not voice:
            voice = self._provider.default_voice()
        if not voice:
            self._send_error(req_id, 'No voice configured for this character (and no provider default).')
            return

        sanitized = sanitize_for_tts(text)
        with self._synth_lock:
            with self._active_lock:
                self._active_id = req_id
                self._cancel_id = None
                # NOTE: do NOT set _speakers_busy here. The "speakers playing"
                # flag is driven entirely by Unity's Tts.PlaybackStarted /
                # Tts.PlaybackEnded envelopes — Unity is the authoritative
                # source on whether sound is actually coming out of the
                # speakers right now. Setting it here would (a) mute too early
                # (synth runs before audio plays) and (b) miss the post-tool
                # case if Unity treats pre-tool/post-tool as separate playback
                # sessions and fires PlaybackEnded between them.
            try:
                cancelled = False
                for sample_rate, channels, samples in self._provider.stream_synthesize(sanitized, voice):
                    # Check before send so a barge-in observed mid-iteration
                    # doesn't push one more chunk past the cancellation. The
                    # check is cheap (lock + two pointer compares) compared to
                    # synthesis itself.
                    with self._active_lock:
                        if self._cancel_id == req_id:
                            cancelled = True
                            break
                    self._send_chunk(req_id, sample_rate, channels, samples)
                # Always send AudioEnd — Unity's StreamingAudioBuffer treats it
                # as "no more chunks coming, you can stop polling". The barge-in
                # case still wants this; the wire-side Tts.Cancel that the STT
                # path sends in parallel is what tells Unity to drop the chunks
                # already in its buffer.
                self._send_end(req_id)
                if cancelled:
                    print(f'[tts] synth {req_id[:8]}… cancelled mid-stream')
            except Exception as e:
                print(f'[tts] synth failed: {e}', file=sys.stderr)
                self._send_error(req_id, f'TTS synthesis failed: {e}')
            finally:
                with self._active_lock:
                    if self._active_id == req_id:
                        self._active_id = None
                    self._cancel_id = None

    def _send_chunk(self, req_id: str, sample_rate: int, channels: int, samples) -> None:
        import numpy as np
        arr = np.asarray(samples, dtype=np.float32)
        b64 = base64.b64encode(arr.tobytes()).decode('ascii')
        self._send({
            'id': _new_id(),
            'type': TYPE_TTS_AUDIO_CHUNK,
            'replyTo': req_id,
            'payload': {
                'sampleRate': int(sample_rate),
                'channels': int(channels),
                'samples': b64,
            },
        })

    def _send_end(self, req_id: str) -> None:
        self._send({
            'id': _new_id(),
            'type': TYPE_TTS_AUDIO_END,
            'replyTo': req_id,
            'payload': {},
        })

    def _send_error(self, req_id: str, message: str) -> None:
        self._send({
            'id': _new_id(),
            'type': TYPE_TTS_ERROR,
            'replyTo': req_id,
            'payload': {'message': message},
        })


# ----------- factory -----------

def tts_config_signature(config: dict) -> str:
    """Stable string fingerprint of everything `build_provider`
    consumes (the whole `tts` block + the env overrides that can shadow it).
    Lets the caller skip a needless provider reload when an unrelated setting
    was saved — pocket-tts rebuild means re-instantiating the ONNX engine, so
    it's worth avoiding when nothing TTS-related actually changed."""
    tts_cfg = (config.get('tts') if isinstance(config, dict) else None) or {}
    env_keys = (
        'APP_TTS_PROVIDER', 'APP_TTS_PRECISION', 'APP_TTS_LANGUAGE',
        'APP_TTS_LSD_STEPS', 'APP_TTS_TEMPERATURE',
        'APP_TTS_BASE_URL', 'APP_TTS_API_KEY',
    )
    env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k)}
    return json.dumps({'tts': tts_cfg, 'env': env}, sort_keys=True, default=str)


def build_pocket_encoder(config: dict) -> PocketTtsProvider:
    """A PocketTtsProvider built solely for encoding reference clips into embeddings,
    independent of the active TTS provider. Used when a character's pocket voice needs
    its embedding generated while ElevenLabs (say) is the active provider — see
    'generate on demand'. Honors the same `tts.pocket` precision/language settings."""
    tts_cfg = (config.get('tts') if isinstance(config, dict) else None) or {}
    p_cfg = tts_cfg.get('pocket') or {}
    precision = str(_env_or('APP_TTS_PRECISION', p_cfg.get('precision', 'int8'))).lower()
    if precision not in ('int8', 'fp32'):
        precision = 'int8'
    language = str(_env_or('APP_TTS_LANGUAGE', p_cfg.get('language', 'english_2026-04')))
    lsd_steps = int(_env_or('APP_TTS_LSD_STEPS', p_cfg.get('lsdSteps', 1)))
    temperature = float(_env_or('APP_TTS_TEMPERATURE', p_cfg.get('temperature', 0.7)))
    return PocketTtsProvider(
        precision=precision, language=language,
        lsd_steps=lsd_steps, temperature=temperature,
    )


def build_provider(config: dict) -> TtsProvider:
    """Construct the active TTS provider from a parsed app.config.json dict.
    Shared by the startup factory and the hot-reload path so both interpret
    `tts.provider` (and its env overrides) identically."""
    tts_cfg = (config.get('tts') if isinstance(config, dict) else None) or {}
    provider_name = str(_env_or('APP_TTS_PROVIDER', tts_cfg.get('provider', 'pocket'))).lower()

    if provider_name == 'pocket':
        return build_pocket_encoder(config)
    elif provider_name == 'elevenlabs':
        e_cfg = tts_cfg.get('elevenlabs') or {}
        return ElevenLabsProvider(
            base_url=str(_env_or('APP_TTS_BASE_URL', e_cfg.get('baseUrl', ''))),
            api_key=str(_env_or('APP_TTS_API_KEY', e_cfg.get('apiKey', ''))),
            model=str(e_cfg.get('model', '')),
            stability=float(e_cfg.get('stability', 0.5)),
            similarity_boost=float(e_cfg.get('similarityBoost', 0.75)),
            use_speaker_boost=_coerce_bool(e_cfg.get('useSpeakerBoost', True)),
            speed=float(e_cfg.get('speed', 1.0)),
            request_timeout_seconds=int(e_cfg.get('requestTimeoutSeconds', 30)),
        )
    raise ValueError(f'Unknown TTS provider in config: {provider_name!r}. Expected "pocket" or "elevenlabs".')


def make_tts_controller(
    config: dict,
    send_envelope: Callable[[dict], None],
) -> TtsController:
    """Build a TtsController from a parsed app.config.json dict.

    `send_envelope(env)` is invoked whenever the controller wants to send an
    envelope to Unity — typically `Bridge.send_envelope`.
    """
    return TtsController(
        send_envelope=send_envelope,
        provider=build_provider(config),
    )
