"""Streaming speech-to-text + mic capture for the app.

Built on `sherpa-onnx` (https://github.com/k2-fsa/sherpa-onnx). One streaming
`OnlineRecognizer` (English Zipformer) serves both modes:

  * Push-to-talk — user toggles mic; partials stream live; final transcript on stop
    (or on the recognizer's endpoint detector firing, which acts as auto-stop).
  * Hands-free — mic stays open; the endpoint detector segments utterances; each
    final transcript is emitted to the renderer and, if `autoSubmit` is on, also
    submitted to Unity as a `Chat.SubmitUserMessage` (no Send-button click needed).

A second sherpa-onnx pass — `VoiceActivityDetector` (Silero VAD) — runs alongside
the recognizer in hands-free mode to detect the *start* of speech for barge-in:
when the user starts talking while TTS is playing, we cancel the in-flight TTS
synthesis and emit a `Tts.Cancel` envelope so Unity can drain its playback buffer.

Public surface (everything below the helpers is intentionally private):
  - `make_voice_controller(config, push_envelope, send_envelope, tts_hooks)`
    factory; reads `config['stt']` + `APP_*` env-var overrides.
  - `VoiceController.start_load()` — kick off model loads on a background thread.
  - `VoiceController.patch_unity_envelope(env)` — yields envelopes to push to the
    renderer for an envelope coming from Unity. For Chat.Init we patch
    `voiceSupported=True` and follow up with envelopes carrying current state.
  - `VoiceController.intercept_renderer_send(type, payload)` — returns True iff
    the envelope is a voice control message we handle locally instead of
    forwarding to Unity.
"""
from __future__ import annotations

import os
import queue
import re
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterable

# ----------- protocol constants we own -----------
# Stay in this module: voice is the only consumer of these strings outside the type
# definitions in the renderer's protocol.ts.
# WS protocol types (Python ↔ Unity over WebSocket):
TYPE_CHAT_INIT = 'Chat.Init'                                   # Unity → us (patched before forward)
TYPE_CHAT_SUBMIT_USER_MESSAGE = 'Chat.SubmitUserMessage'       # us → Unity (auto-submit path)
TYPE_TTS_CANCEL = 'Tts.Cancel'                                 # us → Unity (barge-in)

# Internal renderer-bound event types — NOT WS envelopes. These ride the
# pywebview JS bridge (window.__chatPush) and surface in the React side as
# App events; they never leave the app process. Mirrors the type
# strings declared in src/renderer/appEvents.ts.
EVT_VOICE_RECORDING = 'Voice.Recording'                        # us → renderer (active flag)
EVT_VOICE_BUSY = 'Voice.Busy'                                  # us → renderer (loading/transcribing)
EVT_VOICE_TRANSCRIPT = 'Voice.Transcript'                      # us → renderer (final text)
EVT_VOICE_PARTIAL = 'Voice.Partial'                            # us → renderer (in-progress text)
EVT_HANDS_FREE_CHANGED = 'Voice.HandsFreeChanged'              # us → renderer (state echo)
EVT_AUTO_SUBMIT_CHANGED = 'Voice.AutoSubmitChanged'            # us → renderer (state echo)
EVT_WAKE_WORD_CHANGED = 'Voice.WakeWordChanged'                # us → renderer (state echo)
EVT_WAKE_WORD_ARMED = 'Voice.WakeWordArmed'                    # us → renderer (armed status pulse)

# ----------- STT defaults -----------
STT_TARGET_SR = 16000
# Audio block size pushed from the PortAudio callback into the worker queue. 100 ms
# at 16 kHz — same cadence the sherpa-onnx microphone examples use. Smaller windows
# tighten partial-transcript latency at the cost of more CPU per decode loop; larger
# windows trade latency for throughput. 100 ms hits the sweet spot.
STT_BLOCK_SAMPLES = 1600
# Silero VAD requires 16 kHz mono, 512-sample windows. sherpa-onnx's
# VoiceActivityDetector handles the windowing internally, but we set this here so
# the constant is one place to change if the model swaps.
VAD_WINDOW_SAMPLES = 512
# Where to fetch silero_vad.onnx if it isn't already cached. k2-fsa's release tag
# pins a known-compatible build — safer than chasing snakers4/silero-vad HEAD.
SILERO_VAD_URL = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx'

# Catalog of sherpa-onnx pre-packaged streaming model bundles on HuggingFace. Keys
# are user-facing aliases (the `model` field in app.config.json / the
# `APP_STT_MODEL` env var). Values tell SttEngine which `from_*` factory to
# use and which files to pull from the bundle's HF repo.
#
# Bundle filenames come from each repo's tree on HuggingFace; the streaming
# Zipformer bundles all follow the same encoder/decoder/joiner-epoch-99-avg-1 shape
# but differ in suffix (chunk-16-left-128 on the standard model, none on the 20M).
STT_MODEL_CONFIGS = {
    # === Top tier: 2026 NeMo (broadest training, modern arch) ===
    # NVIDIA Nemotron-Speech-Streaming 0.6B — Cache-Aware FastConformer-RNNT,
    # 600M params, trained on ~250k hours of English (LibriLight + LibriSpeech +
    # Fisher + Switchboard + MLS + People's Speech + YODAS2 + Common Voice +
    # VoxPopuli + Europarl + FLEURS + NVIDIA Granary). Native punctuation +
    # capitalization. Published WERs on conversational benchmarks (AMI 11.7,
    # Earnings22 12.5, TEDLIUM 3.5, GigaSpeech 9.7) — the only streaming
    # English bundle in this zoo with real-world conversational WERs published.
    # Default. 663 MB int8; license is NVIDIA Open Model License.
    'nemotron-0.6b': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14',
        'files': {
            'encoder': 'encoder.int8.onnx',
            'decoder': 'decoder.int8.onnx',
            'joiner':  'joiner.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    # === Middle tier: NeMo FastConformer-Hybrid streaming (May 2024) ===
    # Same training-corpus family as Nemotron (NeMo ASRSET 3.0 — much broader
    # than LS+GigaSpeech) but 114M params instead of 600M. 138 MB int8. Pick
    # one of three chunk sizes: 1040ms (highest accuracy, WER LS-other 5.4),
    # 480ms (balanced, 5.7), or 80ms (lowest latency, 6.4).
    'fast-conformer-1040ms': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms-int8',
        'files': {
            'encoder': 'encoder.int8.onnx',
            'decoder': 'decoder.int8.onnx',
            'joiner':  'joiner.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    'fast-conformer-480ms': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms-int8',
        'files': {
            'encoder': 'encoder.int8.onnx',
            'decoder': 'decoder.int8.onnx',
            'joiner':  'joiner.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    'fast-conformer-80ms': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms-int8',
        'files': {
            'encoder': 'encoder.int8.onnx',
            'decoder': 'decoder.int8.onnx',
            'joiner':  'joiner.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    # === Lower tier: 2023 Zipformer fallbacks (Apache-2.0, smaller) ===
    # LibriSpeech + GigaSpeech (~11k hours total). Decent on read speech, falls
    # behind NeMo on conversational. Kept as a permissive-license fallback.
    'streaming-zipformer-en': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21',
        'files': {
            'encoder': 'encoder-epoch-99-avg-1.int8.onnx',
            'decoder': 'decoder-epoch-99-avg-1.int8.onnx',
            'joiner':  'joiner-epoch-99-avg-1.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    # LibriSpeech-only full-size streaming Zipformer. Smaller than -2023-06-21
    # (~250 MB int8) but trained on a narrower corpus — only worth it if your
    # audio is close to read speech.
    'streaming-zipformer-en-libri': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26',
        'files': {
            'encoder': 'encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            'decoder': 'decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            'joiner':  'joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
    # Tiny ~20M-param streaming Zipformer. Lowest latency and only ~75 MB on
    # disk, at noticeably worse accuracy. RAM-constrained / first-run-size
    # scenarios only.
    'streaming-zipformer-en-20M': {
        'kind': 'transducer',
        'hf_repo': 'csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17',
        'files': {
            'encoder': 'encoder-epoch-99-avg-1.int8.onnx',
            'decoder': 'decoder-epoch-99-avg-1.int8.onnx',
            'joiner':  'joiner-epoch-99-avg-1.int8.onnx',
            'tokens':  'tokens.txt',
        },
    },
}


def _new_id() -> str:
    return 'm_' + uuid.uuid4().hex


def _env_or(name: str, default):
    v = os.environ.get(name)
    return v if v is not None and v != '' else default


def _coerce_bool(v) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _hf_cache_dir() -> str:
    """Mirror app.py's MODELS_DIR redirect — HF_HOME is the one place we know
    the app has agreed to drop large model files. Falls back to ~/.cache/huggingface
    if neither env var is set (shouldn't happen — app.py sets both at import
    time — but defensive code costs nothing)."""
    for var in ('HF_HUB_CACHE', 'HF_HOME'):
        v = os.environ.get(var)
        if v:
            return v
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')


def _ensure_silero_vad() -> str:
    """Download silero_vad.onnx into the HF cache if it isn't there and return the
    absolute path. Uses a plain HTTP fetch instead of huggingface_hub because the
    file is hosted as a GitHub release asset, not as a HF repo file."""
    target = os.path.join(_hf_cache_dir(), 'silero_vad.onnx')
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    print(f'[stt]   fetching silero_vad.onnx → {target}')
    import urllib.request
    # Atomic-rename so a half-downloaded file doesn't poison the cache.
    tmp = target + '.part'
    urllib.request.urlretrieve(SILERO_VAD_URL, tmp)
    os.replace(tmp, target)
    return target


# ----------- streaming engine -----------

class SttEngine:
    """Lazy-loads a sherpa-onnx OnlineRecognizer on a background thread so the chat
    window can open immediately. The first download takes seconds; we publish
    `ready` only after the first decode loop has run a warm-up pass."""

    def __init__(
        self,
        model_name: str,
        kind_override: str | None = None,
        num_threads: int = 1,
        provider: str = 'cpu',
        rule1_silence_s: float = 2.4,
        rule2_silence_s: float = 1.2,
        rule3_utterance_s: float = 20.0,
        on_state_change: Callable[[], None] | None = None,
    ):
        self._model_name = model_name
        self._kind_override = kind_override
        self._num_threads = max(1, int(num_threads))
        self._provider = provider
        self._rule1 = rule1_silence_s
        self._rule2 = rule2_silence_s
        self._rule3 = rule3_utterance_s
        self._recognizer = None
        self._ready = False
        self._error: str | None = None
        self._lock = threading.Lock()
        self._on_state_change = on_state_change

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def load_async(self) -> None:
        threading.Thread(target=self._load, name='stt-load', daemon=True).start()

    def _resolve_config(self) -> dict:
        """Look up the model alias in STT_MODEL_CONFIGS. Unknown aliases fall through
        as raw `<owner>/<repo>` HF IDs — in that case the caller must have set
        `kind_override` (APP_STT_KIND)."""
        cfg = STT_MODEL_CONFIGS.get(self._model_name)
        if cfg is not None:
            if self._kind_override:
                cfg = {**cfg, 'kind': self._kind_override}
            return cfg
        if not self._kind_override:
            raise RuntimeError(
                f"Unknown STT model alias '{self._model_name}'. Pick one of "
                f"{sorted(STT_MODEL_CONFIGS)} or supply APP_STT_KIND.")
        defaults = {
            'transducer': {
                'encoder': 'encoder.onnx',
                'decoder': 'decoder.onnx',
                'joiner':  'joiner.onnx',
                'tokens':  'tokens.txt',
            },
        }
        files = defaults.get(self._kind_override)
        if files is None:
            raise RuntimeError(
                f"Unsupported APP_STT_KIND '{self._kind_override}' "
                f"(streaming only — must be 'transducer').")
        return {'kind': self._kind_override, 'hf_repo': self._model_name, 'files': files}

    def _download_files(self, hf_repo: str, files: dict) -> dict:
        """Pull each bundle file from HuggingFace into the local HF cache."""
        from huggingface_hub import hf_hub_download
        import download_progress
        out: dict[str, str] = {}
        for key, filename in files.items():
            print(f'[stt]   fetching {hf_repo}/{filename}')
            out[key] = hf_hub_download(repo_id=hf_repo, filename=filename,
                                       tqdm_class=download_progress.tqdm_class())
        return out

    def _build_recognizer(self, kind: str, paths: dict):
        import sherpa_onnx
        if kind != 'transducer':
            raise RuntimeError(f"Only streaming transducer models are supported; got '{kind}'.")
        print(f'[stt] endpoint rules: rule1={self._rule1}s rule2={self._rule2}s rule3={self._rule3}s')
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=paths['tokens'],
            encoder=paths['encoder'],
            decoder=paths['decoder'],
            joiner=paths['joiner'],
            num_threads=self._num_threads,
            sample_rate=STT_TARGET_SR,
            feature_dim=80,
            provider=self._provider,
            decoding_method='greedy_search',
            # Endpoint detection drives both PTT auto-stop (when enabled) and
            # hands-free utterance segmentation. The rule numbers mean:
            #   rule1: silence_s after no decoded frames yet     → "user started silent then stopped"
            #   rule2: silence_s after at least one decoded frame → "user paused after speaking"
            #   rule3: utterance_s total length                  → "stop runaway utterances"
            # Defaults mirror the streaming-decode-from-microphone-with-endpoint-detection.py
            # example in the sherpa-onnx repo.
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self._rule1,
            rule2_min_trailing_silence=self._rule2,
            rule3_min_utterance_length=self._rule3,
        )

    def _load(self) -> None:
        try:
            cfg = self._resolve_config()
            print(f'[stt] loading {cfg["hf_repo"]} (kind={cfg["kind"]})…')
            paths = self._download_files(cfg['hf_repo'], cfg['files'])
            recognizer = self._build_recognizer(cfg['kind'], paths)
            # Warm-up decode: feeding 1 s of silence forces ORT to compile its
            # execution-plan kernels here (while the renderer shows "loading…")
            # instead of on the user's first mic press.
            print('[stt] warming up ORT kernels…')
            try:
                import numpy as np
                s = recognizer.create_stream()
                s.accept_waveform(STT_TARGET_SR, np.zeros(STT_TARGET_SR, dtype=np.float32))
                while recognizer.is_ready(s):
                    recognizer.decode_stream(s)
            except Exception as e:
                print(f'[stt] warm-up failed (model still usable): {e}', file=sys.stderr)
            with self._lock:
                self._recognizer = recognizer
                self._ready = True
            print('[stt] model ready')
        except Exception as e:
            with self._lock:
                self._error = f'{type(e).__name__}: {e}'
            print(f'[stt] load failed: {self._error}', file=sys.stderr)
        finally:
            cb = self._on_state_change
            if cb is not None:
                try: cb()
                except Exception as e:
                    print(f'[stt] state callback failed: {e}', file=sys.stderr)

    # Streaming surface — VoiceController drives these directly.

    def create_stream(self):
        with self._lock:
            r = self._recognizer
        if r is None:
            raise RuntimeError('STT model not loaded yet')
        return r.create_stream()

    def feed(self, stream, samples) -> None:
        """Push a chunk of float32 PCM @ 16 kHz into the stream and drain decode work."""
        with self._lock:
            r = self._recognizer
        if r is None or stream is None:
            return
        stream.accept_waveform(STT_TARGET_SR, samples)
        while r.is_ready(stream):
            r.decode_stream(stream)

    def get_text(self, stream) -> str:
        with self._lock:
            r = self._recognizer
        if r is None or stream is None:
            return ''
        return (r.get_result(stream) or '').strip()

    def is_endpoint(self, stream) -> bool:
        with self._lock:
            r = self._recognizer
        if r is None or stream is None:
            return False
        return bool(r.is_endpoint(stream))

    def reset(self, stream) -> None:
        with self._lock:
            r = self._recognizer
        if r is not None and stream is not None:
            r.reset(stream)

    def finalize(self, stream) -> str:
        """Mark the input finished, drain remaining decode work, return final text.
        Used for PTT when the user releases the mic — equivalent to forcing an
        endpoint."""
        with self._lock:
            r = self._recognizer
        if r is None or stream is None:
            return ''
        stream.input_finished()
        while r.is_ready(stream):
            r.decode_stream(stream)
        return (r.get_result(stream) or '').strip()


# ----------- VAD (separate sherpa-onnx instance — only used in hands-free) -----------

class VadDetector:
    """Wraps sherpa_onnx.VoiceActivityDetector. We only care about *speech start*
    here — sherpa's recognizer endpoint detector handles end-of-utterance, and
    that's a different signal (silence after frames decoded, not raw audio
    energy). Speech-start is the trigger for TTS barge-in.

    Loaded on demand the first time hands-free turns on, to avoid paying the
    ~30 MB silero_vad.onnx download cost on every cold start when the user only
    uses PTT."""

    def __init__(self, threshold: float = 0.5, min_silence_s: float = 0.5,
                 min_speech_s: float = 0.1):
        self._threshold = threshold
        self._min_silence_s = min_silence_s
        self._min_speech_s = min_speech_s
        self._vad = None
        self._lock = threading.Lock()
        self._prev_speech = False

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._vad is not None:
                return
            print('[vad] loading silero VAD…')
            import sherpa_onnx
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = _ensure_silero_vad()
            cfg.silero_vad.threshold = self._threshold
            cfg.silero_vad.min_silence_duration = self._min_silence_s
            cfg.silero_vad.min_speech_duration = self._min_speech_s
            cfg.silero_vad.window_size = VAD_WINDOW_SAMPLES
            cfg.sample_rate = STT_TARGET_SR
            self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
            print('[vad] ready')

    def feed_and_check_speech_start(self, samples) -> bool:
        """Returns True iff this chunk pushed the detector from silence → speech.
        The caller uses the False→True edge to trigger barge-in; we don't care
        about False→True→False flicker within one chunk."""
        with self._lock:
            v = self._vad
        if v is None:
            return False
        v.accept_waveform(samples)
        # Drain any completed segments so the internal buffer doesn't grow forever.
        # We don't consume the segments — the recognizer reads the same audio via
        # its own stream.
        while not v.empty():
            v.pop()
        speech = bool(v.is_speech_detected())
        edge = speech and not self._prev_speech
        self._prev_speech = speech
        return edge

    def reset(self) -> None:
        with self._lock:
            self._prev_speech = False
            if self._vad is not None:
                self._vad.reset()


# ----------- mic capture -----------

class MicRecorder:
    """sounddevice InputStream wrapper. Streams 100 ms float32 mono blocks at
    16 kHz to `on_chunk(samples)`. The callback runs on PortAudio's audio thread —
    keep it lightweight (no decode work, no envelope sends)."""

    def __init__(self, on_chunk: Callable[[Any], None]):
        self._on_chunk = on_chunk
        self._stream = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd

        def callback(indata, frames, time_info, status):
            if status:
                print(f'[mic] stream status: {status}', file=sys.stderr)
            # indata shape is (frames, 1); copy because PortAudio reuses the buffer.
            mono = indata[:, 0].copy()
            try:
                self._on_chunk(mono)
            except Exception as e:
                print(f'[mic] on_chunk failed: {e}', file=sys.stderr)

        self._stream = sd.InputStream(
            samplerate=STT_TARGET_SR, channels=1, dtype='float32',
            callback=callback, blocksize=STT_BLOCK_SAMPLES,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try: stream.stop()
        except Exception as e: print(f'[mic] stop failed: {e}', file=sys.stderr)
        try: stream.close()
        except Exception as e: print(f'[mic] close failed: {e}', file=sys.stderr)


# ----------- VoiceController: the public facade Bridge uses -----------

class VoiceController:
    """Owns the streaming engine, mic, VAD, and the PTT/hands-free state machine.
    Bridge delegates anything voice-shaped here.

    Outputs go through two callables injected at construction:
      * `push_envelope(env)` — to the renderer (Chat.Voice* envelopes)
      * `send_envelope(env)` — to Unity via the WS bridge (Chat.SubmitUserMessage
        for auto-submit, Tts.Cancel for barge-in)
    Keeping these as injections instead of importing keeps the module decoupled
    from app.py.

    `tts_hooks` is a pair `(is_active, cancel)` — `is_active()` returns True iff
    TTS playback is in flight on Unity, and `cancel(reason)` aborts whatever is
    currently being synthesized. Both can be no-ops on machines where TTS isn't
    wired up; barge-in just becomes a no-op."""

    def __init__(
        self,
        push_envelope: Callable[[dict], None],
        send_envelope: Callable[[dict], None],
        model_name: str,
        kind_override: str | None = None,
        num_threads: int = 1,
        provider: str = 'cpu',
        endpoint_silence_ms: int = 1200,
        hands_free_default: bool = False,
        auto_submit_default: bool = True,
        wake_word: str = '',
        wake_word_enabled_default: bool = False,
        wake_word_arm_seconds: float = 15.0,
        tts_echo_suppression: bool = True,
        barge_in_enabled: bool = True,
        tts_hooks: tuple[Callable[[], bool], Callable[[], bool], Callable[[str], None]] | None = None,
    ):
        self._push = push_envelope
        self._send = send_envelope
        # tts_hooks: (is_active, is_playing, cancel).
        #   is_active:  synth is currently generating chunks — for cancel logic.
        #   is_playing: audio sent to Unity is still being heard (true until
        #               Unity emits Tts.PlaybackEnded). Used for echo gating.
        #   cancel:     abort the in-flight synth.
        self._tts_is_active, self._tts_is_playing, self._tts_cancel = (
            tts_hooks or ((lambda: False), (lambda: False), (lambda _r: None))
        )

        # sherpa-onnx's endpointer fires when ANY of three rules matches:
        #   rule1: trailing_silence ≥ rule1_min_trailing_silence (must_contain_nonsilence=false)
        #          → fires even without recognized speech. If we leave this at the
        #            sherpa-onnx example default of 2.4 s, it short-circuits rule2:
        #            the user's "wait 10 s after speech" gets cut off after 2.4 s.
        #   rule2: trailing_silence ≥ rule2_min_trailing_silence (must_contain_nonsilence=true)
        #          → fires only after speech is decoded. This is the rule we want
        #            to drive from the user's endpointSilenceMs setting.
        #   rule3: utterance_length ≥ rule3_min_utterance_length (default 20 s)
        #          → runaway-utterance cutoff. Independent of silence.
        # So both rule1 AND rule2 need to use the same silence threshold;
        # otherwise the smaller value wins and the user's setting does nothing.
        silence_s = max(0.2, endpoint_silence_ms / 1000.0)
        self._engine = SttEngine(
            model_name=model_name, kind_override=kind_override,
            num_threads=num_threads, provider=provider,
            rule1_silence_s=silence_s,
            rule2_silence_s=silence_s,
            on_state_change=self._on_engine_state_change,
        )
        self._vad = VadDetector()

        # State machine — exactly one of these modes is active at a time.
        self._mode: str = 'idle'  # 'idle' | 'ptt' | 'handsfree'
        self._stream = None  # OnlineStream for the current utterance (PTT) or session (handsfree)
        self._mic = MicRecorder(on_chunk=self._on_mic_chunk)
        self._chunk_q: queue.Queue = queue.Queue()
        self._last_partial = ''
        self._busy = True
        self._reason: str | None = 'loading'
        self._lock = threading.Lock()

        # User-controllable toggles. Persisted only in-memory; the renderer
        # re-echoes them on every Chat.Init via patch_unity_envelope.
        self._hands_free_default = bool(hands_free_default)
        self._auto_submit = bool(auto_submit_default)
        self._hands_free_requested = self._hands_free_default

        # Wake-word gating. When `_wake_word_enabled` is true and a wake-word
        # phrase is set, auto-submit only fires for utterances that arrived
        # *after* the phrase was heard (within `_wake_word_arm_seconds`). Match
        # is a case-insensitive substring over the normalized partial transcript
        # — Nemotron emits punctuation, so we strip non-word chars before
        # comparing. `_wake_word_armed_until` is a monotonic timestamp; 0 means
        # disarmed. `_wake_word_pattern` is the compiled regex used to *strip*
        # the wake phrase from the submitted text (single-utterance flow).
        self._wake_word_phrase = (wake_word or '').strip()
        self._wake_word_enabled = bool(wake_word_enabled_default) and bool(self._wake_word_phrase)
        self._wake_word_arm_seconds = max(1.0, float(wake_word_arm_seconds))
        self._wake_word_armed_until = 0.0
        self._wake_word_pattern = self._compile_wake_pattern(self._wake_word_phrase)

        # TTS echo suppression. While Unity is playing TTS audio, the AI's voice
        # bleeds from the speakers into the mic and the recognizer transcribes
        # it as the user. We mute the recognizer for the entire window — from
        # the first chunk we send, until Unity emits Tts.PlaybackEnded telling
        # us its StreamingAudioBuffer drained. The VAD keeps running while
        # muted so barge-in still fires; a VAD speech onset force-unmutes for
        # the rest of that utterance. Disable with `ttsEchoSuppression=false`
        # when there's no acoustic feedback path (headphones, push-to-talk on a
        # separate mic, etc.) — the entire mute logic becomes a no-op.
        self._tts_echo_suppression = bool(tts_echo_suppression)
        # Barge-in (user speaking interrupts TTS) is opt-in because the VAD can
        # false-trigger during natural inter-sentence pauses and on speaker
        # bleed, which latches the recognizer open for the rest of the reply.
        # When false, the VAD model is never loaded and the worker skips the
        # speech-onset check entirely.
        self._barge_in_enabled = bool(barge_in_enabled)
        self._tts_force_unmuted = False  # set by barge-in until next is_playing flips
        self._recognizer_muted = False   # log-edge tracking so we only print transitions

        # Worker thread runs forever — pulls audio off the queue and drives both
        # the recognizer and the VAD. Started here so PTT clicks don't pay any
        # thread-spawn latency.
        self._stopping = False
        threading.Thread(target=self._worker_loop, name='stt-worker', daemon=True).start()

    # ----- public surface -----

    def start_load(self) -> None:
        """Kick off the model load on a background thread. Safe to call multiple
        times; SttEngine load_async is idempotent across re-calls."""
        self._engine.load_async()

    def ready_or_errored(self) -> bool:
        """True once the STT model load has resolved — loaded successfully OR failed. A failure
        leaves `ready` False but sets `error`, so we treat that as resolved too (the chat-loading
        overlay must not block forever on a model that will never load). Used as the STT readiness
        gate for the chat-loading overlay."""
        return self._engine.ready or self._engine.error is not None

    def patch_unity_envelope(self, env: dict) -> Iterable[dict]:
        """Inject voice-supported flag into Chat.Init, then echo our local
        toggles so the renderer can paint the right state on first paint."""
        if env.get('type') == TYPE_CHAT_INIT:
            p = env.get('payload') or {}
            p['voiceSupported'] = True
            env['payload'] = p
            yield env
            yield self._busy_envelope()
            yield self._hands_free_envelope()
            yield self._auto_submit_envelope()
            yield self._wake_word_envelope()
        else:
            yield env

    # ----- public command surface (called directly from ChatApi over the
    #       pywebview JS bridge — not via WS envelopes) -----

    def start_ptt(self) -> None:
        """Renderer pressed the mic button in push-to-talk mode."""
        self._handle_ptt_start()

    def stop_ptt(self) -> None:
        """Renderer released the mic button in push-to-talk mode."""
        self._handle_ptt_stop()

    def set_hands_free(self, enabled: bool) -> None:
        """Renderer toggled hands-free mode."""
        self._set_hands_free(bool(enabled))

    def set_auto_submit(self, enabled: bool) -> None:
        """Renderer toggled auto-submit (only meaningful in hands-free)."""
        self._set_auto_submit(bool(enabled))

    def set_wake_word(self, enabled: bool, phrase: str | None = None) -> None:
        """Renderer toggled wake-word mode (optionally changing the phrase)."""
        self._set_wake_word(bool(enabled), phrase if isinstance(phrase, str) else None)

    def clear_in_flight_utterance(self) -> None:
        """Reset whatever the streaming recognizer has accumulated for the current
        utterance so it doesn't endpoint-fire seconds later and submit text the
        user has visibly cancelled. Also clears the wake-word arm, since "cancel"
        implies "drop everything in-flight". Safe to call any time; no-op if the
        recognizer isn't currently holding a stream."""
        stream = self._stream
        if stream is not None:
            try:
                self._engine.reset(stream)
            except Exception as e:
                print(f'[stt] clear-utterance reset failed: {e}', file=sys.stderr)
        if self._last_partial:
            self._last_partial = ''
            self._push(self._partial_envelope(''))
        if self._wake_word_armed_until > 0.0:
            self._wake_word_armed_until = 0.0
            self._push(self._wake_word_armed_envelope(False))

    # ----- renderer-bound event builders (NOT WS envelopes — these ride the
    #       pywebview JS bridge via Bridge._push_or_buffer → window.__chatPush
    #       and surface in the React side via window.app.onChatEvent).
    #       Type strings are renderer-internal; their canonical declarations are
    #       in src/renderer/appEvents.ts. -----

    def _busy_envelope(self) -> dict:
        payload: dict = {'busy': self._busy}
        if self._busy and self._reason:
            payload['reason'] = self._reason
        return {'id': _new_id(), 'type': EVT_VOICE_BUSY, 'payload': payload}

    def _recording_envelope(self, active: bool) -> dict:
        return {'id': _new_id(), 'type': EVT_VOICE_RECORDING, 'payload': {'active': active}}

    def _transcript_envelope(self, text: str, submitted: bool = False) -> dict:
        # `submitted=true` tells the renderer this transcript was already sent to
        # Unity as a Chat.SubmitUserMessage — don't append it to the draft.
        return {
            'id': _new_id(), 'type': EVT_VOICE_TRANSCRIPT,
            'payload': {'text': text or '', 'submitted': submitted},
        }

    def _partial_envelope(self, text: str) -> dict:
        return {'id': _new_id(), 'type': EVT_VOICE_PARTIAL, 'payload': {'text': text or ''}}

    def _hands_free_envelope(self) -> dict:
        return {
            'id': _new_id(), 'type': EVT_HANDS_FREE_CHANGED,
            'payload': {'enabled': self._mode == 'handsfree' or self._hands_free_requested},
        }

    def _auto_submit_envelope(self) -> dict:
        return {
            'id': _new_id(), 'type': EVT_AUTO_SUBMIT_CHANGED,
            'payload': {'enabled': self._auto_submit},
        }

    def _wake_word_envelope(self) -> dict:
        return {
            'id': _new_id(), 'type': EVT_WAKE_WORD_CHANGED,
            'payload': {
                'enabled': self._wake_word_enabled,
                'phrase': self._wake_word_phrase,
                'armSeconds': self._wake_word_arm_seconds,
            },
        }

    def _wake_word_armed_envelope(self, armed: bool, expires_in_s: float = 0.0) -> dict:
        # Edge-only pulse so the renderer can animate a "listening for command"
        # indicator. `expiresInSeconds` lets the UI show a countdown or fade.
        return {
            'id': _new_id(), 'type': EVT_WAKE_WORD_ARMED,
            'payload': {'armed': armed, 'expiresInSeconds': max(0.0, expires_in_s)},
        }

    @staticmethod
    def _compile_wake_pattern(phrase: str):
        """Build a case-insensitive regex that matches the wake phrase even if
        the recognizer inserted punctuation or extra whitespace between words.
        Returns None if `phrase` is empty."""
        phrase = (phrase or '').strip()
        if not phrase:
            return None
        # Split on whitespace and rejoin with `[\W_]+` so "hey assistant" matches
        # "hey, assistant" / "Hey  assistant!" / "hey-assistant" — Nemotron emits
        # punct+caps so naive substring matching misses common cases.
        parts = [re.escape(p) for p in phrase.split() if p]
        if not parts:
            return None
        pattern = r'\b' + r'[\W_]+'.join(parts) + r'\b'
        return re.compile(pattern, re.IGNORECASE)

    # ----- state plumbing -----

    def _set_busy(self, busy: bool, reason: str | None = None) -> None:
        with self._lock:
            if self._busy == busy and self._reason == reason:
                return
            self._busy = busy
            self._reason = reason if busy else None
        self._push(self._busy_envelope())

    def _on_engine_state_change(self) -> None:
        if self._engine.ready:
            self._set_busy(False)
            # If hands-free was requested before the model finished loading (e.g.,
            # set via config), flip it on now.
            if self._hands_free_requested and self._mode == 'idle':
                self._enter_hands_free()
        else:
            err = self._engine.error or 'unknown'
            print(f'[stt] not ready, mic stays disabled: {err}', file=sys.stderr)

    def _set_hands_free(self, enabled: bool) -> None:
        self._hands_free_requested = enabled
        if enabled:
            if self._engine.ready and self._mode != 'handsfree':
                # If a PTT recording is in flight, finalize it first so we don't
                # drop the user's last words on the floor.
                if self._mode == 'ptt':
                    self._handle_ptt_stop()
                self._enter_hands_free()
        else:
            if self._mode == 'handsfree':
                self._exit_hands_free()
        self._push(self._hands_free_envelope())

    def _set_auto_submit(self, enabled: bool) -> None:
        self._auto_submit = enabled
        self._push(self._auto_submit_envelope())

    def _set_wake_word(self, enabled: bool, phrase: str | None = None) -> None:
        """Toggle wake-word gating and optionally update the phrase. Disarms
        immediately if turning off."""
        if phrase is not None:
            self._wake_word_phrase = phrase.strip()
            self._wake_word_pattern = self._compile_wake_pattern(self._wake_word_phrase)
        # Enabling with no phrase is meaningless — silently coerce to off.
        self._wake_word_enabled = bool(enabled) and bool(self._wake_word_phrase) and self._wake_word_pattern is not None
        if not self._wake_word_enabled and self._wake_word_armed_until > 0.0:
            self._wake_word_armed_until = 0.0
            self._push(self._wake_word_armed_envelope(False))
        self._push(self._wake_word_envelope())

    def _wake_word_check(self, partial_text: str) -> None:
        """Called from the worker loop on each updated partial transcript while
        in hands-free mode. If the wake phrase shows up, arm the system and
        reset the stream so the wake phrase itself isn't carried into the
        eventual submitted command."""
        if not self._wake_word_enabled or not self._wake_word_pattern:
            return
        if self._tts_is_playing():
            # Avoid self-trigger when TTS playback bleeds back into the mic and
            # the recognizer transcribes our own assistant saying the wake phrase.
            return
        if not partial_text or not self._wake_word_pattern.search(partial_text):
            return
        now = time.monotonic()
        expires = now + self._wake_word_arm_seconds
        # Re-arming inside the existing window just bumps the expiry; otherwise
        # this is a fresh arm event.
        fresh_arm = self._wake_word_armed_until < now
        self._wake_word_armed_until = expires
        # Drop the wake phrase + everything before it from the current stream
        # by resetting — the next utterance the recognizer emits is the command.
        if self._stream is not None:
            try:
                self._engine.reset(self._stream)
            except Exception as e:
                print(f'[stt] wake-word reset failed: {e}', file=sys.stderr)
        self._last_partial = ''
        self._push(self._partial_envelope(''))
        if fresh_arm:
            print(f'[stt] wake-word armed for {self._wake_word_arm_seconds:.0f}s')
            self._push(self._wake_word_armed_envelope(True, self._wake_word_arm_seconds))

    # ----- PTT path -----

    def _handle_ptt_start(self) -> None:
        if not self._engine.ready:
            print('[stt] PTT start ignored: model not ready', file=sys.stderr)
            return
        if self._mode != 'idle':
            return
        # User explicitly clicked mic — treat as intent to take the floor.
        # Cancel any in-flight TTS so the recording isn't polluted with the AI's
        # voice, and latch the echo mute open so PTT audio is captured from
        # the first chunk (even if Unity hasn't acknowledged PlaybackEnded).
        if self._tts_is_playing():
            self._on_barge_in()
        self._tts_force_unmuted = True
        try:
            self._stream = self._engine.create_stream()
            self._last_partial = ''
            self._mode = 'ptt'
            self._mic.start()
        except Exception as e:
            print(f'[mic] PTT start failed: {e}', file=sys.stderr)
            self._mode = 'idle'
            self._stream = None
            return
        self._push(self._recording_envelope(True))

    def _handle_ptt_stop(self) -> None:
        if self._mode != 'ptt':
            return
        self._mic.stop()
        self._push(self._recording_envelope(False))
        stream = self._stream
        self._mode = 'idle'
        self._stream = None
        # Drain the queue so any in-flight chunks land before we finalize.
        self._drain_queue_into(stream)
        try:
            final = self._engine.finalize(stream)
        except Exception as e:
            print(f'[stt] finalize failed: {e}', file=sys.stderr)
            final = ''
        self._last_partial = ''
        if final:
            self._push(self._transcript_envelope(final, submitted=False))

    # ----- hands-free path -----

    def _enter_hands_free(self) -> None:
        if self._mode == 'handsfree':
            return
        # Only spin up the VAD model when barge-in is actually enabled. Skipping
        # this saves the silero_vad.onnx download + load on machines that don't
        # use barge-in (and keeps the worker loop cheap because it'll never feed
        # the VAD).
        if self._barge_in_enabled:
            try:
                self._vad.ensure_loaded()
            except Exception as e:
                print(f'[vad] load failed, barge-in disabled: {e}', file=sys.stderr)
        try:
            self._stream = self._engine.create_stream()
            self._last_partial = ''
            self._mode = 'handsfree'
            if self._barge_in_enabled:
                self._vad.reset()
            self._mic.start()
        except Exception as e:
            print(f'[mic] hands-free start failed: {e}', file=sys.stderr)
            self._mode = 'idle'
            self._stream = None
            return
        self._push(self._recording_envelope(True))

    def _exit_hands_free(self) -> None:
        if self._mode != 'handsfree':
            return
        self._mic.stop()
        self._push(self._recording_envelope(False))
        self._mode = 'idle'
        self._stream = None
        self._last_partial = ''
        self._vad.reset()
        # Discard whatever's still in the queue — the user explicitly turned it off.
        try:
            while True:
                self._chunk_q.get_nowait()
        except queue.Empty:
            pass

    # ----- audio thread → worker thread → state machine -----

    def _on_mic_chunk(self, samples) -> None:
        # Runs on PortAudio's audio thread. Keep this O(1): no decode, no envelope.
        self._chunk_q.put(samples)

    def _drain_queue_into(self, stream) -> None:
        """Pull any pending mic chunks into `stream` before finalizing. Avoids
        racing the worker thread when PTT stops with audio still buffered."""
        if stream is None:
            return
        try:
            while True:
                chunk = self._chunk_q.get_nowait()
                self._engine.feed(stream, chunk)
        except queue.Empty:
            pass

    def _worker_loop(self) -> None:
        while not self._stopping:
            try:
                samples = self._chunk_q.get(timeout=0.25)
            except queue.Empty:
                continue
            mode = self._mode
            stream = self._stream
            if stream is None or mode == 'idle':
                continue
            try:
                # TTS-echo mute. While Unity is playing audio we sent, don't
                # feed the recognizer — otherwise the mic picks up the speakers
                # and we transcribe the AI's own voice as the user. The mute
                # window is the exact interval [first chunk sent .. Unity emits
                # Tts.PlaybackEnded]; no guessing. Skipped entirely when the
                # user opted out via `ttsEchoSuppression=false` (headphones).
                if self._tts_echo_suppression:
                    tts_playing = self._tts_is_playing()
                    if not tts_playing and self._tts_force_unmuted:
                        # Playback is genuinely done — reset the barge-in latch
                        # so the next TTS reply mutes again.
                        self._tts_force_unmuted = False
                    muted = tts_playing and not self._tts_force_unmuted
                else:
                    muted = False

                # VAD only runs in hands-free AND only when barge-in is enabled.
                # PTT doesn't need it (the user explicitly held the mic); when
                # barge-in is disabled we skip the VAD entirely to avoid
                # false-trigger unmutes during natural inter-sentence pauses.
                if mode == 'handsfree' and self._barge_in_enabled:
                    try:
                        if self._vad.feed_and_check_speech_start(samples):
                            self._on_barge_in()
                            # User is taking the floor — latch open so the first
                            # words of their utterance reach the recognizer
                            # even if Unity hasn't acknowledged PlaybackEnded
                            # yet.
                            self._tts_force_unmuted = True
                            muted = False
                    except Exception as e:
                        print(f'[vad] feed failed: {e}', file=sys.stderr)

                if muted:
                    if not self._recognizer_muted:
                        self._recognizer_muted = True
                        print('[stt] muted (TTS playing)')
                    continue
                if self._recognizer_muted:
                    self._recognizer_muted = False
                    print('[stt] unmuted')
                self._engine.feed(stream, samples)
                # Partial transcript — only emit on changes. The recognizer happily
                # returns the same text 10 times in a row otherwise.
                current = self._engine.get_text(stream)
                if current != self._last_partial:
                    self._last_partial = current
                    self._push(self._partial_envelope(current))
                    # Check for wake-word match on every partial change in
                    # hands-free mode. Doing it here (not at endpoint) is what
                    # gives the "okay google" feel: the system arms within
                    # ~100-300 ms of the user finishing the wake phrase, even
                    # though the endpoint silence rule is multi-second. Note
                    # this can re-enter _worker_loop's `stream`/`_last_partial`
                    # via `_engine.reset()` — that's fine, we re-read them next
                    # iteration.
                    if mode == 'handsfree':
                        self._wake_word_check(current)
                # Endpoint detection — drives hands-free utterance segmentation.
                # In PTT we ignore endpoint (the user controls start/stop), with
                # one exception: if we wanted to expose VAD auto-stop for PTT we'd
                # check here. For now PTT is strictly user-controlled.
                if mode == 'handsfree' and self._engine.is_endpoint(stream):
                    final = current
                    print(f'[stt] endpoint fired (rule2 silence_s={self._engine._rule2:.1f}) text={final!r}')
                    self._engine.reset(stream)
                    self._last_partial = ''
                    self._emit_final_hands_free(final)
            except Exception as e:
                print(f'[stt] worker error: {e}', file=sys.stderr)

    def _on_barge_in(self) -> None:
        """User started speaking while TTS is playing. Cancel the in-flight synth
        and tell Unity to drain its playback buffer."""
        if not self._tts_is_playing():
            return
        print('[stt] barge-in detected, cancelling TTS')
        try:
            self._tts_cancel('barge-in')
        except Exception as e:
            print(f'[stt] tts cancel hook failed: {e}', file=sys.stderr)
        self._send({'id': _new_id(), 'type': TYPE_TTS_CANCEL, 'payload': {'reason': 'barge-in'}})

    def _emit_final_hands_free(self, text: str) -> None:
        text = (text or '').strip()
        if not text:
            return
        # Wake-word gate: when armed, allow the auto-submit (and disarm); when
        # disabled, fall through. When enabled-but-not-armed, drop the utterance
        # silently — that's the whole point of the wake phrase.
        if self._wake_word_enabled:
            now = time.monotonic()
            armed = now < self._wake_word_armed_until
            if not armed:
                print(f'[stt] dropped (wake-word required, not armed): {text!r}')
                return
            # Strip a trailing wake-phrase echo in case the recognizer carried
            # it past the reset (small streaming context windows can do that).
            if self._wake_word_pattern is not None:
                text = self._wake_word_pattern.sub('', text).strip(' ,.!?-')
            if not text:
                # Wake phrase only, no command after it. Stay armed and wait
                # for the actual command on a future endpoint.
                return
            # Consume the arm — one submission per wake event.
            self._wake_word_armed_until = 0.0
            self._push(self._wake_word_armed_envelope(False))
        if self._auto_submit:
            # Send straight to Unity as if the user had clicked Send. The renderer
            # gets a `submitted=true` echo so it can clear the live caption /
            # avoid double-appending to the draft.
            self._send({
                'id': _new_id(), 'type': TYPE_CHAT_SUBMIT_USER_MESSAGE,
                'payload': {'text': text},
            })
            self._push(self._transcript_envelope(text, submitted=True))
        else:
            self._push(self._transcript_envelope(text, submitted=False))


# ----------- factory -----------

def make_voice_controller(
    config: dict,
    push_envelope: Callable[[dict], None],
    send_envelope: Callable[[dict], None],
    tts_hooks: tuple[Callable[[], bool], Callable[[str], None]] | None = None,
) -> VoiceController:
    """Build a VoiceController from a parsed app.config.json dict. Env vars
    override matching fields per-run.

      * `push_envelope(env)` — sends to the renderer (chat window).
      * `send_envelope(env)` — sends to Unity via the WS bridge.
      * `tts_hooks` — `(is_active, cancel)` pair from TtsController for barge-in.
        Pass None when TTS isn't wired (barge-in becomes a no-op).
    """
    stt_cfg = (config.get('stt') if isinstance(config, dict) else None) or {}
    model = str(_env_or('APP_STT_MODEL', stt_cfg.get('model', 'nemotron-0.6b')))
    kind_override = _env_or('APP_STT_KIND', stt_cfg.get('kind')) or None
    num_threads = int(_env_or('APP_STT_NUM_THREADS', stt_cfg.get('numThreads', 1)))
    provider = str(_env_or('APP_STT_PROVIDER', stt_cfg.get('provider', 'cpu')))
    endpoint_silence_ms = int(_env_or('APP_STT_ENDPOINT_SILENCE_MS',
                                       stt_cfg.get('endpointSilenceMs', stt_cfg.get('vadSilenceMs', 1200))))
    hands_free = _coerce_bool(_env_or('APP_HANDS_FREE', stt_cfg.get('handsFreeMode', False)))
    auto_submit = _coerce_bool(_env_or('APP_AUTO_SUBMIT', stt_cfg.get('autoSubmit', True)))
    wake_word = str(_env_or('APP_WAKE_WORD', stt_cfg.get('wakeWord', 'hey assistant')) or '')
    wake_enabled = _coerce_bool(_env_or('APP_WAKE_WORD_ENABLED', stt_cfg.get('wakeWordEnabled', False)))
    wake_arm_s = float(_env_or('APP_WAKE_WORD_ARM_SECONDS', stt_cfg.get('wakeWordArmSeconds', 15.0)))
    tts_echo_suppression = _coerce_bool(_env_or('APP_TTS_ECHO_SUPPRESSION', stt_cfg.get('ttsEchoSuppression', True)))
    barge_in_enabled = _coerce_bool(_env_or('APP_BARGE_IN_ENABLED', stt_cfg.get('bargeInEnabled', False)))
    return VoiceController(
        push_envelope=push_envelope,
        send_envelope=send_envelope,
        model_name=model,
        kind_override=kind_override,
        num_threads=num_threads,
        provider=provider,
        endpoint_silence_ms=endpoint_silence_ms,
        hands_free_default=hands_free,
        auto_submit_default=auto_submit,
        wake_word=wake_word,
        wake_word_enabled_default=wake_enabled,
        wake_word_arm_seconds=wake_arm_s,
        tts_echo_suppression=tts_echo_suppression,
        barge_in_enabled=barge_in_enabled,
        tts_hooks=tts_hooks,
    )
