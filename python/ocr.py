"""OCR for the app — runs RapidOCR on base64-encoded images and replies with
recognized text. Wired up by Bridge in app.py: any envelope whose type starts with
`Ocr.` is delegated here.

The read_image_ocr tool on the Unity side uses this when the configured LLM backend isn't
vision-capable; otherwise read_image forwards the image bytes directly to the model and this
path is bypassed entirely.

Wire format (mirrors C# AppProtocol):
  in:  {type:'Ocr.Extract', id, payload:{images:[{label, base64}]}}
  out: {type:'Ocr.ExtractResult', replyTo:<id>, payload:{images:[{label, text, error?}]}}
  err: {type:'Ocr.Error',         replyTo:<id>, payload:{message}}

Model load is deferred until the first Ocr.Extract arrives — the rapidocr package eager-
imports ONNX Runtime and three ~15 MB models, and we don't want to pay that on startup just
in case the user only ever uses the vision path.
"""
from __future__ import annotations

import base64
import io
import sys
import threading
import traceback
from typing import Callable


TYPE_OCR_EXTRACT = 'Ocr.Extract'
TYPE_OCR_EXTRACT_RESULT = 'Ocr.ExtractResult'
TYPE_OCR_ERROR = 'Ocr.Error'

OCR_TYPE_PREFIX = 'Ocr.'


class OcrController:
    """Owns the RapidOCR engine and serializes Ocr.Extract requests. Engine is lazy-loaded
    on the first request so startup stays cheap. Each request runs on its own worker thread
    so the WS reader thread isn't blocked by OCR (which can take 1–5s per image)."""

    def __init__(self, send_envelope: Callable[[dict], None]):
        self._send = send_envelope
        # Engine instance is created on first use. The rapidocr package is a heavy import
        # (ORT + three models) — keeping it out of module import time means the app
        # still starts fast for users who never hit the OCR path.
        self._engine = None
        self._engine_lock = threading.Lock()
        # Serialize OCR calls. RapidOCR's underlying ORT sessions aren't documented as
        # thread-safe across concurrent infer calls; rather than risk it, we serialize.
        # Individual requests already run off the WS thread on their own workers.
        self._infer_lock = threading.Lock()

    # ----- public -----

    def is_ocr_envelope_type(self, type_: str) -> bool:
        return isinstance(type_, str) and type_.startswith(OCR_TYPE_PREFIX)

    def handle_envelope(self, env: dict) -> None:
        t = env.get('type')
        if t != TYPE_OCR_EXTRACT:
            # All other Ocr.* types are outbound responses we generate ourselves — nothing
            # to do if one happens to land here (it shouldn't).
            return
        threading.Thread(
            target=self._handle_extract, args=(env,),
            name='ocr-extract', daemon=True,
        ).start()

    def start_load(self) -> None:
        """Optional warm-up; calls _ensure_engine() on a background thread so the first
        real request doesn't pay the load cost. Safe to call multiple times — _ensure_engine
        is idempotent."""
        threading.Thread(target=self._ensure_engine, name='ocr-load', daemon=True).start()

    # ----- internals -----

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        with self._engine_lock:
            if self._engine is not None:
                return self._engine
            print('[ocr] loading RapidOCR engine (first request — may take ~5s)')
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:
                raise RuntimeError(
                    'rapidocr-onnxruntime is not installed. Run '
                    '`pip install rapidocr-onnxruntime Pillow` in CompanionApp/python/.venv.'
                ) from e
            self._engine = RapidOCR()
            print('[ocr] RapidOCR engine ready')
            return self._engine

    def _handle_extract(self, env: dict) -> None:
        request_id = env.get('id')
        payload = env.get('payload') or {}
        images = payload.get('images')

        if not isinstance(images, list) or not images:
            self._send_error(request_id, "payload.images must be a non-empty array")
            return

        try:
            engine = self._ensure_engine()
        except Exception as e:
            print(f'[ocr] engine load failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            self._send_error(request_id, f'engine load failed: {e}')
            return

        results: list[dict] = []
        for i, img in enumerate(images):
            label = ''
            if isinstance(img, dict):
                label = str(img.get('label') or f'image_{i + 1}')
                b64 = img.get('base64') or ''
            else:
                results.append({'label': f'image_{i + 1}', 'text': '', 'error': 'image entry is not an object'})
                continue

            text, err = self._ocr_one(engine, b64)
            entry = {'label': label, 'text': text}
            if err:
                entry['error'] = err
            results.append(entry)

        self._send({
            'id': _new_id(),
            'type': TYPE_OCR_EXTRACT_RESULT,
            'replyTo': request_id,
            'payload': {'images': results},
        })

    def ocr_image_sync(self, raw: bytes) -> tuple[str, str | None]:
        """Direct sync entry point for in-process callers (the Python tool runner).
        Returns (recognized_text, error_or_None). Engine is loaded on first call
        and serialized via _infer_lock so this is safe to call from multiple
        threads / asyncio.to_thread."""
        try:
            engine = self._ensure_engine()
        except Exception as e:
            return '', f'OCR engine load failed: {e}'
        return self._ocr_one(engine, base64.b64encode(raw).decode('ascii'))

    def _ocr_one(self, engine, b64: str) -> tuple[str, str | None]:
        if not isinstance(b64, str) or not b64:
            return '', 'empty base64 string'

        # Pillow → numpy is the most reliable decode path across PNG/JPG/GIF/WEBP/BMP.
        # RapidOCR accepts numpy ndarrays directly.
        try:
            from PIL import Image
            import numpy as np
        except ImportError as e:
            return '', f'missing image-decode dependency: {e}'

        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as e:
            return '', f'invalid base64: {e}'

        try:
            with Image.open(io.BytesIO(raw)) as im:
                # Convert to RGB so RapidOCR sees a 3-channel array. RGBA / palette images
                # would otherwise pass through with 4 or 1 channels and confuse the ORT
                # input binding.
                im = im.convert('RGB')
                arr = np.array(im)
        except Exception as e:
            return '', f'could not decode image: {e}'

        try:
            with self._infer_lock:
                # RapidOCR returns (result, elapsed) where result is a list of
                # [bbox, text, score] triples — or None when nothing was recognized.
                result, _elapsed = engine(arr)
        except Exception as e:
            print(f'[ocr] inference failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            return '', f'OCR inference failed: {e}'

        if not result:
            return '', None

        lines: list[str] = []
        for item in result:
            try:
                # `[bbox, text, score]`. We only care about text.
                if len(item) >= 2 and isinstance(item[1], str):
                    s = item[1].strip()
                    if s:
                        lines.append(s)
            except Exception:
                continue
        return '\n'.join(lines), None

    def _send_error(self, reply_to: str | None, message: str) -> None:
        self._send({
            'id': _new_id(),
            'type': TYPE_OCR_ERROR,
            'replyTo': reply_to,
            'payload': {'message': message},
        })


def _new_id() -> str:
    import uuid
    return 'm_' + uuid.uuid4().hex


def make_ocr_controller(send_envelope: Callable[[dict], None]) -> OcrController:
    return OcrController(send_envelope=send_envelope)
