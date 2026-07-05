"""Image transcoding / resizing for the app.

Used by Unity's read_image tool. Pillow decodes the source (any format it supports:
PNG, JPG, GIF, WEBP, BMP, TIFF, ICO — plus HEIC/AVIF when the respective plugin
packages are installed), the controller downscales to the caller's pixel cap
preserving aspect ratio, then re-encodes as JPEG and ships the base64 bytes back.

Doing this in Pillow rather than Unity's ImageConversion gives us a single decode
implementation that covers every format the user is likely to drop into the
workspace, sidesteps Unity's lack of native WEBP support, and avoids the per-
provider PNG-alpha / animation-frame quirks that have bitten us before.

Wire format (mirrors C# AppProtocol):
  in:  {type:'Image.Process',       id, payload:{images:[{label, base64}], maxEdgePixels, jpegQuality}}
  out: {type:'Image.ProcessResult', replyTo:<id>, payload:{images:[{label, base64, width, height, error?}]}}
  err: {type:'Image.Error',         replyTo:<id>, payload:{message}}

Pillow is already in requirements.txt for the OCR path; no new deps.

Module name is `image_proc` (not `image`) so it doesn't shadow Pillow's PIL.Image submodule.
"""
from __future__ import annotations

import base64
import io
import sys
import threading
import traceback
from typing import Callable


TYPE_IMAGE_PROCESS = 'Image.Process'
TYPE_IMAGE_PROCESS_RESULT = 'Image.ProcessResult'
TYPE_IMAGE_ERROR = 'Image.Error'

IMAGE_TYPE_PREFIX = 'Image.'


class ImageController:
    """Receives Image.Process envelopes, runs the work on a per-request thread (Pillow
    decode + LANCZOS resize + JPEG encode), and replies with Image.ProcessResult."""

    def __init__(self, send_envelope: Callable[[dict], None]):
        self._send = send_envelope
        # Pillow's image-level operations are thread-safe (each Image instance is
        # independent), so we don't need to serialize like OCR does. Just guard against
        # the rare case where module-level state changes (e.g. a plugin registration)
        # races with a decode.
        self._import_lock = threading.Lock()
        self._pil_loaded = False

    # ----- public -----

    def is_image_envelope_type(self, type_: str) -> bool:
        return isinstance(type_, str) and type_.startswith(IMAGE_TYPE_PREFIX)

    def handle_envelope(self, env: dict) -> None:
        t = env.get('type')
        if t != TYPE_IMAGE_PROCESS:
            return
        threading.Thread(
            target=self._handle_process, args=(env,),
            name='image-process', daemon=True,
        ).start()

    # ----- internals -----

    def _ensure_pil(self):
        if self._pil_loaded:
            return
        with self._import_lock:
            if self._pil_loaded:
                return
            try:
                # Pillow is already pulled in by the OCR path. Touch it now to surface
                # missing-install errors with a clear message on first use.
                from PIL import Image as _PILImage  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    'Pillow is not installed. Run '
                    '`pip install Pillow` in CompanionApp/python/.venv.'
                ) from e
            # HEIC / AVIF support is optional: register the plugins if their packages are
            # importable. We don't add them to requirements.txt because most users won't
            # have HEIC files, but if pillow-heif is present, we light up that path.
            for plugin_name in ('pillow_heif', 'pillow_avif'):
                try:
                    __import__(plugin_name)
                except ImportError:
                    pass
            self._pil_loaded = True

    def _handle_process(self, env: dict) -> None:
        request_id = env.get('id')
        payload = env.get('payload') or {}
        images = payload.get('images')
        max_edge = int(payload.get('maxEdgePixels') or 2000)
        if max_edge < 64:
            max_edge = 64
        jpeg_quality = int(payload.get('jpegQuality') or 85)
        if jpeg_quality < 1:
            jpeg_quality = 1
        elif jpeg_quality > 100:
            jpeg_quality = 100

        if not isinstance(images, list) or not images:
            self._send_error(request_id, 'payload.images must be a non-empty array')
            return

        try:
            self._ensure_pil()
        except Exception as e:
            print(f'[image] PIL load failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            self._send_error(request_id, f'PIL load failed: {e}')
            return

        results: list[dict] = []
        for i, img in enumerate(images):
            label = f'image_{i + 1}'
            if not isinstance(img, dict):
                results.append({'label': label, 'base64': '', 'width': 0, 'height': 0,
                                'error': 'image entry is not an object'})
                continue
            label = str(img.get('label') or label)
            b64 = img.get('base64') or ''

            jpeg_b64, w, h, err = self._process_one(b64, max_edge, jpeg_quality)
            entry = {'label': label, 'base64': jpeg_b64, 'width': w, 'height': h}
            if err:
                entry['error'] = err
            results.append(entry)

        self._send({
            'id': _new_id(),
            'type': TYPE_IMAGE_PROCESS_RESULT,
            'replyTo': request_id,
            'payload': {'images': results},
        })

    def process_image_sync(self, raw: bytes, max_edge: int = 2000, quality: int = 85):
        """Direct sync entry point for in-process callers (the Python tool runner).
        Returns (jpeg_bytes, width, height, error_or_None). Caller deals with base64
        if it needs to ride the LLM wire as image_url."""
        try:
            self._ensure_pil()
        except Exception as e:
            return b'', 0, 0, f'PIL load failed: {e}'
        try:
            from PIL import Image
        except ImportError as e:
            return b'', 0, 0, f'PIL not available: {e}'
        try:
            with Image.open(io.BytesIO(raw)) as im:
                try: im.seek(0)
                except (AttributeError, EOFError): pass
                if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                    background = Image.new('RGB', im.size, (255, 255, 255))
                    rgba = im.convert('RGBA')
                    background.paste(rgba, mask=rgba.split()[3])
                    im = background
                elif im.mode != 'RGB':
                    im = im.convert('RGB')
                w, h = im.size
                if max(w, h) > max_edge:
                    scale = max_edge / float(max(w, h))
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    im = im.resize((new_w, new_h), Image.LANCZOS)
                    w, h = new_w, new_h
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=max(1, min(100, quality)), optimize=True)
                return buf.getvalue(), w, h, None
        except Exception as e:
            print(f'[image] decode/resize failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            return b'', 0, 0, f'image processing failed: {e}'

    def process_avatar_sync(self, raw: bytes, size: int = 250, quality: int = 90):
        """Profile-picture treatment: flatten transparency onto white, center-crop to a
        square, downscale to at most `size`×`size` (never upscale), JPEG-encode.
        Returns (jpeg_bytes, error_or_None)."""
        try:
            self._ensure_pil()
            from PIL import Image
        except Exception as e:
            return b'', f'PIL not available: {e}'
        try:
            with Image.open(io.BytesIO(raw)) as im:
                try: im.seek(0)
                except (AttributeError, EOFError): pass
                if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                    background = Image.new('RGB', im.size, (255, 255, 255))
                    rgba = im.convert('RGBA')
                    background.paste(rgba, mask=rgba.split()[3])
                    im = background
                elif im.mode != 'RGB':
                    im = im.convert('RGB')
                w, h = im.size
                edge = min(w, h)
                left = (w - edge) // 2
                top = (h - edge) // 2
                im = im.crop((left, top, left + edge, top + edge))
                if edge > size:
                    im = im.resize((size, size), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=max(1, min(100, quality)), optimize=True)
                return buf.getvalue(), None
        except Exception as e:
            print(f'[image] avatar processing failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            return b'', f'image processing failed: {e}'

    def _process_one(self, b64: str, max_edge: int, quality: int):
        if not isinstance(b64, str) or not b64:
            return '', 0, 0, 'empty base64 string'
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as e:
            return '', 0, 0, f'invalid base64: {e}'

        try:
            from PIL import Image
        except ImportError as e:
            return '', 0, 0, f'PIL not available: {e}'

        try:
            with Image.open(io.BytesIO(raw)) as im:
                # Animated formats (GIF / animated WEBP) — first frame only. Vision LLMs
                # treat the whole thing as a still anyway and base64'ing a multi-frame
                # animation would just bloat the payload.
                try:
                    im.seek(0)
                except (AttributeError, EOFError):
                    pass

                # JPEG doesn't carry alpha; composite onto white so transparent regions
                # don't go black. RGB conversion handles palette images and all other
                # exotic modes uniformly.
                if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                    background = Image.new('RGB', im.size, (255, 255, 255))
                    rgba = im.convert('RGBA')
                    background.paste(rgba, mask=rgba.split()[3])
                    im = background
                elif im.mode != 'RGB':
                    im = im.convert('RGB')

                w, h = im.size
                if max(w, h) > max_edge:
                    scale = max_edge / float(max(w, h))
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    im = im.resize((new_w, new_h), Image.LANCZOS)
                    w, h = new_w, new_h

                buf = io.BytesIO()
                # `optimize` tries a couple of Huffman tables and picks the smaller one —
                # ~5% savings for a few ms of CPU. Worth it for context-cost reduction.
                im.save(buf, format='JPEG', quality=quality, optimize=True)
                out_bytes = buf.getvalue()
        except Exception as e:
            print(f'[image] decode/resize failed: {e}\n{traceback.format_exc()}', file=sys.stderr)
            return '', 0, 0, f'image processing failed: {e}'

        return base64.b64encode(out_bytes).decode('ascii'), w, h, None

    def _send_error(self, reply_to, message: str) -> None:
        self._send({
            'id': _new_id(),
            'type': TYPE_IMAGE_ERROR,
            'replyTo': reply_to,
            'payload': {'message': message},
        })


def _new_id() -> str:
    import uuid
    return 'm_' + uuid.uuid4().hex


def make_image_controller(send_envelope: Callable[[dict], None]) -> ImageController:
    return ImageController(send_envelope=send_envelope)
