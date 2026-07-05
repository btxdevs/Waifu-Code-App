"""Global capture of huggingface_hub model-download progress.

On first run the app downloads its STT bundle and pocket-TTS snapshot (hundreds of
MB) before chat is ready. huggingface_hub renders these as tqdm progress bars; we hand it
a `tqdm` subclass (via the `tqdm_class=` parameter its download functions already accept —
no monkeypatching) that mirrors each byte-counting bar into a shared, thread-safe state.
`snapshot()` returns the aggregate so the chat loading overlay can show a real progress
bar instead of an indefinite spinner.

Wiring:
  - stt.py  hf_hub_download(..., tqdm_class=download_progress.tqdm_class())
  - tts.py  snapshot_download(..., tqdm_class=download_progress.tqdm_class())
  - _session.py attaches snapshot() to each Chat.Loading envelope.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
# id(bar) -> (completed_bytes, total_bytes, desc). One entry per live byte-unit bar.
_bars: dict[int, tuple[float, float, str]] = {}
_cls = None  # cached tqdm subclass (built lazily once huggingface_hub is importable)


def _set(key: int, completed, total, desc) -> None:
    with _lock:
        _bars[key] = (float(completed or 0), float(total or 0), str(desc or ""))


def _drop(key: int) -> None:
    with _lock:
        _bars.pop(key, None)


def snapshot() -> dict | None:
    """Aggregate of all in-flight byte downloads, or None when nothing is downloading.

    Returns {active, completed, total, percent, label}. `percent` is 0..1; `total` may be
    0 briefly before a bar learns its size, in which case percent is 0 (indeterminate)."""
    with _lock:
        if not _bars:
            return None
        completed = sum(c for c, _t, _d in _bars.values())
        total = sum(t for _c, t, _d in _bars.values())
        # Use the description of the largest bar as the human label.
        label, biggest = "", -1.0
        for _c, t, d in _bars.values():
            if t > biggest:
                biggest, label = t, d
    percent = (completed / total) if total > 0 else 0.0
    return {
        "active": True,
        "completed": completed,
        "total": total,
        "percent": round(percent, 4),
        "label": label,
    }


def tqdm_class():
    """The reporting tqdm subclass to pass as `tqdm_class=` to hf download calls. Falls back
    to huggingface_hub's own tqdm if subclassing fails for any reason."""
    global _cls
    if _cls is not None:
        return _cls

    from huggingface_hub.utils import tqdm as _hf_tqdm  # hf's tqdm subclass (honors disable signals)

    class _ReportingTqdm(_hf_tqdm):  # type: ignore[misc, valid-type]
        # Only byte-unit bars are the model downloads; the "Fetching N files" count bar is ignored.
        def _report(self) -> None:
            if getattr(self, "unit", "") == "B":
                _set(id(self), getattr(self, "n", 0), getattr(self, "total", 0) or 0, getattr(self, "desc", ""))

        def update(self, n=1):
            r = super().update(n)
            self._report()
            return r

        def refresh(self, *a, **k):
            r = super().refresh(*a, **k)
            self._report()
            return r

        def close(self, *a, **k):
            try:
                return super().close(*a, **k)
            finally:
                _drop(id(self))

    _cls = _ReportingTqdm
    return _cls
