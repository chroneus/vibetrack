"""File saving utilities for media artifacts.

All heavy dependencies (numpy, PIL) are imported lazily so this module
stays zero-dep at import time.  Files are saved under
``<log_dir>/media/<sanitized_tag>/<step>.<ext>``.  The returned path is
always **relative to log_dir** so the database stays portable across
machines.  The web server resolves them to absolute at serve time using
the experiment's ``log_dir``.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# ── helpers ────────────────────────────────────────────────────────


def _sanitize_tag(tag: str) -> str:
    """Make *tag* filesystem-safe (replace ``/`` with ``_``)."""
    return tag.replace("\x00", "").replace("/", "_").replace("\\", "_").replace(":", "_")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_filename(name: str, default: str) -> str:
    """Collapse a caller-supplied filename to a safe basename."""
    base = Path(name.replace("\\", "/")).name
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    safe = safe.strip("._")
    return safe or default


def _is_numpy_array(obj: Any) -> bool:
    """Check without importing numpy."""
    return type(obj).__module__ == "numpy" and type(obj).__name__ == "ndarray"


def _is_pil_image(obj: Any) -> bool:
    """Check without importing PIL."""
    mod = type(obj).__module__ or ""
    return mod.startswith("PIL") and "Image" in type(obj).__name__


def _is_torch_tensor(obj: Any) -> bool:
    """Check without importing torch."""
    return type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor"


# ── public API ─────────────────────────────────────────────────────


def save_image(
    data: Any,
    log_dir: str,
    tag: str,
    step: int,
) -> str:
    """Save an image and return the path relative to *log_dir*.

    *data* may be:

    * ``str`` / ``pathlib.Path`` — existing file, will be copied.
    * numpy ``ndarray`` — saved as PNG (requires ``Pillow``).
    * PIL ``Image`` — saved as PNG.
    """
    dest_dir = _ensure_dir(Path(log_dir) / "media" / _sanitize_tag(tag))

    if isinstance(data, (str, Path)):
        ext = Path(data).suffix or ".png"
        dest = dest_dir / f"{step}{ext}"
        shutil.copy2(str(data), str(dest))
    elif _is_pil_image(data):
        dest = dest_dir / f"{step}.png"
        data.save(str(dest))
    elif _is_torch_tensor(data):
        # Convert torch tensor to numpy and recurse
        arr = data.detach().cpu().numpy()
        # CHW → HWC for 3-channel images
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(2)
        return save_image(arr, log_dir, tag, step)
    elif _is_numpy_array(data):
        import numpy as np  # type: ignore[import-untyped]

        try:
            from PIL import Image as PILImage  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Saving numpy arrays as images requires Pillow: pip install Pillow"
            )
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        if data.dtype.kind == "f":
            data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        img = PILImage.fromarray(data)
        dest = dest_dir / f"{step}.png"
        img.save(str(dest))
    else:
        raise TypeError(f"Unsupported image type: {type(data)}")

    return str(dest.relative_to(Path(log_dir).resolve()))


def save_audio(
    data: Any,
    log_dir: str,
    tag: str,
    step: int,
    sample_rate: int = 44100,
) -> str:
    """Save audio and return the path relative to *log_dir*.

    *data* may be:

    * ``str`` / ``pathlib.Path`` — existing file, will be copied.
    * numpy ``ndarray`` — 1-D float array written as 16-bit PCM WAV
      using the stdlib ``wave`` module (zero extra deps).
    * ``bytes`` — written as raw ``.wav`` (caller is responsible for
      correct encoding).
    """
    dest_dir = _ensure_dir(Path(log_dir) / "media" / _sanitize_tag(tag))

    if isinstance(data, (str, Path)):
        ext = Path(data).suffix or ".wav"
        dest = dest_dir / f"{step}{ext}"
        shutil.copy2(str(data), str(dest))
    elif _is_numpy_array(data):
        import numpy as np  # type: ignore[import-untyped]

        dest = dest_dir / f"{step}.wav"
        # Normalize float → int16
        arr = data.flatten()
        if arr.dtype.kind == "f":
            arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
        elif arr.dtype != np.int16:
            arr = arr.astype(np.int16)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())
    elif isinstance(data, (bytes, bytearray)):
        dest = dest_dir / f"{step}.wav"
        dest.write_bytes(data)
    else:
        raise TypeError(f"Unsupported audio type: {type(data)}")

    return str(dest.relative_to(Path(log_dir).resolve()))


def save_video(
    data: Any,
    log_dir: str,
    tag: str,
    step: int,
) -> str:
    """Save video and return the path relative to *log_dir*.

    *data* may be:

    * ``str`` / ``pathlib.Path`` — existing file, will be copied.
    * ``bytes`` — written as ``.mp4``.
    """
    dest_dir = _ensure_dir(Path(log_dir) / "media" / _sanitize_tag(tag))

    if isinstance(data, (str, Path)):
        ext = Path(data).suffix or ".mp4"
        dest = dest_dir / f"{step}{ext}"
        shutil.copy2(str(data), str(dest))
    elif isinstance(data, (bytes, bytearray)):
        dest = dest_dir / f"{step}.mp4"
        dest.write_bytes(data)
    else:
        raise TypeError(f"Unsupported video type: {type(data)}")

    return str(dest.relative_to(Path(log_dir).resolve()))


def save_artifact(
    data: Any,
    log_dir: str,
    tag: str,
    step: int,
    filename: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Save a generic file artifact.

    Returns ``(relative_path, metadata_dict)`` where metadata includes
    *file_size*, *mime_type*, and *original_filename*.

    *data* may be:

    * ``str`` / ``pathlib.Path`` — existing file, will be copied.
    * ``bytes`` — written with *filename* or ``<step>.bin``.
    """
    dest_dir = _ensure_dir(Path(log_dir) / "media" / _sanitize_tag(tag))

    if isinstance(data, (str, Path)):
        src = Path(data)
        original_filename = src.name
        ext = src.suffix or ""
        if filename:
            fallback = f"{step}{ext}" if ext else f"{step}.bin"
            dest_name = _sanitize_filename(filename, fallback)
        elif ext:
            dest_name = f"{step}{ext}"
        else:
            dest_name = f"{step}_{_sanitize_filename(src.name, 'artifact')}"
        dest = dest_dir / dest_name
        shutil.copy2(str(src), str(dest))
    elif isinstance(data, (bytes, bytearray)):
        original_filename = filename or f"{step}.bin"
        dest_name = _sanitize_filename(original_filename, f"{step}.bin")
        dest = dest_dir / dest_name
        dest.write_bytes(data)
    else:
        raise TypeError(f"Unsupported artifact type: {type(data)}")

    mime_type, _ = mimetypes.guess_type(str(dest))
    metadata: Dict[str, Any] = {
        "file_size": dest.stat().st_size,
        "mime_type": mime_type or "application/octet-stream",
        "original_filename": original_filename,
    }
    return str(dest.relative_to(Path(log_dir).resolve())), metadata
