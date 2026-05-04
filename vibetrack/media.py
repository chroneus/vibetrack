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
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── helpers ────────────────────────────────────────────────────────


def _sanitize_tag(tag: str) -> str:
    """Make *tag* filesystem-safe (replace ``/`` with ``_``)."""
    return (
        tag.replace("\x00", "").replace("/", "_").replace("\\", "_").replace(":", "_")
    )


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


def _to_numpy_array(data: Any) -> Optional[Any]:
    """Return a numpy array for tensor/ndarray inputs without importing eagerly."""
    if _is_torch_tensor(data):
        return data.detach().cpu().numpy()
    if _is_numpy_array(data):
        return data
    return None


def _normalize_single_image_array(data: Any, dataformats: str) -> Any:
    """Move image arrays to HWC/HW layout accepted by Pillow."""
    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(data)
    fmt = (dataformats or "").upper()

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3 and len(fmt) == 3 and "H" in fmt and "W" in fmt:
        axes: List[int] = [fmt.index("H"), fmt.index("W")]
        if "C" in fmt:
            axes.append(fmt.index("C"))
        arr = np.transpose(arr, axes)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return arr


def _make_image_grid(images: List[Any]) -> Any:
    """Pack a batch of HWC/HW images into a near-square grid."""
    import numpy as np  # type: ignore[import-untyped]

    if not images:
        raise ValueError("image batch is empty")

    prepared = []
    max_h = 0
    max_w = 0
    max_c = 1
    for img in images:
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"expected image array with 2 or 3 dims, got {arr.shape}")
        prepared.append(arr)
        max_h = max(max_h, arr.shape[0])
        max_w = max(max_w, arr.shape[1])
        max_c = max(max_c, arr.shape[2])

    cols = int(ceil(sqrt(len(prepared))))
    rows = int(ceil(len(prepared) / cols))
    grid = np.zeros((rows * max_h, cols * max_w, max_c), dtype=prepared[0].dtype)
    for i, img in enumerate(prepared):
        r = i // cols
        c = i % cols
        if img.shape[2] == 1 and max_c > 1:
            img = np.repeat(img, max_c, axis=2)
        grid[
            r * max_h : r * max_h + img.shape[0],
            c * max_w : c * max_w + img.shape[1],
            : img.shape[2],
        ] = img

    if max_c == 1:
        return grid.squeeze(2)
    return grid


def _normalize_image_array(data: Any, dataformats: str) -> Any:
    """Normalize single or batched image arrays according to TensorBoard dataformats."""
    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(data)
    fmt = (dataformats or "").upper()
    if "N" in fmt and arr.ndim == len(fmt):
        n_axis = fmt.index("N")
        arr = np.moveaxis(arr, n_axis, 0)
        child_fmt = fmt.replace("N", "", 1)
        return _make_image_grid(
            [_normalize_single_image_array(frame, child_fmt) for frame in arr]
        )
    return _normalize_single_image_array(arr, fmt)


def _as_uint8_image(data: Any) -> Any:
    """Convert numeric image arrays to uint8 with TensorBoard-like [0, 1] floats."""
    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(data)
    if arr.dtype.kind == "b":
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype.kind == "f":
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _normalize_video_frames(data: Any) -> List[Any]:
    """Convert video tensor/array inputs to a list of HWC uint8 frames."""
    import numpy as np  # type: ignore[import-untyped]

    arr = np.asarray(data)
    if arr.ndim == 5:
        # TensorBoard documents N,T,C,H,W. Also accept N,T,H,W,C.
        if arr.shape[2] in (1, 3, 4):
            arr = np.moveaxis(arr, 2, -1)
        frames = []
        for t in range(arr.shape[1]):
            frames.append(
                _as_uint8_image(
                    _make_image_grid([arr[n, t] for n in range(arr.shape[0])])
                )
            )
        return frames
    if arr.ndim == 4:
        # Accept T,C,H,W or T,H,W,C.
        if arr.shape[1] in (1, 3, 4):
            arr = np.moveaxis(arr, 1, -1)
        return [_as_uint8_image(frame) for frame in arr]
    raise TypeError(f"Unsupported video array shape: {arr.shape}")


# ── public API ─────────────────────────────────────────────────────


def save_image(
    data: Any,
    log_dir: str,
    tag: str,
    step: int,
    dataformats: str = "",
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
    elif _to_numpy_array(data) is not None:
        try:
            from PIL import Image as PILImage  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Saving numpy arrays as images requires Pillow: pip install Pillow"
            )
        arr = _normalize_image_array(_to_numpy_array(data), dataformats)
        img = PILImage.fromarray(_as_uint8_image(arr))
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
    elif _to_numpy_array(data) is not None:
        import numpy as np  # type: ignore[import-untyped]

        dest = dest_dir / f"{step}.wav"
        # Normalize float → int16
        arr = _to_numpy_array(data).flatten()
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
    fps: Union[int, float] = 4,
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
    elif _to_numpy_array(data) is not None:
        try:
            try:
                from moviepy import ImageSequenceClip  # type: ignore[import-untyped]
            except ImportError:
                from moviepy.editor import ImageSequenceClip  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "Saving tensor/array videos requires moviepy: pip install moviepy"
            )
        frames = _normalize_video_frames(_to_numpy_array(data))
        dest = dest_dir / f"{step}.mp4"
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(dest), codec="libx264", audio=False, logger=None)
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
