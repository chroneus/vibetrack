"""Media wrapper types for W&B-style log() dispatch.

Usage::

    import vibetrack

    vibetrack.log({
        "sample": vibetrack.Image(numpy_array),
        "audio":  vibetrack.Audio(waveform, sample_rate=16000),
        "clip":   vibetrack.Video("path/to/clip.mp4"),
        "model":  vibetrack.Artifact("checkpoint.pt"),
    })
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class Image:
    """Wrapper for logging images via ``log()``.

    *data* can be a file path (``str``), a numpy array, or a PIL Image.
    """

    __slots__ = ("data", "caption")

    def __init__(self, data: Any, caption: str = "") -> None:
        self.data = data
        self.caption = caption

    def __repr__(self) -> str:
        return f"Image(data={type(self.data).__name__}, caption={self.caption!r})"


class Audio:
    """Wrapper for logging audio via ``log()``.

    *data* can be a file path (``str``), a numpy array (raw PCM), or ``bytes``.
    """

    __slots__ = ("data", "sample_rate")

    def __init__(self, data: Any, sample_rate: int = 44100) -> None:
        self.data = data
        self.sample_rate = sample_rate

    def __repr__(self) -> str:
        return f"Audio(data={type(self.data).__name__}, sample_rate={self.sample_rate})"


class Video:
    """Wrapper for logging video via ``log()``.

    *data* can be a file path (``str``) or ``bytes``.
    """

    __slots__ = ("data", "fps")

    def __init__(self, data: Any, fps: int = 4) -> None:
        self.data = data
        self.fps = fps

    def __repr__(self) -> str:
        return f"Video(data={type(self.data).__name__}, fps={self.fps})"


class Artifact:
    """Wrapper for logging generic file artifacts via ``log()``.

    *path* should be a file path string.
    """

    __slots__ = ("path", "metadata")

    def __init__(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.path = path
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"Artifact(path={self.path!r})"
