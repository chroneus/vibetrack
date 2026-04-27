"""LogEvent and EventHandle — shared types for per-event adapter dispatch.

Every ``SummaryWriter.add_*`` / ``log()`` call emits a :class:`LogEvent` and
returns an :class:`EventHandle`.  Registered dispatchers receive the event via
``adapter.send([event])``; callers can also forward a single event ad-hoc with
``handle.to("telegram")``.

Built-in kinds: ``"scalar"``, ``"text"``, ``"image"``, ``"audio"``, ``"video"``,
``"artifact"``, ``"histogram"``, ``"hparams"``.  ``kind`` is a plain ``str`` so
third-party adapters may introduce their own categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..writer import SummaryWriter


@dataclass
class LogEvent:
    kind: str
    tag: str
    step: Optional[int]
    value: Any
    walltime: float
    run_name: str
    project: Optional[str]
    extra: Dict[str, Any] = field(default_factory=dict)


class EventHandle:
    """Returned by every ``add_*`` / ``log()`` call.

    Supports ``.to(name, **creds)`` chaining for one-shot dispatch of this
    single event to an adapter, independently of any writer-level registrations.
    """

    __slots__ = ("_writer", "_event")

    def __init__(self, writer: "SummaryWriter", event: LogEvent) -> None:
        self._writer = writer
        self._event = event

    def to(self, name: str, **creds: Any) -> "EventHandle":
        self._writer._one_shot_send(self._event, name, **creds)
        return self


class NullEventHandle:
    """Stand-in returned by disabled writers (non-zero rank, closed, errored).

    ``.to(...)`` is a no-op so user code keeps working without branching.
    """

    __slots__ = ()

    def to(self, name: str, **creds: Any) -> "NullEventHandle":
        return self


class MultiEventHandle:
    """Aggregate handle returned by batch log calls (``add_scalars``, ``log``).

    Forwards ``.to(...)`` to every wrapped handle so chained one-shot sends
    fan out over all events emitted by the batch.
    """

    __slots__ = ("_handles",)

    def __init__(self, handles: list) -> None:
        self._handles = handles

    def to(self, name: str, **creds: Any) -> "MultiEventHandle":
        for h in self._handles:
            h.to(name, **creds)
        return self


_NULL_HANDLE = NullEventHandle()


def null_handle() -> NullEventHandle:
    return _NULL_HANDLE
