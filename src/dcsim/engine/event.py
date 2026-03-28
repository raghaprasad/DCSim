"""Event system: payloads, events, and the priority queue.

Events are ordered by (time, priority, sequence) for deterministic execution.
The EventQueue uses a heapq with lazy tombstone cancellation.
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from typing import Any

from dcsim.engine.clock import SimTime


@dataclass
class EventPayload:
    """Data carried by an event.

    Subclass or instantiate directly for different event types.
    """

    event_type: str
    parent_event_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.event_type}: {self.data}" if self.data else self.event_type


@dataclass(order=True)
class Event:
    """A scheduled simulation event.

    Ordering: (time, priority, sequence). Lower priority number = fires first.
    Sequence is auto-assigned by EventQueue for deterministic tie-breaking.
    """

    time: SimTime
    priority: int
    sequence: int
    event_id: str = field(compare=False)
    payload: EventPayload = field(compare=False)


class EventQueue:
    """heapq-backed priority queue with lazy cancellation."""

    __slots__ = ("_heap", "_counter", "_cancelled")

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._counter: int = 0
        self._cancelled: set[str] = set()

    def schedule(
        self,
        time: SimTime,
        payload: EventPayload,
        priority: int = 0,
    ) -> Event:
        event_id = uuid.uuid4().hex[:12]
        event = Event(
            time=time,
            priority=priority,
            sequence=self._counter,
            event_id=event_id,
            payload=payload,
        )
        self._counter += 1
        heapq.heappush(self._heap, event)
        return event

    def schedule_relative(
        self,
        current_time: SimTime,
        delta: SimTime,
        payload: EventPayload,
        priority: int = 0,
    ) -> Event:
        return self.schedule(current_time + delta, payload, priority)

    def cancel(self, event: Event) -> bool:
        """Lazily cancel an event. Returns True if not already cancelled."""
        if event.event_id in self._cancelled:
            return False
        self._cancelled.add(event.event_id)
        return True

    def pop(self) -> Event | None:
        """Pop the next non-cancelled event, or None if empty."""
        while self._heap:
            event = heapq.heappop(self._heap)
            if event.event_id not in self._cancelled:
                return event
            self._cancelled.discard(event.event_id)
        return None

    def peek(self) -> Event | None:
        """Peek at the next non-cancelled event without removing it."""
        while self._heap:
            if self._heap[0].event_id not in self._cancelled:
                return self._heap[0]
            event = heapq.heappop(self._heap)
            self._cancelled.discard(event.event_id)
        return None

    def is_empty(self) -> bool:
        return self.peek() is None

    def __len__(self) -> int:
        return len(self._heap) - len(self._cancelled)
