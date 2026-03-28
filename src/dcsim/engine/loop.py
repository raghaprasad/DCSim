"""Simulation loop: the core run/step engine and context."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from dcsim.engine.clock import SimTime, SimulationClock
from dcsim.engine.event import Event, EventPayload, EventQueue

EventHandler = Callable[["Event", "SimulationContext"], list[EventPayload] | None]


class SimulationContext:
    """Shared context passed to every event handler.

    Extended in later phases with hardware, scheduler, logger, metrics.
    No __slots__ — phases add attributes dynamically.
    """

    def __init__(self, clock: SimulationClock, queue: EventQueue) -> None:
        self.clock = clock
        self.queue = queue


@dataclass
class SimulationResult:
    """Summary returned after a simulation run."""

    events_processed: int = 0
    final_time: SimTime = 0
    stopped_by_max_time: bool = False
    stopped_by_empty_queue: bool = False


class SimulationLoop:
    """Discrete event simulation loop.

    Register handlers for event types, then call run() or step().
    """

    def __init__(self) -> None:
        self._clock = SimulationClock()
        self._queue = EventQueue()
        self._context = SimulationContext(self._clock, self._queue)
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._events_processed: int = 0
        self._running: bool = False

    @property
    def clock(self) -> SimulationClock:
        return self._clock

    @property
    def queue(self) -> EventQueue:
        return self._queue

    @property
    def context(self) -> SimulationContext:
        return self._context

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def schedule(
        self,
        time: SimTime,
        payload: EventPayload,
        priority: int = 0,
    ) -> Event:
        return self._queue.schedule(time, payload, priority)

    def step(self) -> bool:
        """Process a single event. Returns False if no event was available."""
        event = self._queue.pop()
        if event is None:
            return False
        self._clock.advance_to(event.time)
        self._dispatch(event)
        self._events_processed += 1
        return True

    def run(self, until: SimTime | None = None) -> SimulationResult:
        """Run the simulation until the queue is empty or until max time."""
        self._running = True
        result = SimulationResult()

        while self._running:
            next_event = self._queue.peek()
            if next_event is None:
                result.stopped_by_empty_queue = True
                break
            if until is not None and next_event.time > until:
                result.stopped_by_max_time = True
                break
            self.step()

        result.events_processed = self._events_processed
        result.final_time = self._clock.now()
        self._running = False
        return result

    def pause(self) -> None:
        self._running = False

    def _dispatch(self, event: Event) -> None:
        handlers = self._handlers.get(event.payload.event_type, [])
        for handler in handlers:
            new_payloads = handler(event, self._context)
            if new_payloads:
                for payload in new_payloads:
                    if payload.parent_event_id is None:
                        payload.parent_event_id = event.event_id
                    self._queue.schedule(
                        self._clock.now(),
                        payload,
                        priority=0,
                    )
