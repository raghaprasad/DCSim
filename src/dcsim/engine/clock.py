"""Simulation time model and clock.

Time is represented as integer microseconds since simulation start (t=0).
Integer representation avoids floating-point drift and ensures determinism.
"""

from __future__ import annotations

SimTime = int  # Microseconds since simulation start

# Time unit constants
MICROSECOND: SimTime = 1
MILLISECOND: SimTime = 1_000
SECOND: SimTime = 1_000_000


class SimulationClock:
    """Monotonically advancing simulation clock.

    The clock can only move forward. It is owned by the SimulationLoop
    and read by event handlers via SimulationContext.
    """

    __slots__ = ("_now",)

    def __init__(self) -> None:
        self._now: SimTime = 0

    def now(self) -> SimTime:
        return self._now

    def advance_to(self, t: SimTime) -> None:
        if t < self._now:
            raise ValueError(
                f"Cannot move clock backward: current={self._now}, requested={t}"
            )
        self._now = t

    def format_time(self, t: SimTime | None = None) -> str:
        """Human-readable time string."""
        if t is None:
            t = self._now
        if t < MILLISECOND:
            return f"{t}us"
        if t < SECOND:
            return f"{t / MILLISECOND:.3f}ms"
        return f"{t / SECOND:.6f}s"
