"""Workload base classes and WorkloadManager.

Provides the Workload ABC and WorkloadManager that wires workloads into the
simulation event loop. No scheduler — all GPUs are assigned directly at t=0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dcsim.engine.clock import SimTime
from dcsim.engine.event import Event, EventPayload, EventQueue
from dcsim.engine.loop import EventHandler, SimulationContext, SimulationLoop


class WorkloadPhase(Enum):
    COMPUTE = "compute"
    COMMUNICATE = "communicate"


@dataclass
class Workload(ABC):
    """Abstract base class for simulation workloads."""

    job_id: str
    gpu_ids: list[str]
    current_step: int = 0
    total_steps: int = 10
    state: str = "pending"  # "pending", "running", "interrupted", "completed"

    @abstractmethod
    def get_next_phase(self, gpu_states: dict[str, dict[str, Any]], now: SimTime) -> tuple[WorkloadPhase, SimTime] | None:
        """Returns (phase_type, duration_us) or None if done."""
        ...

    @abstractmethod
    def on_gpu_failed(self, gpu_id: str) -> str:
        """Returns 'abort' or 'continue'."""
        ...


class WorkloadManager:
    """Wires workloads into the event loop. No scheduler — directly assigns GPUs."""

    def __init__(self) -> None:
        self._sim: SimulationLoop | None = None
        self._workload: Workload | None = None
        self._pending_event: Event | None = None
        self._current_phase: WorkloadPhase | None = None
        self._phase_start_time: SimTime = 0
        self._phase_total_duration: SimTime = 0

        # Internal GPU state tracking (no hardware graph dependency)
        self._gpu_throttle_factors: dict[str, float] = {}
        self._gpu_failed: set[str] = set()

        # Link state tracking for comms blocking
        self._links_down: set[str] = set()
        self._comms_blocked: bool = False
        self._comms_elapsed_before_block: SimTime = 0
        self._reroute_penalty_us: SimTime = 10_000  # 10ms default

    def setup(self, sim: SimulationLoop, workload: Workload) -> None:
        """Initialize the workload manager and wire handlers into the event loop."""
        self._sim = sim
        self._workload = workload

        # Initialize GPU state tracking
        for gpu_id in workload.gpu_ids:
            self._gpu_throttle_factors[gpu_id] = 1.0

        # Register event handlers
        sim.register_handler("workload.phase.complete", self._handle_phase_complete)
        sim.register_handler("cascade.gpu.job_interrupted", self._handle_gpu_interrupted)
        sim.register_handler("cascade.gpu.throttled", self._handle_gpu_throttled)
        sim.register_handler("hardware.gpu.repair", self._handle_gpu_repair)
        sim.register_handler("hardware.gpu.fail", self._handle_gpu_fail)
        sim.register_handler("hardware.gpu.throttle", self._handle_gpu_throttle)
        sim.register_handler("hardware.gpu.unthrottle", self._handle_gpu_unthrottle)
        sim.register_handler("hardware.link.fail", self._handle_link_fail)
        sim.register_handler("hardware.link.repair", self._handle_link_repair)

        # Mark workload as running and schedule first phase at t=0
        workload.state = "running"

        # Attach self to context for external access
        sim.context.workload_manager = self

        self._schedule_next_phase(0)

    def _get_gpu_states(self) -> dict[str, dict[str, Any]]:
        """Build a gpu_states dict from internal tracking."""
        states: dict[str, dict[str, Any]] = {}
        for gpu_id in self._workload.gpu_ids:
            states[gpu_id] = {
                "throttle_factor": self._gpu_throttle_factors.get(gpu_id, 1.0),
                "failed": gpu_id in self._gpu_failed,
            }
        return states

    def _schedule_next_phase(self, now: SimTime) -> None:
        """Determine and schedule the next workload phase."""
        wl = self._workload
        if wl is None or wl.state != "running":
            return

        gpu_states = self._get_gpu_states()
        result = wl.get_next_phase(gpu_states, now)
        if result is None:
            # Workload is done
            wl.state = "completed"
            self._sim.queue.schedule(
                now,
                EventPayload(
                    event_type="workload.job.complete",
                    data={"job_id": wl.job_id, "total_steps": wl.total_steps},
                ),
                priority=30,
            )
            return

        phase, duration = result
        self._current_phase = phase
        self._phase_start_time = now
        self._phase_total_duration = duration

        # Emit phase start event
        self._sim.queue.schedule(
            now,
            EventPayload(
                event_type="workload.phase.start",
                data={
                    "job_id": wl.job_id,
                    "phase": phase.value,
                    "step": wl.current_step,
                    "duration_us": duration,
                },
            ),
            priority=20,
        )

        # Schedule phase completion
        self._pending_event = self._sim.queue.schedule(
            now + duration,
            EventPayload(
                event_type="workload.phase.complete",
                data={
                    "job_id": wl.job_id,
                    "phase": phase.value,
                    "step": wl.current_step,
                },
            ),
            priority=30,
        )

    def _handle_phase_complete(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle completion of a workload phase."""
        wl = self._workload
        if wl is None or wl.state != "running":
            return None

        data = event.payload.data
        if data.get("job_id") != wl.job_id:
            return None

        self._pending_event = None
        now = ctx.clock.now()
        phase_str = data.get("phase")

        if phase_str == WorkloadPhase.COMMUNICATE.value:
            # Communication phase complete — step is done
            wl.current_step += 1
            # Emit step complete
            self._sim.queue.schedule(
                now,
                EventPayload(
                    event_type="workload.step.complete",
                    data={
                        "job_id": wl.job_id,
                        "step": wl.current_step,
                    },
                ),
                priority=30,
            )

        # Schedule next phase
        self._schedule_next_phase(now)
        return None

    def _handle_gpu_fail(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle a hardware GPU failure event — update internal state and cascade."""
        wl = self._workload
        if wl is None:
            return None

        gpu_id = event.payload.data.get("gpu_id", event.payload.data.get("component_id", ""))
        if gpu_id not in self._gpu_throttle_factors:
            return None

        self._gpu_failed.add(gpu_id)

        # If the workload is running, emit cascade event
        if wl.state == "running":
            return [
                EventPayload(
                    event_type="cascade.gpu.job_interrupted",
                    data={"gpu_id": gpu_id, "job_id": wl.job_id},
                )
            ]
        return None

    def _handle_gpu_throttle(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle a hardware GPU throttle event — update internal state and cascade."""
        wl = self._workload
        if wl is None:
            return None

        gpu_id = event.payload.data.get("gpu_id", event.payload.data.get("component_id", ""))
        throttle_factor = event.payload.data.get("throttle_factor", 0.5)
        if gpu_id not in self._gpu_throttle_factors:
            return None

        self._gpu_throttle_factors[gpu_id] = throttle_factor

        if wl.state == "running":
            return [
                EventPayload(
                    event_type="cascade.gpu.throttled",
                    data={
                        "gpu_id": gpu_id,
                        "job_id": wl.job_id,
                        "throttle_factor": throttle_factor,
                    },
                )
            ]
        return None

    def _handle_gpu_unthrottle(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle GPU unthrottle — restore throttle factor."""
        wl = self._workload
        if wl is None:
            return None

        gpu_id = event.payload.data.get("gpu_id", event.payload.data.get("component_id", ""))
        if gpu_id in self._gpu_throttle_factors:
            self._gpu_throttle_factors[gpu_id] = 1.0
        return None

    def _handle_gpu_interrupted(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle cascade.gpu.job_interrupted — cancel pending work, abort workload."""
        wl = self._workload
        if wl is None or wl.state != "running":
            return None

        data = event.payload.data
        if data.get("job_id") != wl.job_id:
            return None

        # Cancel any pending phase completion
        if self._pending_event is not None:
            self._sim.queue.cancel(self._pending_event)
            self._pending_event = None

        wl.state = "interrupted"
        self._current_phase = None

        return [
            EventPayload(
                event_type="workload.job.failed",
                data={
                    "job_id": wl.job_id,
                    "last_step": wl.current_step,
                    "reason": "gpu_failure",
                },
            )
        ]

    def _handle_gpu_throttled(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle cascade.gpu.throttled — recalculate phase duration if in compute."""
        wl = self._workload
        if wl is None or wl.state != "running":
            return None

        data = event.payload.data
        if data.get("job_id") != wl.job_id:
            return None

        # Only recalculate during compute phase
        if self._current_phase != WorkloadPhase.COMPUTE:
            return None

        # Cancel current pending phase completion
        if self._pending_event is not None:
            self._sim.queue.cancel(self._pending_event)
            self._pending_event = None

        now = ctx.clock.now()

        # Recalculate: get new duration from workload using updated throttle factors
        gpu_states = self._get_gpu_states()
        result = wl.get_next_phase(gpu_states, now)
        if result is None:
            return None

        phase, new_full_duration = result

        # Calculate: how much time has elapsed in this phase?
        elapsed = now - self._phase_start_time
        # The remaining time should be based on the new full duration minus the elapsed time
        # But elapsed was at the OLD rate. We need to figure out the fraction completed.
        # fraction_done = elapsed / old_duration
        # remaining_at_new_rate = (1 - fraction_done) * new_full_duration
        if self._phase_total_duration > 0:
            fraction_done = elapsed / self._phase_total_duration
        else:
            fraction_done = 0.0

        remaining = int((1.0 - fraction_done) * new_full_duration)
        if remaining < 0:
            remaining = 0

        self._phase_total_duration = new_full_duration
        self._phase_start_time = now  # Reset for any future recalculation
        # Update the "old duration" to be remaining so future fraction calcs work
        self._phase_total_duration = remaining

        # Reschedule phase completion
        self._pending_event = self._sim.queue.schedule(
            now + remaining,
            EventPayload(
                event_type="workload.phase.complete",
                data={
                    "job_id": wl.job_id,
                    "phase": phase.value,
                    "step": wl.current_step,
                },
            ),
            priority=30,
        )
        return None

    def _handle_gpu_repair(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle GPU repair — resume workload if it was interrupted."""
        wl = self._workload
        if wl is None:
            return None

        gpu_id = event.payload.data.get("gpu_id", event.payload.data.get("component_id", ""))
        if gpu_id in self._gpu_failed:
            self._gpu_failed.discard(gpu_id)
            self._gpu_throttle_factors[gpu_id] = 1.0

        # If workload is interrupted and all GPUs healthy, resume
        if wl.state == "interrupted" and len(self._gpu_failed) == 0:
            wl.state = "running"
            now = ctx.clock.now()
            self._schedule_next_phase(now)

        return None

    def _handle_link_fail(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle link failure — if in comms phase, block comms."""
        wl = self._workload
        if wl is None or wl.state != "running":
            return None

        link_id = event.payload.data.get("link_id", event.payload.data.get("component_id", ""))
        self._links_down.add(link_id)

        if self._current_phase == WorkloadPhase.COMMUNICATE and not self._comms_blocked:
            # Cancel pending phase completion
            if self._pending_event is not None:
                self._sim.queue.cancel(self._pending_event)
                self._pending_event = None

            now = ctx.clock.now()
            # Track how much comms time had elapsed
            self._comms_elapsed_before_block = now - self._phase_start_time
            self._comms_blocked = True

        return None

    def _handle_link_repair(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        """Handle link repair — resume comms if blocked."""
        wl = self._workload
        if wl is None:
            return None

        link_id = event.payload.data.get("link_id", event.payload.data.get("component_id", ""))
        self._links_down.discard(link_id)

        # Resume comms if all links back up and we were blocked
        if self._comms_blocked and len(self._links_down) == 0 and wl.state == "running":
            self._comms_blocked = False
            now = ctx.clock.now()

            # Remaining comms time + reroute penalty
            remaining_comms = self._phase_total_duration - self._comms_elapsed_before_block
            if remaining_comms < 0:
                remaining_comms = 0
            total_remaining = remaining_comms + self._reroute_penalty_us

            # Update tracking for potential future interruptions
            self._phase_start_time = now
            self._phase_total_duration = total_remaining

            # Reschedule phase completion
            self._pending_event = self._sim.queue.schedule(
                now + total_remaining,
                EventPayload(
                    event_type="workload.phase.complete",
                    data={
                        "job_id": wl.job_id,
                        "phase": WorkloadPhase.COMMUNICATE.value,
                        "step": wl.current_step,
                    },
                ),
                priority=30,
            )

        return None
