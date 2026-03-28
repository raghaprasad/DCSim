"""AllReduce training workload.

Implements a synchronous AllReduce training loop:
  COMPUTE (100ms) -> COMMUNICATE (50ms) -> step++ -> repeat

The compute phase duration is governed by the slowest GPU (min throttle_factor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dcsim.engine.clock import SimTime
from dcsim.workloads.base import Workload, WorkloadPhase


@dataclass
class AllReduceTraining(Workload):
    """Synchronous AllReduce training workload.

    Each step consists of a COMPUTE phase followed by a COMMUNICATE phase.
    The compute duration scales inversely with the minimum throttle factor
    across all GPUs (synchronous barrier — slowest GPU dominates).
    """

    base_compute_us: SimTime = 100_000   # 100ms default
    comms_duration_us: SimTime = 50_000  # 50ms default

    # Internal tracking of current sub-phase within a step
    _in_communicate: bool = field(default=False, repr=False)

    def get_next_phase(self, gpu_states: dict[str, dict[str, Any]], now: SimTime) -> tuple[WorkloadPhase, SimTime] | None:
        """Return the next phase and its duration, or None if training is done."""
        if self.current_step >= self.total_steps:
            return None

        if not self._in_communicate:
            # COMPUTE phase: duration = base_compute_us / min(throttle_factor)
            min_throttle = 1.0
            for state in gpu_states.values():
                tf = state.get("throttle_factor", 1.0)
                if tf < min_throttle:
                    min_throttle = tf
            if min_throttle <= 0:
                min_throttle = 0.01  # Guard against division by zero

            duration = int(self.base_compute_us / min_throttle)
            self._in_communicate = True  # Next call will return COMMUNICATE
            return (WorkloadPhase.COMPUTE, duration)
        else:
            # COMMUNICATE phase
            self._in_communicate = False  # Next call will return COMPUTE (for next step)
            return (WorkloadPhase.COMMUNICATE, self.comms_duration_us)

    def on_gpu_failed(self, gpu_id: str) -> str:
        """AllReduce training aborts on any GPU failure."""
        return "abort"
