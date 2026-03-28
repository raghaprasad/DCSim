"""Integration tests: full stack wiring of engine + hardware + chaos + workload + logger.

4 tests:
1. Baseline with logging: 32 GPUs, no chaos, all 10 steps complete, logger captures events.
2. GPU failure via ChaosInjector: GPU fails mid-training, workload aborts, then resumes after repair.
3. Thermal throttle via ChaosInjector: GPU throttled, compute phases slow down.
4. Link flap via ChaosInjector: link down during comms, comms blocked then resumed.
"""

from __future__ import annotations

from dcsim.engine.clock import MILLISECOND, SECOND
from dcsim.engine.loop import SimulationLoop
from dcsim.chaos.injector import ChaosEvent, ChaosInjector
from dcsim.hardware.topology import build_standard_cluster
from dcsim.observer.logger import EventLogger
from dcsim.workloads.allreduce import AllReduceTraining
from dcsim.workloads.base import WorkloadManager


EVENT_TYPES = [
    "hardware.gpu.fail",
    "hardware.gpu.repair",
    "hardware.gpu.throttle",
    "hardware.gpu.unthrottle",
    "hardware.link.fail",
    "hardware.link.repair",
    "hardware.switch.fail",
    "cascade.gpu.job_interrupted",
    "cascade.gpu.throttled",
    "cascade.link.down",
    "workload.phase.start",
    "workload.phase.complete",
    "workload.step.complete",
    "workload.job.complete",
    "workload.job.failed",
]


def _make_gpu_ids() -> list[str]:
    return [f"node-{n}/gpu-{g}" for n in range(4) for g in range(8)]


def _wire_simulation(
    chaos_events: list[ChaosEvent] | None = None,
    total_steps: int = 10,
) -> tuple[SimulationLoop, AllReduceTraining, WorkloadManager, EventLogger]:
    """Wire all components together following the canonical pattern.

    Does NOT call graph.setup(sim) to avoid duplicate cascade events.
    """
    gpu_ids = _make_gpu_ids()

    sim = SimulationLoop()
    workload = AllReduceTraining(
        job_id="train-1",
        gpu_ids=gpu_ids,
        total_steps=total_steps,
        base_compute_us=100_000,
        comms_duration_us=50_000,
    )
    manager = WorkloadManager()
    manager.setup(sim, workload)

    logger = EventLogger()
    handler = logger.make_handler()
    for evt_type in EVENT_TYPES:
        sim.register_handler(evt_type, handler)

    if chaos_events:
        injector = ChaosInjector()
        injector.inject(chaos_events, sim.queue)

    return sim, workload, manager, logger


class TestIntegrationBaselineWithLogging:
    """Baseline: 32 GPUs, no chaos. All 10 steps complete, logger captures events."""

    def test_baseline_logged(self):
        sim, workload, manager, logger = _wire_simulation()
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10
        assert result.final_time == 1_500_000

        timeline = logger.get_timeline()
        assert len(timeline) > 0

        # Should have exactly 10 step complete events
        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10

        # Should have exactly 1 job complete event
        job_completes = [e for e in timeline if e.event_type == "workload.job.complete"]
        assert len(job_completes) == 1

        # Should have 20 phase starts (10 compute + 10 communicate)
        phase_starts = [e for e in timeline if e.event_type == "workload.phase.start"]
        assert len(phase_starts) == 20

        # No failure events
        failures = [e for e in timeline if "fail" in e.event_type]
        assert len(failures) == 0


class TestIntegrationGPUFailure:
    """GPU failure via ChaosInjector: GPU fails at t=460ms, repairs at t=10.46s."""

    def test_gpu_failure_and_recovery(self):
        chaos_events = [
            ChaosEvent(
                target_id="node-3/gpu-1",
                event_type="hardware.gpu.fail",
                time=460 * MILLISECOND,
                duration=10_000 * MILLISECOND,
            ),
        ]
        sim, workload, manager, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()

        # Should have GPU fail and repair events logged
        gpu_fails = [e for e in timeline if e.event_type == "hardware.gpu.fail"]
        gpu_repairs = [e for e in timeline if e.event_type == "hardware.gpu.repair"]
        assert len(gpu_fails) == 1
        assert len(gpu_repairs) == 1
        assert gpu_fails[0].timestamp == 460_000
        assert gpu_repairs[0].timestamp == 10_460_000

        # Should have job interrupted event
        interrupted = [e for e in timeline if e.event_type == "workload.job.failed"]
        assert len(interrupted) == 1

        # Total time must be > 11s (repair at ~10.46s + remaining 7 steps)
        assert result.final_time > 11_000_000

        # Should still complete all 10 steps
        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


class TestIntegrationThermalThrottle:
    """Thermal throttle via ChaosInjector: GPU throttled to 0.33 at t=320ms."""

    def test_throttle_slows_training(self):
        chaos_events = [
            ChaosEvent(
                target_id="node-1/gpu-4",
                event_type="hardware.gpu.throttle",
                time=320 * MILLISECOND,
                duration=None,  # No auto-repair: stays throttled
                properties={"throttle_factor": 0.33},
            ),
        ]
        sim, workload, manager, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()

        # Should have throttle event logged
        throttles = [e for e in timeline if e.event_type == "hardware.gpu.throttle"]
        assert len(throttles) == 1

        # Total time should be > baseline 1.5s because throttled compute is 3x slower
        assert result.final_time > 1_500_000

        # All 10 steps must complete
        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


class TestIntegrationLinkFlap:
    """Link flap via ChaosInjector: link down at t=110ms, up at t=210ms.

    Step 0 comms runs 100-150ms. Link fails at t=110ms (during comms), blocking it.
    Link repairs at t=210ms, comms resumes with reroute penalty (10ms default).
    Total time should exceed baseline 1.5s.
    """

    def test_link_flap_delays_comms(self):
        chaos_events = [
            ChaosEvent(
                target_id="link-tor-0-spine-0",
                event_type="hardware.link.fail",
                time=110 * MILLISECOND,
                duration=100 * MILLISECOND,  # repairs at 210ms
            ),
        ]
        sim, workload, manager, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()

        # Should have link fail and repair events logged
        link_fails = [e for e in timeline if e.event_type == "hardware.link.fail"]
        link_repairs = [e for e in timeline if e.event_type == "hardware.link.repair"]
        assert len(link_fails) == 1
        assert len(link_repairs) == 1
        assert link_fails[0].timestamp == 110_000
        assert link_repairs[0].timestamp == 210_000

        # Total time should be > baseline 1.5s due to comms delay + reroute penalty
        assert result.final_time > 1_500_000

        # All 10 steps must complete
        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10
