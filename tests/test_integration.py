"""Integration tests: full stack wiring of engine + hardware + chaos + workload + logger.

Single-fault tests:
1. Baseline: no chaos, 10 steps at 1500ms
2. GPU failure (XID): 10s penalty
3. Thermal throttle: 3x compute slowdown
4. Link flap: comms delay + reroute penalty

Multi-fault combinatorial tests:
5. Dual throttle: different factors, only the worst GPU is the laggard
6. Triple throttle: three GPUs at different factors, worst dominates
7. Throttle + GPU failure: throttle first, then failure mid-run
8. Throttle + link flap: throttle slows compute, link flap delays comms
9. GPU failure + link flap: failure during compute, link flap during comms
10. Triple combo: throttle + GPU failure + link flap
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
    """Link flap via ChaosInjector: link down at t=110ms, up at t=210ms."""

    def test_link_flap_delays_comms(self):
        chaos_events = [
            ChaosEvent(
                target_id="link-tor-0-spine-0",
                event_type="hardware.link.fail",
                time=110 * MILLISECOND,
                duration=100 * MILLISECOND,
            ),
        ]
        sim, workload, manager, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()
        link_fails = [e for e in timeline if e.event_type == "hardware.link.fail"]
        link_repairs = [e for e in timeline if e.event_type == "hardware.link.repair"]
        assert len(link_fails) == 1
        assert len(link_repairs) == 1
        assert link_fails[0].timestamp == 110_000
        assert link_repairs[0].timestamp == 210_000
        assert result.final_time > 1_500_000

        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


# ---------------------------------------------------------------------------
# Multi-fault combinatorial tests
# ---------------------------------------------------------------------------

BASELINE_US = 1_500_000  # 10 steps * 150ms


class TestDualThrottle:
    """Two GPUs throttled at different factors. Only the worst (lowest) factor
    determines the compute duration because training is synchronous."""

    def test_worst_gpu_dominates(self):
        # GPU 7 at 0.5x (mild), GPU 20 at 0.2x (severe)
        # min(0.5, 0.2) = 0.2 → compute = 100ms / 0.2 = 500ms per step
        chaos_events = [
            ChaosEvent("node-0/gpu-7", "hardware.gpu.throttle", 140 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.5}),
            ChaosEvent("node-2/gpu-4", "hardware.gpu.throttle", 510 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.2}),
        ]
        sim, workload, manager, logger = _wire_simulation(chaos_events)
        result = sim.run(until=120 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        # Verify both throttle events fired
        throttles = [e for e in logger.get_timeline() if e.event_type == "hardware.gpu.throttle"]
        assert len(throttles) == 2

        # After GPU 20 throttled at 510ms, compute = 500ms/step.
        # Total must be significantly above baseline
        assert result.final_time > BASELINE_US * 2

        # The period between GPU 7 throttle (140ms) and GPU 20 throttle (510ms)
        # has compute at 200ms (1/0.5). After GPU 20, compute jumps to 500ms (1/0.2).
        # So total should be well above what a single 0.5x throttle would produce.
        # Single 0.5x from t=140ms: ~2400ms. Dual with 0.2x: much higher.
        single_05x_estimate = 2_500_000  # rough upper bound for single 0.5x
        assert result.final_time > single_05x_estimate, (
            f"Dual throttle ({result.final_time/1000:.0f}ms) should exceed "
            f"single 0.5x estimate ({single_05x_estimate/1000:.0f}ms)"
        )

    def test_milder_throttle_has_no_additional_effect(self):
        # GPU 7 at 0.2x (severe, arrives first), GPU 20 at 0.5x (mild, arrives second)
        # After both: min(0.2, 0.5) = 0.2 → the second throttle changes nothing
        chaos_severe_first = [
            ChaosEvent("node-0/gpu-7", "hardware.gpu.throttle", 140 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.2}),
            ChaosEvent("node-2/gpu-4", "hardware.gpu.throttle", 510 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.5}),
        ]
        sim1, wl1, _, _ = _wire_simulation(chaos_severe_first)
        sim1.run(until=120 * SECOND)

        # Compare: only the severe throttle, no mild one
        chaos_severe_only = [
            ChaosEvent("node-0/gpu-7", "hardware.gpu.throttle", 140 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.2}),
        ]
        sim2, wl2, _, _ = _wire_simulation(chaos_severe_only)
        sim2.run(until=120 * SECOND)

        # Both should finish at the same time — mild throttle adds nothing
        assert sim1.clock.now() == sim2.clock.now(), (
            f"Adding a milder throttle should not change completion time: "
            f"dual={sim1.clock.now()/1000:.1f}ms vs single={sim2.clock.now()/1000:.1f}ms"
        )


class TestTripleThrottle:
    """Three GPUs at different throttle factors. Only the worst dominates."""

    def test_worst_of_three_dominates(self):
        # GPU A at 0.5x, GPU B at 0.33x, GPU C at 0.1x (extreme)
        # min = 0.1 → compute = 1000ms per step
        chaos_events = [
            ChaosEvent("node-0/gpu-0", "hardware.gpu.throttle", 50 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.5}),
            ChaosEvent("node-1/gpu-3", "hardware.gpu.throttle", 200 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.33}),
            ChaosEvent("node-3/gpu-7", "hardware.gpu.throttle", 400 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.1}),
        ]
        sim, workload, _, logger = _wire_simulation(chaos_events)
        result = sim.run(until=300 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        throttles = [e for e in logger.get_timeline() if e.event_type == "hardware.gpu.throttle"]
        assert len(throttles) == 3

        # After the 0.1x throttle, each compute = 1000ms, each step = 1050ms
        # Total should be very high
        assert result.final_time > 7_000_000  # > 7 seconds


class TestThrottlePlusGPUFailure:
    """Throttle first, then GPU failure mid-run. Both effects compound."""

    def test_throttle_then_failure(self):
        chaos_events = [
            # GPU throttled at 200ms (during step 1 compute: 150-250ms)
            ChaosEvent("node-1/gpu-4", "hardware.gpu.throttle", 200 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.33}),
            # Different GPU fails at 800ms, repairs after 5s
            ChaosEvent("node-3/gpu-1", "hardware.gpu.fail", 800 * MILLISECOND,
                       duration=5 * SECOND),
        ]
        sim, workload, _, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()
        throttles = [e for e in timeline if e.event_type == "hardware.gpu.throttle"]
        failures = [e for e in timeline if e.event_type == "hardware.gpu.fail"]
        repairs = [e for e in timeline if e.event_type == "hardware.gpu.repair"]
        assert len(throttles) == 1
        assert len(failures) == 1
        assert len(repairs) == 1

        # Must be slower than throttle-only AND have the 5s failure penalty
        assert result.final_time > 5 * SECOND + BASELINE_US

        # After repair, training resumes — still throttled at 0.33x
        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


class TestThrottlePlusLinkFlap:
    """Throttle slows compute, link flap delays comms. Both penalties stack."""

    def test_throttle_and_link_flap(self):
        chaos_events = [
            # GPU throttled from the start
            ChaosEvent("node-2/gpu-0", "hardware.gpu.throttle", 50 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.33}),
            # Link flap during step 1 comms (at 0.33x: step 0 = 303ms compute + 50ms comms
            # = finishes ~353ms. Step 1 compute starts ~353ms, finishes ~656ms.
            # Step 1 comms starts ~656ms)
            ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 660 * MILLISECOND,
                       duration=50 * MILLISECOND),
        ]
        sim, workload, _, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()
        throttles = [e for e in timeline if e.event_type == "hardware.gpu.throttle"]
        link_fails = [e for e in timeline if e.event_type == "hardware.link.fail"]
        assert len(throttles) == 1
        assert len(link_fails) == 1

        # Must be slower than throttle-only (throttle alone ~3084ms)
        throttle_only_approx = 3_000_000
        assert result.final_time > throttle_only_approx

        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


class TestGPUFailurePlusLinkFlap:
    """GPU failure during compute, link flap during comms. Both penalties apply."""

    def test_failure_and_link_flap(self):
        chaos_events = [
            # Link flap during step 0 comms (100-150ms)
            ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 110 * MILLISECOND,
                       duration=100 * MILLISECOND),
            # GPU fails at 460ms, 8s repair
            ChaosEvent("node-3/gpu-1", "hardware.gpu.fail", 460 * MILLISECOND,
                       duration=8 * SECOND),
        ]
        sim, workload, _, logger = _wire_simulation(chaos_events)
        result = sim.run(until=60 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()
        link_fails = [e for e in timeline if e.event_type == "hardware.link.fail"]
        gpu_fails = [e for e in timeline if e.event_type == "hardware.gpu.fail"]
        assert len(link_fails) == 1
        assert len(gpu_fails) == 1

        # Must include the 8s failure penalty plus the link flap delay
        assert result.final_time > 8 * SECOND + BASELINE_US

        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10


class TestTripleCombo:
    """All three fault types simultaneously: throttle + GPU failure + link flap."""

    def test_all_three_faults(self):
        chaos_events = [
            # Throttle GPU at 50ms (early, affects all subsequent compute)
            ChaosEvent("node-0/gpu-3", "hardware.gpu.throttle", 50 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.25}),
            # Link flap during a comms window
            ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 450 * MILLISECOND,
                       duration=80 * MILLISECOND),
            # GPU failure at 1s, 6s repair (different GPU than throttled one)
            ChaosEvent("node-2/gpu-5", "hardware.gpu.fail", 1 * SECOND,
                       duration=6 * SECOND),
        ]
        sim, workload, _, logger = _wire_simulation(chaos_events)
        result = sim.run(until=120 * SECOND)

        assert workload.state == "completed"
        assert workload.current_step == 10

        timeline = logger.get_timeline()
        throttles = [e for e in timeline if e.event_type == "hardware.gpu.throttle"]
        link_fails = [e for e in timeline if e.event_type == "hardware.link.fail"]
        gpu_fails = [e for e in timeline if e.event_type == "hardware.gpu.fail"]
        assert len(throttles) == 1
        assert len(link_fails) == 1
        assert len(gpu_fails) == 1

        # All three penalties compound: throttle (4x compute) + link flap + 6s failure
        assert result.final_time > 6 * SECOND + BASELINE_US

        step_completes = [e for e in timeline if e.event_type == "workload.step.complete"]
        assert len(step_completes) == 10

    def test_triple_combo_slower_than_any_single(self):
        """Triple combo must be slower than any individual fault alone."""
        # Run each fault individually
        single_throttle = [
            ChaosEvent("node-0/gpu-3", "hardware.gpu.throttle", 50 * MILLISECOND,
                       duration=None, properties={"throttle_factor": 0.25}),
        ]
        single_failure = [
            ChaosEvent("node-2/gpu-5", "hardware.gpu.fail", 1 * SECOND,
                       duration=6 * SECOND),
        ]
        single_link = [
            ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 450 * MILLISECOND,
                       duration=80 * MILLISECOND),
        ]

        times = {}
        for name, events in [("throttle", single_throttle), ("failure", single_failure), ("link", single_link)]:
            sim, wl, _, _ = _wire_simulation(events)
            sim.run(until=120 * SECOND)
            times[name] = sim.clock.now()

        # Triple combo
        all_events = single_throttle + single_failure + single_link
        sim, wl, _, _ = _wire_simulation(all_events)
        sim.run(until=120 * SECOND)
        combo_time = sim.clock.now()

        for name, t in times.items():
            assert combo_time > t, (
                f"Triple combo ({combo_time/1000:.0f}ms) should be slower than "
                f"{name} alone ({t/1000:.0f}ms)"
            )
