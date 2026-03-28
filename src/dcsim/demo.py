"""Demo runner: 4 scenarios showcasing the full simulation stack.

Scenarios:
1. Baseline: 32 GPUs, 10 steps, no chaos
2. GPU failure: XID error at t=460ms, repairs after 10s
3. Thermal throttle: GPU throttled to 0.33x at t=320ms, unthrottled at t=5s
4. Link flap: link down at t=160ms, up at t=260ms during comms window
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dcsim.engine.clock import MILLISECOND, SECOND
from dcsim.engine.loop import SimulationLoop
from dcsim.chaos.injector import ChaosEvent, ChaosInjector
from dcsim.observer.logger import EventLogger, LogEntry
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


@dataclass
class ScenarioResult:
    """Result of a single demo scenario."""

    name: str
    workload_state: str
    steps_completed: int
    total_steps: int
    final_time_us: int
    events_processed: int
    timeline: list[LogEntry]
    log_dicts: list[dict[str, Any]]


def _make_gpu_ids() -> list[str]:
    return [f"node-{n}/gpu-{g}" for n in range(4) for g in range(8)]


def run_scenario(
    name: str,
    chaos_events: list[ChaosEvent] | None = None,
    total_steps: int = 10,
) -> ScenarioResult:
    """Run a single scenario and return structured results."""
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

    result = sim.run(until=60 * SECOND)

    # Find actual training completion time (not sim end time which may
    # include post-training events like unthrottle/repair)
    training_end_us = result.final_time
    for entry in logger.get_timeline():
        if entry.event_type == "workload.job.complete":
            training_end_us = entry.timestamp
            break

    return ScenarioResult(
        name=name,
        workload_state=workload.state,
        steps_completed=workload.current_step,
        total_steps=workload.total_steps,
        final_time_us=training_end_us,
        events_processed=result.events_processed,
        timeline=logger.get_timeline(),
        log_dicts=logger.export_dicts(),
    )


def scenario_baseline() -> ScenarioResult:
    """Scenario 1: Baseline -- no chaos, 10 steps."""
    return run_scenario(name="Baseline (no chaos)")


def scenario_gpu_failure() -> ScenarioResult:
    """Scenario 2: GPU XID failure at t=460ms, repairs after 10s."""
    chaos = [
        ChaosEvent(
            target_id="node-3/gpu-1",
            event_type="hardware.gpu.fail",
            time=460 * MILLISECOND,
            duration=10_000 * MILLISECOND,
        ),
    ]
    return run_scenario(name="GPU Failure (XID error)", chaos_events=chaos)


def scenario_thermal_throttle() -> ScenarioResult:
    """Scenario 3: GPU throttled to 0.33x at t=320ms, unthrottled at t=5s."""
    chaos = [
        ChaosEvent(
            target_id="node-1/gpu-4",
            event_type="hardware.gpu.throttle",
            time=320 * MILLISECOND,
            duration=4_680 * MILLISECOND,  # unthrottles at 5s
            properties={"throttle_factor": 0.33},
        ),
    ]
    return run_scenario(name="Thermal Throttle (0.33x)", chaos_events=chaos)


def scenario_link_flap() -> ScenarioResult:
    """Scenario 4: Link flap -- down at t=110ms, up at t=210ms (during step 0 comms)."""
    chaos = [
        ChaosEvent(
            target_id="link-tor-0-spine-0",
            event_type="hardware.link.fail",
            time=110 * MILLISECOND,
            duration=100 * MILLISECOND,  # repairs at 210ms, during comms blockage
        ),
    ]
    return run_scenario(name="Link Flap (100ms outage)", chaos_events=chaos)


def run_all_scenarios() -> list[ScenarioResult]:
    """Run all 4 demo scenarios and return results."""
    return [
        scenario_baseline(),
        scenario_gpu_failure(),
        scenario_thermal_throttle(),
        scenario_link_flap(),
    ]


def print_summary(results: list[ScenarioResult]) -> None:
    """Print a human-readable summary table of all scenario results."""
    print()
    print("=" * 80)
    print("DCSim Demo Results")
    print("=" * 80)
    print()
    print(f"{'Scenario':<30} {'State':<14} {'Steps':<8} {'Time (ms)':<14} {'Events':<8}")
    print("-" * 80)

    for r in results:
        time_ms = r.final_time_us / 1_000
        print(
            f"{r.name:<30} {r.workload_state:<14} "
            f"{r.steps_completed}/{r.total_steps:<5} "
            f"{time_ms:>10.1f}    {r.events_processed:<8}"
        )

    print("-" * 80)
    print()

    for r in results:
        print(f"--- {r.name} ---")
        step_events = [e for e in r.timeline if e.event_type == "workload.step.complete"]
        chaos_events = [
            e for e in r.timeline
            if e.event_type.startswith("hardware.") or e.event_type.startswith("cascade.")
        ]
        print(f"  Steps completed: {len(step_events)}")
        print(f"  Hardware/cascade events: {len(chaos_events)}")

        if chaos_events:
            print("  Key events:")
            for e in chaos_events[:10]:
                t_ms = e.timestamp / 1_000
                print(f"    t={t_ms:.1f}ms  {e.event_type}  {e.component_id or ''}")

        print()


def main() -> None:
    """Run all scenarios and print results."""
    results = run_all_scenarios()
    print_summary(results)


if __name__ == "__main__":
    main()
