"""Demo runner: 4 scenarios showcasing the full simulation stack.

Scenarios:
1. Baseline: 32 GPUs, 10 steps, no chaos
2. GPU failure: XID error at t=460ms, repairs after 10s
3. Thermal throttle: GPU throttled to 0.33x at t=320ms, unthrottled at t=5s
4. Link flap: link down at t=160ms, up at t=260ms during comms window
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dcsim.engine.clock import MILLISECOND, SECOND, SimTime
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
    """Run all named scenarios from NAMED_SCENARIOS."""
    results: list[ScenarioResult] = []
    for name, events in NAMED_SCENARIOS.items():
        results.append(run_scenario(
            name=name,
            chaos_events=events if events else None,
        ))
    return results


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


# -- CLI chaos event parsing --

# Shorthand -> full event type mapping
_EVENT_SHORTHAND: dict[str, str] = {
    "gpu.fail": "hardware.gpu.fail",
    "gpu.throttle": "hardware.gpu.throttle",
    "gpu.repair": "hardware.gpu.repair",
    "gpu.unthrottle": "hardware.gpu.unthrottle",
    "link.fail": "hardware.link.fail",
    "link.repair": "hardware.link.repair",
    "switch.fail": "hardware.switch.fail",
}

_TIME_RE = re.compile(r"^(\d+(?:\.\d+)?)(us|ms|s)$")


def _parse_time(s: str) -> SimTime:
    """Parse a time string like '320ms', '10s', '1000us' into SimTime (microseconds)."""
    m = _TIME_RE.match(s)
    if not m:
        raise ValueError(f"Invalid time format: {s!r}. Use e.g. '320ms', '10s', '1000us'.")
    value, unit = float(m.group(1)), m.group(2)
    multiplier = {"us": 1, "ms": 1_000, "s": 1_000_000}[unit]
    return int(value * multiplier)


def parse_chaos_string(s: str) -> ChaosEvent:
    """Parse a chaos event string into a ChaosEvent.

    Format: "<event_type> <target_id> <time> [duration] [key=value ...]"

    Examples:
        "gpu.throttle node-1/gpu-4 320ms 5s throttle_factor=0.33"
        "link.fail link-tor-0-spine-0 110ms 100ms"
        "gpu.fail node-3/gpu-1 460ms 10s"
        "gpu.fail node-3/gpu-1 460ms"  (no auto-repair)
    """
    parts = s.strip().split()
    if len(parts) < 3:
        raise ValueError(
            f"Chaos string needs at least 3 parts: '<event_type> <target> <time>'. Got: {s!r}"
        )

    raw_type, target_id, time_str = parts[0], parts[1], parts[2]
    event_type = _EVENT_SHORTHAND.get(raw_type, raw_type)
    time = _parse_time(time_str)

    # Remaining parts: optional duration, then key=value properties
    duration: SimTime | None = None
    properties: dict[str, Any] = {}
    rest = parts[3:]

    if rest and "=" not in rest[0]:
        duration = _parse_time(rest[0])
        rest = rest[1:]

    for kv in rest:
        if "=" not in kv:
            raise ValueError(f"Expected key=value property, got: {kv!r}")
        key, val = kv.split("=", 1)
        # Try to parse as float, fall back to string
        try:
            properties[key] = float(val)
        except ValueError:
            properties[key] = val

    return ChaosEvent(
        target_id=target_id,
        event_type=event_type,
        time=time,
        duration=duration,
        properties=properties,
    )


def load_chaos_file(path: str) -> list[ChaosEvent]:
    """Load chaos events from a JSON file.

    Expected format:
    [
        {
            "target_id": "node-1/gpu-4",
            "event_type": "gpu.throttle",
            "time": "320ms",
            "duration": "5s",
            "properties": {"throttle_factor": 0.33}
        }
    ]
    """
    data = json.loads(Path(path).read_text())
    events: list[ChaosEvent] = []
    for entry in data:
        raw_type = entry["event_type"]
        event_type = _EVENT_SHORTHAND.get(raw_type, raw_type)
        time = _parse_time(entry["time"]) if isinstance(entry["time"], str) else int(entry["time"])
        duration = None
        if entry.get("duration") is not None:
            duration = _parse_time(entry["duration"]) if isinstance(entry["duration"], str) else int(entry["duration"])
        properties = entry.get("properties", {})
        events.append(ChaosEvent(
            target_id=entry["target_id"],
            event_type=event_type,
            time=time,
            duration=duration,
            properties=properties,
        ))
    return events


NAMED_SCENARIOS: dict[str, list[ChaosEvent]] = {
    # --- Single-fault scenarios ---
    "baseline": [],
    "thermal_throttle": [
        ChaosEvent("node-1/gpu-4", "hardware.gpu.throttle", 320 * MILLISECOND,
                   duration=4_680 * MILLISECOND, properties={"throttle_factor": 0.33}),
    ],
    "link_flap": [
        ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 110 * MILLISECOND,
                   duration=100 * MILLISECOND),
    ],
    "xid_79": [
        ChaosEvent("node-3/gpu-1", "hardware.gpu.fail", 460 * MILLISECOND,
                   duration=10_000 * MILLISECOND),
    ],

    # --- Multi-fault scenarios ---

    # Dual throttle: two GPUs in different racks, different severity.
    # GPU 7 at 0.5x is mild; GPU 20 at 0.2x is severe and becomes the laggard.
    "dual_throttle": [
        ChaosEvent("node-0/gpu-7", "hardware.gpu.throttle", 140 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.5}),
        ChaosEvent("node-2/gpu-4", "hardware.gpu.throttle", 510 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.2}),
    ],

    # Throttle + XID: one GPU sluggish from the start, then a second GPU dies
    # mid-run. Training crawls, then halts for 8s, then resumes — still slow.
    "throttle_then_xid": [
        ChaosEvent("node-1/gpu-4", "hardware.gpu.throttle", 80 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.33}),
        ChaosEvent("node-3/gpu-1", "hardware.gpu.fail", 700 * MILLISECOND,
                   duration=8 * SECOND),
    ],

    # Throttle + link flap: GPU overheating slows compute, then a link flap
    # hits during the (already delayed) comms phase. Double penalty.
    "throttle_plus_link_flap": [
        ChaosEvent("node-0/gpu-3", "hardware.gpu.throttle", 50 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.25}),
        ChaosEvent("link-tor-1-spine-0", "hardware.link.fail", 500 * MILLISECOND,
                   duration=80 * MILLISECOND),
    ],

    # XID + link flap: GPU failure knocks out training for 6s, and during
    # recovery a network blip adds insult to injury.
    "xid_plus_link_flap": [
        ChaosEvent("node-2/gpu-5", "hardware.gpu.fail", 320 * MILLISECOND,
                   duration=6 * SECOND),
        ChaosEvent("link-tor-0-spine-1", "hardware.link.fail", 6_500 * MILLISECOND,
                   duration=60 * MILLISECOND),
    ],

    # Cascading rack failure: both spine links from rack 0 go down in quick
    # succession (partial then full isolation), then recover at different times.
    "cascading_link_failure": [
        ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 110 * MILLISECOND,
                   duration=150 * MILLISECOND),
        ChaosEvent("link-tor-0-spine-1", "hardware.link.fail", 130 * MILLISECOND,
                   duration=200 * MILLISECOND),
    ],

    # Everything breaks: thermal throttle on rack 0 GPU, XID on rack 1 GPU,
    # and a link flap — all overlapping. The worst-case scenario.
    "perfect_storm": [
        ChaosEvent("node-0/gpu-2", "hardware.gpu.throttle", 100 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.25}),
        ChaosEvent("node-3/gpu-6", "hardware.gpu.fail", 800 * MILLISECOND,
                   duration=7 * SECOND),
        ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 7_900 * MILLISECOND,
                   duration=100 * MILLISECOND),
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dcsim.demo",
        description="DCSim demo runner. Run preset scenarios or inject custom chaos events.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(NAMED_SCENARIOS.keys()) + ["all"],
        default=None,
        help="Run a named scenario (default: run all if no --chaos given)",
    )
    parser.add_argument(
        "--chaos",
        action="append",
        metavar="EVENT",
        help=(
            'Inject a custom chaos event. Format: "<type> <target> <time> [duration] [key=val ...]". '
            'Example: "gpu.throttle node-1/gpu-4 320ms 5s throttle_factor=0.33". '
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--chaos-file",
        metavar="PATH",
        help="Load chaos events from a JSON file.",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        default=None,
        help="Name for the scenario (used in reports). Defaults to 'Custom chaos'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps (default: 10)",
    )
    return parser


def main() -> None:
    """Run scenarios based on CLI arguments and print results."""
    parser = build_parser()
    args = parser.parse_args()

    results: list[ScenarioResult] = []

    if args.chaos or args.chaos_file:
        # Custom chaos mode
        chaos_events: list[ChaosEvent] = []
        if args.chaos:
            for s in args.chaos:
                chaos_events.append(parse_chaos_string(s))
        if args.chaos_file:
            chaos_events.extend(load_chaos_file(args.chaos_file))

        results.append(run_scenario(
            name=args.name or "Custom chaos",
            chaos_events=chaos_events,
            total_steps=args.steps,
        ))
    elif args.scenario and args.scenario != "all":
        events = NAMED_SCENARIOS[args.scenario]
        results.append(run_scenario(
            name=args.scenario,
            chaos_events=events if events else None,
            total_steps=args.steps,
        ))
    else:
        # Default: run all named scenarios
        for name, events in NAMED_SCENARIOS.items():
            results.append(run_scenario(
                name=name,
                chaos_events=events if events else None,
                total_steps=args.steps,
            ))

    print_summary(results)


if __name__ == "__main__":
    main()
