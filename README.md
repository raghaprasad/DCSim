# DCSim: SimCity for Datacenter Chaos Engineering

A discrete event simulator that shows what happens to AI training runs when datacenter hardware inevitably breaks. Inject GPU failures, thermal throttling, and network outages into a virtual 32-GPU cluster and watch the impact unfold.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m dcsim --html reports/ --open
```

This runs 4 scenarios and opens an interactive HTML report in your browser.

## What You'll See

A 32-GPU cluster (4 nodes x 8 GPUs, 2 racks) runs 10 AllReduce training iterations. Each iteration = 100ms compute + 50ms allreduce = 150ms baseline. Then chaos happens:

```
Scenario                       State          Steps    Time (ms)
------------------------------------------------------------------------
Baseline (no chaos)            completed      10/10        1500.0
GPU Failure (XID error)        completed      10/10       11410.0
Thermal Throttle (0.33x)       completed      10/10        3083.6
Link Flap (100ms outage)       completed      10/10        1610.0
```

Each scenario report includes:
- **Datacenter topology diagram** — SVG showing racks, nodes, GPUs, and switches with affected components color-coded (green=healthy, orange=throttled, red=failed)
- **GPU state timeline** — Gantt chart showing compute/communicate/idle/failed states over time
- **Iteration durations** — bar chart comparing per-step compute vs communicate time
- **Event log** — timestamped table of every hardware failure, cascade, and workload reaction

### Scenario Details

| Scenario | What happens | Impact |
|----------|-------------|--------|
| **Baseline** | No failures | 1,500ms — 10 clean iterations at 150ms each |
| **Thermal Throttle** | GPU `node-1/gpu-4` throttled to 0.33x at t=320ms | 3,084ms (+106%) — every iteration slows because training is synchronous: all 31 healthy GPUs wait for the one slow GPU |
| **Link Flap** | ToR-to-spine link down for 100ms at t=110ms | 1,610ms (+7%) — allreduce blocked during outage, 10ms reroute penalty on recovery |
| **XID Failure** | GPU `node-3/gpu-1` hard failure at t=460ms, 10s repair | 11,410ms (+661%) — training halts for 10s while GPU reboots, resumes from interrupted step |

## CLI Options

```bash
python -m dcsim                              # Terminal summary only
python -m dcsim --html reports/              # Generate per-scenario HTML reports
python -m dcsim --html reports/ --open       # Generate and open in browser
python -m dcsim --scenario thermal-throttle  # Run a single scenario
```

Available scenarios: `baseline`, `gpu-failure`, `thermal-throttle`, `link-flap`, `all`

## Architecture

Five event-driven layers communicating through a priority queue:

```
Engine          Priority queue + event loop. All times in integer microseconds.
    ↓
Hardware        32 GPUs, 4 switches, links — each a state machine. Failures cascade
                (switch dies → links go down → GPUs isolated).
    ↓
Workloads       AllReduce training: COMPUTE → COMMUNICATE → step++. Synchronous
                barrier means the slowest GPU dominates every iteration.
    ↓
Chaos           Injects hardware events at specified times. Auto-schedules repairs.
    ↓
Observer        Logs every event with causal chain tracking (parent_event_id).
```

No component calls another directly — they only emit and consume events. This makes the system deterministic (same chaos input = identical output) and each layer independently testable.

## Project Structure

```
src/dcsim/
  engine/           Core simulation engine (frozen — do not modify)
    clock.py          SimTime type, microsecond constants
    event.py          Event, EventPayload, EventQueue (heapq + tombstone cancel)
    loop.py           SimulationLoop, handler dispatch
  hardware/         Datacenter hardware model
    components.py     GPU/Switch/Link state machines with valid transitions
    topology.py       build_standard_cluster() → 4 nodes x 8 GPUs
    graph.py          HardwareGraph with cascade propagation (NetworkX)
  workloads/        AI workload simulation
    base.py           Workload ABC, WorkloadManager (event wiring + chaos handling)
    allreduce.py      AllReduceTraining (compute/communicate cycle)
  chaos/            Failure injection
    injector.py       ChaosInjector + ChaosEvent dataclass
  observer/         Event logging
    logger.py         EventLogger with timeline export
  demo.py           Scenario definitions + runner
  visualize.py      Plotly + SVG chart generation
tests/              37 tests across all layers
runbooks/           7 guides for extending the simulator
```

## Running Tests

```bash
pytest tests/ -v
```

37 tests covering:
- **Engine** (14): deterministic ordering, time advancement, cancellation, handler chaining, 100k event performance
- **Hardware** (10): topology construction, connectivity, state machines, cascade propagation
- **Workloads** (5): baseline timing, throttle impact, link flap, XID failure, step counting
- **Chaos + Observer** (4): injection scheduling, auto-repair, logger timeline/export
- **Integration** (4): full E2E scenarios matching the demo output

## Extending

See the `runbooks/` directory for step-by-step guides:

- `add-hardware-type.md` — Add TPUs, DPUs, storage nodes
- `add-perf-profile.md` — Model H100 vs A100 vs B200 performance differences
- `add-workload.md` — Add inference, pipeline parallelism, etc.
- `add-chaos-event.md` — Add ECC errors, NVLink CRC, power failures
- `add-topology.md` — Add torus, dragonfly, custom topologies
- `add-failure-distribution.md` — Add Weibull, correlated failure models
- `add-visualization.md` — Add new chart types to the report

## Requirements

- Python 3.12+
- Dependencies: `networkx`, `plotly` (installed automatically via `pip install -e .`)
- Dev: `pytest` (installed via `pip install -e ".[dev]"`)
