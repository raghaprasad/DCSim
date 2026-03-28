# DCSim: SimCity for Datacenter Chaos Engineering

A discrete event simulator that shows what happens to AI training runs when datacenter hardware inevitably breaks. Inject GPU failures, thermal throttling, and network outages into a virtual 32-GPU cluster and watch the impact unfold.

## Quick Start (CLI)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m dcsim
```

This runs 10 scenarios (4 single-fault + 6 multi-fault), generates interactive HTML reports in `reports/`, and opens them in your browser.

## Quick Start (Web App)

Run the simulator interactively in your browser — no server, everything runs client-side via WebAssembly:

```bash
# Build the wheel
pip install -e ".[dev]"
bash web/build_wheel.sh

# Serve locally
cd web && python3 -m http.server 8080
# Open http://localhost:8080
```

### Deploy to Vercel

```bash
cd web && npx vercel deploy
```

Or connect the GitHub repo to Vercel with root directory set to `web/`.

## What You'll See

A 32-GPU cluster (4 nodes x 8 GPUs, 2 racks) runs 10 AllReduce training iterations. Each iteration = 100ms compute + 50ms allreduce = 150ms baseline. Then chaos happens:

```
Scenario                       State          Steps    Time (ms)
------------------------------------------------------------------------
baseline                       completed      10/10        1500.0
thermal_throttle               completed      10/10        3083.6
link_flap                      completed      10/10        1610.0
xid_79                         completed      10/10       11410.0
dual_throttle                  completed      10/10        4635.0
throttle_then_xid              completed      10/10       11221.2
throttle_plus_link_flap        completed      10/10        4350.0
xid_plus_link_flap             completed      10/10        7490.0
cascading_link_failure         completed      10/10        1730.0
perfect_storm                  completed      10/10       11000.0
```

Each scenario report includes:
- **Datacenter topology diagram** — SVG showing racks, nodes, GPUs, and switches with affected components color-coded (green=healthy, orange=throttled, red=failed)
- **GPU state timeline** — Gantt chart showing compute/communicate/idle/failed states per GPU group, with sync barrier idle time visible
- **Iteration durations** — bar chart comparing per-step compute vs communicate time
- **Event log** — timestamped table of every hardware failure, cascade, and workload reaction

### Scenario Details

| Scenario | What happens | Impact |
|----------|-------------|--------|
| **baseline** | No failures | 1,500ms — 10 clean iterations at 150ms each |
| **thermal_throttle** | GPU `node-1/gpu-4` throttled to 0.33x at t=320ms | 3,084ms (+106%) — all 31 healthy GPUs idle-wait at sync barrier |
| **link_flap** | ToR-to-spine link down for 100ms at t=110ms | 1,610ms (+7%) — allreduce blocked, 10ms reroute penalty |
| **xid_79** | GPU `node-3/gpu-1` hard failure at t=460ms, 10s repair | 11,410ms (+661%) — training halts for 10s |
| **dual_throttle** | GPU at 0.5x + GPU at 0.2x (different racks) | 4,635ms (+209%) — only the 0.2x GPU is the laggard |
| **throttle_then_xid** | 0.33x throttle then XID failure mid-run | 11,221ms (+648%) — slow compute + 8s repair penalty |
| **throttle_plus_link_flap** | 0.25x throttle + 80ms network outage | 4,350ms (+190%) — both penalties stack |
| **xid_plus_link_flap** | 6s GPU failure + link flap during recovery | 7,490ms (+399%) — sequential penalties |
| **cascading_link_failure** | Both spine links from rack 0 fail in sequence | 1,730ms (+15%) — double comms disruption |
| **perfect_storm** | 0.25x throttle + 7s XID + link flap | 11,000ms (+633%) — everything breaks at once |

## CLI Options

```bash
# Run all 10 preset scenarios (HTML reports generated + opened by default)
python -m dcsim

# Terminal summary only, no HTML
python -m dcsim --no-html

# Run a single preset scenario
python -m dcsim --scenario thermal-throttle

# Custom chaos injection
python -m dcsim --chaos "gpu.throttle node-1/gpu-4 320ms 5s throttle_factor=0.33"
python -m dcsim --chaos "gpu.fail node-3/gpu-1 460ms 10s" --chaos "link.fail link-tor-0-spine-0 110ms 100ms"
python -m dcsim --chaos-file demo/example_chaos.json

# Name your scenario and adjust steps
python -m dcsim --chaos "gpu.fail node-0/gpu-0 1s 10s" --name "Single GPU crash" --steps 20
```

Available preset scenarios: `baseline`, `gpu-failure`, `thermal-throttle`, `link-flap`, `all`

### Custom chaos format

```
<event_type> <target_id> <time> [duration] [key=value ...]
```

| Event type | Shorthand | Target example |
|---|---|---|
| `hardware.gpu.fail` | `gpu.fail` | `node-1/gpu-4` |
| `hardware.gpu.throttle` | `gpu.throttle` | `node-1/gpu-4` (add `throttle_factor=0.33`) |
| `hardware.link.fail` | `link.fail` | `link-tor-0-spine-0` |
| `hardware.switch.fail` | `switch.fail` | `tor-0` |

Times accept `us`, `ms`, `s` suffixes. If `duration` is given, a matching repair event is auto-scheduled.

### Try These Custom Scenarios

**Gradual degradation** — three GPUs throttle at different times and severities:
```
gpu.throttle node-0/gpu-2 100ms throttle_factor=0.5
gpu.throttle node-1/gpu-5 400ms throttle_factor=0.33
gpu.throttle node-3/gpu-0 700ms throttle_factor=0.1
```

**Rack 0 meltdown** — simultaneous failures across an entire rack:
```
gpu.fail node-0/gpu-3 200ms 8s
gpu.throttle node-1/gpu-7 250ms throttle_factor=0.25
link.fail link-tor-0-spine-0 300ms 500ms
link.fail link-tor-0-spine-1 320ms 600ms
```

**Rolling failures** — faults appear one after another, each during recovery from the last:
```
gpu.fail node-0/gpu-0 100ms 3s
gpu.fail node-1/gpu-4 3200ms 3s
gpu.fail node-2/gpu-6 6300ms 3s
```

**Stress test** — many small disruptions that individually are minor but accumulate:
```
gpu.throttle node-0/gpu-1 50ms 200ms throttle_factor=0.5
link.fail link-tor-0-spine-0 150ms 30ms
gpu.throttle node-2/gpu-3 350ms 200ms throttle_factor=0.5
link.fail link-tor-1-spine-1 550ms 30ms
gpu.throttle node-1/gpu-6 750ms 200ms throttle_factor=0.5
link.fail link-tor-0-spine-1 950ms 30ms
```

You can also define scenarios in a JSON file (see `demo/example_chaos.json`):

```json
[
    {
        "target_id": "node-1/gpu-4",
        "event_type": "gpu.throttle",
        "time": "320ms",
        "duration": "5s",
        "properties": {"throttle_factor": 0.33}
    }
]
```

## Architecture

Five event-driven layers communicating through a priority queue:

```
Engine          Priority queue + event loop. All times in integer microseconds.
    |
Hardware        32 GPUs, 4 switches, links -- each a state machine. Failures cascade
                (switch dies -> links go down -> GPUs isolated).
    |
Workloads       AllReduce training: COMPUTE -> COMMUNICATE -> step++. Synchronous
                barrier means the slowest GPU dominates every iteration.
    |
Chaos           Injects hardware events at specified times. Auto-schedules repairs.
    |
Observer        Logs every event with causal chain tracking (parent_event_id).
```

No component calls another directly -- they only emit and consume events. This makes the system deterministic (same chaos input = identical output) and each layer independently testable.

## Project Structure

```
src/dcsim/
  engine/           Core simulation engine (frozen -- do not modify)
    clock.py          SimTime type, microsecond constants
    event.py          Event, EventPayload, EventQueue (heapq + tombstone cancel)
    loop.py           SimulationLoop, handler dispatch
  hardware/         Datacenter hardware model
    components.py     GPU/Switch/Link state machines with valid transitions
    topology.py       build_standard_cluster() -> 4 nodes x 8 GPUs
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
web/                Browser-based UI (Pyodide + Plotly)
  index.html          Single-page app, runs Python in WebAssembly
  dist/               Built wheel loaded by Pyodide at runtime
  build_wheel.sh      Rebuilds the wheel after code changes
  vercel.json         Vercel static deployment config
tests/              45 tests across all layers
runbooks/           7 guides for extending the simulator
```

## Running Tests

```bash
pytest tests/ -v
```

45 tests covering:
- **Engine** (14): deterministic ordering, time advancement, cancellation, handler chaining, 100k event performance
- **Hardware** (10): topology construction, connectivity, state machines, cascade propagation
- **Workloads** (5): baseline timing, throttle impact, link flap, XID failure, step counting
- **Chaos + Observer** (4): injection scheduling, auto-repair, logger timeline/export
- **Integration** (12): single-fault E2E, dual/triple throttle, all multi-fault combos, triple-slower-than-single proof

## Extending

See the `runbooks/` directory for step-by-step guides:

- `add-hardware-type.md` -- Add TPUs, DPUs, storage nodes
- `add-perf-profile.md` -- Model H100 vs A100 vs B200 performance differences
- `add-workload.md` -- Add inference, pipeline parallelism, etc.
- `add-chaos-event.md` -- Add ECC errors, NVLink CRC, power failures
- `add-topology.md` -- Add torus, dragonfly, custom topologies
- `add-failure-distribution.md` -- Add Weibull, correlated failure models
- `add-visualization.md` -- Add new chart types to the report

## Requirements

- Python 3.12+
- Dependencies: `networkx`, `plotly` (installed automatically via `pip install -e .`)
- Dev: `pytest` (installed via `pip install -e ".[dev]"`)
- Web app: modern browser with WebAssembly support (Chrome, Firefox, Safari, Edge)
