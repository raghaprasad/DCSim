# DCSim: Datacenter Chaos Engineering Simulator

## HACKATHON MODE — Simplified Scope

Single-command demo: `python -m dcsim.demo` runs 4 scenarios (baseline + 3 chaos), generates an interactive Plotly HTML report, and opens it in the browser.

### What's IN scope
- Engine (Phase 1) — DONE
- Hardware graph with fixed 32-GPU topology (Phase 2)
- AllReduce training workload only (Phase 4, simplified)
- Manual chaos injection only (Phase 5, simplified)
- Event logger (Phase 5, simplified)
- Demo runner + Plotly visualization (NEW)

### What's CUT
- ~~Scheduler module~~ — hardcode: all 32 GPUs assigned to the one training job
- ~~Inference workload~~ — training only
- ~~Failure distributions~~ — manual chaos events only
- ~~API layer (FastAPI/WebSocket)~~ — not needed, direct Python execution
- ~~Frontend (React)~~ — Plotly HTML charts instead
- ~~SchedulerStrategy / ResourcePool~~ — no multi-job, no queuing

## Setup

```bash
cd /Users/raghaprasad/Workspaces/proto
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure
```
src/dcsim/
  engine/              # DONE — do not modify
    clock.py           # SimTime, MILLISECOND, SECOND
    event.py           # Event, EventPayload, EventQueue
    loop.py            # SimulationLoop, SimulationContext
  hardware/            # Agent A
    components.py      # GPU, Switch, Link, state enums
    topology.py        # build_standard_cluster() → 4 nodes x 8 GPUs
    graph.py           # HardwareGraph wrapping NetworkX
  workloads/           # Agent B
    base.py            # Workload ABC, WorkloadManager
    allreduce.py       # AllReduceTraining
  chaos/               # Agent C
    injector.py        # ChaosInjector (manual only)
  observer/            # Agent C
    logger.py          # EventLogger
  demo.py              # Agent D — single-command demo runner
  visualize.py         # Agent D — Plotly chart generation
tests/
  test_engine.py       # DONE — 14 tests passing
  test_hardware.py     # Agent A
  test_workloads.py    # Agent B
  test_integration.py  # Agent D — demo scenario validation
```

## Time Model

All times are integer **microseconds** (`SimTime = int`).
- `MICROSECOND = 1`, `MILLISECOND = 1_000`, `SECOND = 1_000_000`
- Demo uses milliseconds as human unit: "100ms compute" = `100_000` SimTime
- Import from `dcsim.engine.clock`

## Event Priority Tiers

Lower number = fires first at same timestamp.

| Priority | Tier | Examples |
|----------|------|---------|
| 0 | Hardware state transitions | `hardware.gpu.fail`, `hardware.link.fail` |
| 10 | Cascade propagation | `cascade.gpu.job_interrupted`, `cascade.gpu.throttled` |
| 20 | Workload reactions | `workload.phase.start` |
| 30 | Workload progress | `workload.phase.complete`, `workload.step.complete` |
| 40 | Observer/metrics | `system.metrics.sample` |

---

# PHASE 1: ENGINE — COMPLETE

All agents import from `dcsim.engine`. Here is the exact implemented API:

### `dcsim.engine.clock`
```python
SimTime = int  # Microseconds
MICROSECOND: SimTime = 1
MILLISECOND: SimTime = 1_000
SECOND: SimTime = 1_000_000

class SimulationClock:
    def now(self) -> SimTime
    def advance_to(self, t: SimTime) -> None    # Raises if t < now
    def format_time(self, t: SimTime | None = None) -> str
```

### `dcsim.engine.event`
```python
@dataclass
class EventPayload:
    event_type: str
    parent_event_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    def describe(self) -> str

@dataclass(order=True)
class Event:
    time: SimTime
    priority: int
    sequence: int
    event_id: str           # (compare=False)
    payload: EventPayload   # (compare=False)

class EventQueue:
    def schedule(self, time: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def schedule_relative(self, current_time: SimTime, delta: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def cancel(self, event: Event) -> bool
    def pop(self) -> Event | None
    def peek(self) -> Event | None
    def is_empty(self) -> bool
```

### `dcsim.engine.loop`
```python
EventHandler = Callable[[Event, SimulationContext], list[EventPayload] | None]

class SimulationContext:
    clock: SimulationClock
    queue: EventQueue
    # Phases add attributes: hardware, workload_manager, logger

class SimulationLoop:
    clock: SimulationClock      # property
    queue: EventQueue           # property
    context: SimulationContext  # property
    def register_handler(self, event_type: str, handler: EventHandler) -> None
    def schedule(self, time: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def step(self) -> bool
    def run(self, until: SimTime | None = None) -> SimulationResult
```

**Key behavior**: Returned `list[EventPayload]` from handlers are scheduled at *current sim time* with priority 0. `parent_event_id` is auto-set to triggering event's ID.

---

# AGENT A: HARDWARE GRAPH (Phase 2)

**Files**: `src/dcsim/hardware/components.py`, `topology.py`, `graph.py`
**Tests**: `tests/test_hardware.py`
**Depends on**: Phase 1 only

### `components.py` — State enums and component dataclasses

```python
class GPUState(Enum): IDLE, IN_USE, THROTTLED, FAILED
class LinkState(Enum): UP, DEGRADED, DOWN
class SwitchState(Enum): ACTIVE, DEGRADED, FAILED
class LinkType(Enum): NVLINK, INFINIBAND

@dataclass
class HardwareComponent:
    id: str
    component_type: str  # "gpu", "switch", "link"
    state: GPUState | LinkState | SwitchState

class GPU(HardwareComponent):
    node_id: str
    gpu_index: int
    state: GPUState = GPUState.IDLE
    throttle_factor: float = 1.0  # 1.0 = full speed, 0.33 = thermal throttle

class Switch(HardwareComponent):
    tier: int  # 0=ToR, 1=spine
    state: SwitchState = SwitchState.ACTIVE

class Link(HardwareComponent):
    source_id: str
    target_id: str
    link_type: LinkType
    bandwidth_gbps: float      # 7200 for NVLink, 400 for IB
    latency_us: float          # 0.1 for NVLink, 1.0 for IB
    state: LinkState = LinkState.UP
```

### `topology.py` — Fixed 32-GPU cluster

```python
def build_standard_cluster() -> HardwareGraph:
    """4 nodes x 8 GPUs, 2 ToR switches, 2 spine switches.

    Node 0,1 → ToR tor-0 (rack 0)
    Node 2,3 → ToR tor-1 (rack 1)
    tor-0, tor-1 → spine-0, spine-1 (full mesh)

    Intra-node: NVLink all-to-all (7200 Gbps, 0.1us)
    Node-to-ToR: InfiniBand (400 Gbps, 1.0us)
    ToR-to-Spine: InfiniBand (400 Gbps, 1.0us)
    """
```

GPU naming: `"node-{n}/gpu-{g}"` → `node-0/gpu-0` through `node-3/gpu-7`
Switch naming: `"tor-0"`, `"tor-1"`, `"spine-0"`, `"spine-1"`

### `graph.py` — HardwareGraph

```python
class HardwareGraph:
    # Wraps networkx.Graph
    def add_component(self, c: HardwareComponent)
    def add_link(self, link: Link)
    def get_component(self, id: str) -> HardwareComponent
    def get_gpus(self, state: GPUState | None = None) -> list[GPU]
    def get_bandwidth_between(self, src: str, dst: str) -> float  # bottleneck
    def apply_state_change(self, component_id: str, new_state, event: Event, queue: EventQueue, now: SimTime) -> list[Event]
```

**Critical**: `apply_state_change` handles cascades:
- Switch fails → all connected links go DOWN → emit `cascade.link.down` (priority 10)
- GPU fails while IN_USE → emit `cascade.gpu.job_interrupted` (priority 10)
- GPU throttled while IN_USE → emit `cascade.gpu.throttled` (priority 10)
- Link fails → if it isolates GPUs from each other, emit `cascade.gpu.isolated` (priority 10)
- Link repair → recalculate connectivity

**Register event handlers** with the SimulationLoop for:
- `hardware.gpu.fail` → set GPU FAILED, cascade
- `hardware.gpu.repair` → set GPU IDLE
- `hardware.gpu.throttle` → set GPU THROTTLED, update throttle_factor from event data
- `hardware.gpu.unthrottle` → set GPU back to IN_USE, reset throttle_factor=1.0
- `hardware.link.fail` → set Link DOWN, cascade
- `hardware.link.repair` → set Link UP
- `hardware.switch.fail` → set Switch FAILED, cascade connected links

### Gating tests (6 tests)
1. `build_standard_cluster()`: 32 GPUs, 4 switches, correct link counts
2. Any GPU can reach any other GPU (path exists)
3. GPU state transitions: IDLE→IN_USE→THROTTLED→FAILED→IDLE work; invalid transitions raise
4. ToR switch failure: GPUs on that rack lose cross-rack connectivity
5. Spine link failure: reduced bandwidth but connectivity maintained
6. Engine integration: schedule `hardware.gpu.fail` → cascade events fire in priority order

---

# AGENT B: ALLREDUCE WORKLOAD (Phase 4, simplified)

**Files**: `src/dcsim/workloads/base.py`, `allreduce.py`
**Tests**: `tests/test_workloads.py`
**Depends on**: Phase 1 engine + Phase 2 hardware interfaces

### Simplified model (no scheduler)

The WorkloadManager directly assigns ALL GPUs to the training job at t=0. No scheduler module needed.

### `base.py`

```python
class WorkloadPhase(Enum):
    COMPUTE = "compute"
    COMMUNICATE = "communicate"

@dataclass
class Workload(ABC):
    job_id: str
    gpu_ids: list[str]
    current_step: int = 0
    total_steps: int = 10
    state: str = "pending"  # "pending", "running", "interrupted", "completed"

    @abstractmethod
    def get_next_phase(self, graph, now) -> tuple[WorkloadPhase, SimTime] | None
        # Returns (phase_type, duration) or None if done

    @abstractmethod
    def on_gpu_failed(self, gpu_id: str) -> str
        # Returns "abort" or "continue"

class WorkloadManager:
    """Wires workloads into the event loop. No scheduler — directly assigns GPUs."""

    def setup(self, sim: SimulationLoop, workload: Workload, graph: HardwareGraph):
        # 1. Set all workload GPUs to IN_USE
        # 2. Register handlers for workload.phase.complete, cascade.gpu.job_interrupted, cascade.gpu.throttled
        # 3. Schedule first phase at t=0

    # Handler: workload.phase.complete
    #   → advance to next phase, schedule its completion event
    #   → if all steps done, emit workload.job.complete

    # Handler: cascade.gpu.job_interrupted
    #   → cancel pending phase completion event
    #   → set workload state = "interrupted"
    #   → emit workload.job.failed with data including last_step

    # Handler: cascade.gpu.throttled
    #   → cancel pending phase completion event
    #   → recalculate current phase duration with new throttle_factor
    #   → reschedule phase completion at adjusted time

    # Handler: hardware.gpu.repair (for resumption after XID failure)
    #   → if workload is interrupted and all GPUs now healthy
    #   → resume from last_checkpoint_step, schedule first phase
```

### `allreduce.py`

```python
class AllReduceTraining(Workload):
    base_compute_us: SimTime = 100_000   # 100ms default
    comms_duration_us: SimTime = 50_000  # 50ms default (pre-calculated)
    # For hackathon: comms_duration is a direct parameter, not calculated from gradient size

    def get_next_phase(self, graph, now):
        if self.current_step >= self.total_steps:
            return None
        # Alternates: COMPUTE then COMMUNICATE
        # On COMPUTE: duration = base_compute_us / min(throttle_factor)
        # On COMMUNICATE: duration = comms_duration_us
        # After COMMUNICATE completes: current_step += 1
```

**Key simplification**: `comms_duration_us` is a direct config parameter (50ms), not derived from gradient_size / bandwidth. This avoids needing the graph's bandwidth calculations for the basic demo. The graph is still used for throttle_factor lookups.

### Training step cycle
```
COMPUTE (100ms) → COMMUNICATE (50ms) → step++ → COMPUTE → COMMUNICATE → ...
```
Each step = 1 compute + 1 communicate = 150ms baseline.

### Handling chaos events

**Thermal throttle** (Test Case 1):
- `cascade.gpu.throttled` fires → WorkloadManager cancels pending phase completion
- Recalculates: if in COMPUTE phase and GPU throttle_factor=0.33, remaining compute time = `base_compute_us / 0.33` minus time already elapsed
- Reschedules phase completion at new time
- All other GPUs complete their compute normally but must wait for the slow GPU before allreduce starts (training is synchronous)
- **The sync barrier is implicit**: the COMMUNICATE phase only starts after COMPUTE completes, and COMPUTE duration is governed by the slowest GPU

**Link flap** (Test Case 2):
- `hardware.link.fail` during COMMUNICATE phase → cancel pending phase completion
- Comms is blocked until link is restored
- `hardware.link.repair` → resume comms phase, add penalty time for rerouting overhead
- Event data should include `{"reroute_penalty_us": 10_000}` (10ms)

**XID failure** (Test Case 3):
- `cascade.gpu.job_interrupted` → cancel pending phase, workload state="interrupted"
- `hardware.gpu.repair` event scheduled at t+10,000,000us (10s) by chaos injector
- On repair, WorkloadManager resumes workload from current_step (no checkpoint complexity for hackathon — just resume where you left off)

### Gating tests (5 tests)
1. Baseline: 32 GPUs, 10 steps, no chaos → completes at t=1,500,000us (1500ms)
2. Throttle: GPU throttled to 0.33 at t=320ms → compute takes 3x, total > 1500ms
3. Link flap: link down at t=160ms, up at t=200ms → comms delayed, total > 1500ms
4. XID failure: GPU fails at t=460ms → 10s gap → training resumes
5. Step counting: verify exactly 10 step.complete events in baseline

---

# AGENT C: CHAOS INJECTOR & EVENT LOGGER (Phase 5, simplified)

**Files**: `src/dcsim/chaos/injector.py`, `src/dcsim/observer/logger.py`
**Tests**: `tests/test_chaos.py`, `tests/test_observer.py`
**Depends on**: Phase 1 engine only

### `injector.py` — Manual chaos only

```python
@dataclass
class ChaosEvent:
    target_id: str              # "node-1/gpu-4", "tor-0", "link-tor-0-spine-0"
    event_type: str             # "hardware.gpu.fail", "hardware.gpu.throttle", etc.
    time: SimTime               # When to inject
    duration: SimTime | None    # Auto-repair after this (None = permanent)
    properties: dict = field(default_factory=dict)  # {"throttle_factor": 0.33}

class ChaosInjector:
    def inject(self, events: list[ChaosEvent], queue: EventQueue) -> list[Event]:
        # For each ChaosEvent:
        #   1. Schedule the failure event at chaos.time with priority 0
        #   2. Put chaos properties into EventPayload.data
        #   3. If duration is set, also schedule repair event at chaos.time + duration
        # Return all scheduled Events
```

Repair event type mapping:
- `hardware.gpu.fail` → repair is `hardware.gpu.repair`
- `hardware.gpu.throttle` → repair is `hardware.gpu.unthrottle`
- `hardware.link.fail` → repair is `hardware.link.repair`
- `hardware.switch.fail` → repair is `hardware.switch.repair`

### `logger.py` — Simple event logger

```python
@dataclass
class LogEntry:
    timestamp: SimTime
    event_id: str
    parent_event_id: str | None
    event_type: str
    component_id: str | None
    job_id: str | None
    description: str
    data: dict

class EventLogger:
    entries: list[LogEntry]  # Public for easy access

    def make_handler(self) -> EventHandler:
        """Returns a handler that can be registered for ANY event type."""
        # The handler creates a LogEntry from the event and appends it

    def get_timeline(self) -> list[LogEntry]
    def export_json(self) -> str
    def export_dicts(self) -> list[dict]  # For Plotly consumption
```

The logger registers a SINGLE handler for ALL event types. To do this, register it for each event type used in the simulation, OR modify the SimulationLoop to support a wildcard/catch-all handler.

**Simpler approach**: The WorkloadManager and HardwareGraph call `logger.log(entry)` directly after processing events, rather than the logger being an event handler. This avoids the catch-all registration problem.

### Gating tests (4 tests)
1. ChaosInjector: schedule GPU fail at t=1000 → event fires at t=1000
2. ChaosInjector: fail with duration=5000 → fail event + repair event scheduled
3. EventLogger: log 5 entries → get_timeline returns them in order
4. EventLogger: export_dicts returns list of dicts suitable for DataFrame construction

---

# AGENT D: DEMO RUNNER & VISUALIZATION

**Files**: `src/dcsim/demo.py`, `src/dcsim/visualize.py`
**Tests**: `tests/test_integration.py`
**Depends on**: ALL agents above must be complete

### `demo.py` — Entry point

```bash
python -m dcsim.demo              # Run all 4 scenarios, open HTML report
python -m dcsim.demo --scenario baseline   # Run just one
```

Each scenario:
1. Build the standard 32-GPU cluster
2. Create AllReduceTraining workload (10 steps, 100ms compute, 50ms comms)
3. Wire everything into SimulationLoop
4. Inject chaos events (if any)
5. Run simulation
6. Collect logs
7. Pass to visualizer

### `visualize.py` — Plotly HTML report

Generate a single HTML file with interactive Plotly charts:

1. **GPU State Timeline** (Gantt/heatmap): Each GPU is a row, colored by state over time (green=computing, blue=communicating, yellow=throttled, red=failed, gray=idle/waiting). This is the money shot — visually shows 31 GPUs going gray while 1 is yellow.

2. **Iteration Timeline** (bar chart): Each iteration as a bar showing compute + comms duration. Chaos-affected iterations visibly longer.

3. **Event Log Table**: Timestamped table of key events (failures, recoveries, phase starts/completions).

4. **Comparison View**: Side-by-side baseline vs chaos scenario showing iteration times.

The HTML auto-opens in the browser via `webbrowser.open()`.

**Add `plotly` to pyproject.toml dependencies.**

### Demo scenarios (hardcoded in demo.py)

```python
SCENARIOS = {
    "baseline": [],  # No chaos events

    "thermal_throttle": [
        ChaosEvent("node-1/gpu-4", "hardware.gpu.throttle", 320 * MILLISECOND,
                   duration=None, properties={"throttle_factor": 0.33})
    ],

    "link_flap": [
        ChaosEvent("link-tor-0-spine-0", "hardware.link.fail", 160 * MILLISECOND,
                   duration=40 * MILLISECOND)  # Recovers at t=200ms
    ],

    "xid_79": [
        ChaosEvent("node-3/gpu-1", "hardware.gpu.fail", 460 * MILLISECOND,
                   duration=10 * SECOND)  # 10s repair time
    ],
}
```

### Integration tests validate demo output

```python
class TestBaseline:
    # Run baseline scenario
    # Assert: 10 steps completed, total time ≈ 1,500,000us

class TestThermalThrottle:
    # Assert: total time > 1,500,000us
    # Assert: events show GPU throttled at t=320ms
    # Assert: iteration 3 compute phase > 100ms

class TestLinkFlap:
    # Assert: total time > 1,500,000us
    # Assert: events show link down at t=160ms, up at t=200ms

class TestXID79:
    # Assert: total time > 11,500,000us (baseline + ~10s penalty)
    # Assert: events show GPU fail at t=460ms, repair at t≈10,460ms
```

---

# EVENT TYPE REGISTRY (simplified)

| Event Type | Priority | Source → Consumer |
|---|---|---|
| `hardware.gpu.fail` | 0 | Chaos → Hardware |
| `hardware.gpu.repair` | 0 | Chaos → Hardware, Workloads |
| `hardware.gpu.throttle` | 0 | Chaos → Hardware |
| `hardware.gpu.unthrottle` | 0 | Chaos → Hardware |
| `hardware.link.fail` | 0 | Chaos → Hardware |
| `hardware.link.repair` | 0 | Chaos → Hardware, Workloads |
| `hardware.switch.fail` | 0 | Chaos → Hardware |
| `cascade.gpu.job_interrupted` | 10 | Hardware → Workloads |
| `cascade.gpu.throttled` | 10 | Hardware → Workloads |
| `cascade.link.down` | 10 | Hardware → (logged) |
| `workload.phase.start` | 30 | Workloads → (logged) |
| `workload.phase.complete` | 30 | Workloads → Workloads |
| `workload.step.complete` | 30 | Workloads → (logged) |
| `workload.job.complete` | 30 | Workloads → (logged) |
| `workload.job.failed` | 30 | Workloads → (logged) |

---

# CONVENTIONS

- All source: `src/dcsim/`, tests: `tests/`
- Run tests: `source .venv/bin/activate && pytest tests/ -v`
- GPU naming: `"node-{n}/gpu-{g}"` (node-0/gpu-0 through node-3/gpu-7)
- Switch naming: `"tor-0"`, `"tor-1"`, `"spine-0"`, `"spine-1"`
- Link naming: `"link-{source}-{target}"`
- Event types: dotted namespace `"hardware.gpu.fail"`
- All times in integer microseconds
- `SimulationContext` has no `__slots__` — phases add attributes freely
