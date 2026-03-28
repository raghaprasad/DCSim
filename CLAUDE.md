# DCSim: Datacenter Chaos Engineering Simulator

## Project Overview

A discrete event simulator modeling datacenter hardware failures and their impact on AI training/inference workloads. Python 3.13+ backend, React/TypeScript frontend.

## Setup

```bash
cd /Users/raghaprasad/Workspaces/proto
source .venv/bin/activate
pip install -e ".[dev]"        # Core + test deps
pip install -e ".[api]"        # Add FastAPI/WebSocket deps (Phase 6)
pytest tests/ -v               # Run all tests
```

## Architecture

Six backend layers, each event-driven, communicating only through the engine's event system:

```
Engine (Phase 1) ← COMPLETE
  ↓
Hardware Graph (Phase 2)
  ↓
Scheduler (Phase 3)         Chaos + Observer (Phase 5)
  ↓                           ↓
Workloads (Phase 4)         (hooks into engine)
  ↓
API (Phase 6) → Frontend (Phase 7)
```

## Time Model

All times are integer **microseconds** (`SimTime = int`). Use constants from `dcsim.engine.clock`:
- `MICROSECOND = 1`
- `MILLISECOND = 1_000`
- `SECOND = 1_000_000`

The demo scenarios use **milliseconds** as the human unit. So "100ms compute phase" = `100 * MILLISECOND` = `100_000` in SimTime.

## Event Priority Tiers

Events at the same timestamp fire in priority order. **Lower number = fires first.**

| Priority | Tier | Examples |
|----------|------|---------|
| 0 | Hardware state transitions | `hardware.gpu.fail`, `hardware.link.fail` |
| 10 | Cascade propagation | `cascade.link.down`, `cascade.gpu.isolated` |
| 20 | Scheduler reactions | `scheduler.job.start`, `scheduler.job.interrupt` |
| 30 | Workload state changes | `workload.phase.complete`, `workload.step.complete` |
| 40 | Observer/metrics | `system.metrics.sample` |

---

# PHASE 1: ENGINE — COMPLETE (reference for all agents)

All phases import from `dcsim.engine`. Here is the exact API:

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
    parent_event_id: str | None = None     # Causal chain link
    data: dict[str, Any] = field(default_factory=dict)
    def describe(self) -> str

@dataclass(order=True)
class Event:
    time: SimTime
    priority: int
    sequence: int
    event_id: str           # (compare=False) unique ID
    payload: EventPayload   # (compare=False)

class EventQueue:
    def schedule(self, time: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def schedule_relative(self, current_time: SimTime, delta: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def cancel(self, event: Event) -> bool
    def pop(self) -> Event | None
    def peek(self) -> Event | None
    def is_empty(self) -> bool
    def __len__(self) -> int
```

### `dcsim.engine.loop`
```python
EventHandler = Callable[[Event, SimulationContext], list[EventPayload] | None]

class SimulationContext:
    clock: SimulationClock   # read-only
    queue: EventQueue

class SimulationResult:
    events_processed: int
    final_time: SimTime
    stopped_by_max_time: bool
    stopped_by_empty_queue: bool

class SimulationLoop:
    clock: SimulationClock      # property
    queue: EventQueue           # property
    context: SimulationContext  # property

    def register_handler(self, event_type: str, handler: EventHandler) -> None
    def schedule(self, time: SimTime, payload: EventPayload, priority: int = 0) -> Event
    def step(self) -> bool
    def run(self, until: SimTime | None = None) -> SimulationResult
    def pause(self) -> None
```

**Key behavior**: When a handler returns `list[EventPayload]`, those payloads are scheduled at the *current sim time* with priority 0. Their `parent_event_id` is auto-set to the triggering event's ID if not already set. For future-time events, schedule directly via `ctx.queue.schedule(...)`.

---

# PHASE 2: HARDWARE GRAPH & STATE MACHINES

**Package**: `dcsim.hardware` — `components.py`, `topology.py`, `graph.py`
**Depends on**: Phase 1 (engine) only
**Tests**: `tests/test_hardware.py`

### What to build

**`components.py`** — Hardware component state machines

State enums:
- `GPUState`: IDLE, IN_USE, THROTTLED, FAILED
- `LinkState`: UP, DEGRADED, DOWN
- `SwitchState`: ACTIVE, DEGRADED, FAILED
- `LinkType`: NVLINK (intra-node), INFINIBAND (inter-node)

Component dataclasses (all inherit from `HardwareComponent`):
- `GPU`: id, node_id, gpu_index, state=IDLE, compute_tflops=312.0, memory_gb=80.0, throttle_factor=1.0
- `Link`: id, source_id, target_id, link_type, bandwidth_gbps, latency_us, state=UP
- `Switch`: id, tier (0=ToR, 1=spine), port_count=64, state=ACTIVE

State machine: define valid transitions per component type. Each transition has:
- from_state, to_state, triggering event_type
- Optional side_effects: list of event types to emit

Key GPU transitions:
- IDLE→IN_USE via `scheduler.gpu.allocate`
- IN_USE→IDLE via `scheduler.gpu.release`
- any→FAILED via `hardware.gpu.fail` (side effect: `cascade.gpu.job_interrupted` if was IN_USE)
- FAILED→IDLE via `hardware.gpu.repair`
- IN_USE→THROTTLED via `hardware.gpu.throttle` (side effect: `cascade.gpu.throttled`)
- THROTTLED→IN_USE via `hardware.gpu.unthrottle`

**`topology.py`** — Topology builders

`TopologyBuilder.single_node(num_gpus=8)` → HardwareGraph
- 1 node, N GPUs, all-to-all NVLink connections

`TopologyBuilder.fat_tree(num_nodes=8, gpus_per_node=8)` → HardwareGraph
- Intra-node: NVLink all-to-all (bandwidth=7200 Gbps, latency=0.1us)
- Each node has a NIC connecting to a ToR switch via InfiniBand (400 Gbps, 1.0us)
- ToR switches connect to spine switches
- For 4 nodes: 1 ToR per node, 2 spine switches, full mesh at spine tier
- For 8 nodes: 2 racks of 4, each rack has 1 ToR, 2 spine switches

GPU naming: `"node-{n}/gpu-{g}"` (e.g., `"node-0/gpu-3"`)
Switch naming: `"tor-{r}"`, `"spine-{s}"`
Link naming: `"link-{source}-{target}"`

**`graph.py`** — HardwareGraph wrapping NetworkX

```python
class HardwareGraph:
    def add_component(self, component: HardwareComponent) -> None
    def add_link(self, link: Link) -> None
    def get_component(self, id: str) -> HardwareComponent
    def get_gpus(self, state: GPUState | None = None) -> list[GPU]
    def get_available_gpus(self) -> list[GPU]  # state == IDLE
    def get_links_for(self, component_id: str) -> list[Link]
    def get_path(self, src: str, dst: str) -> list[str]
    def get_operational_path(self, src: str, dst: str) -> list[str] | None
    def get_bandwidth_between(self, src: str, dst: str) -> float  # bottleneck bw
    def get_affected_components(self, failed_id: str) -> list[HardwareComponent]
    def apply_failure(self, component_id: str, queue: EventQueue, now: SimTime) -> list[Event]
    def apply_state_change(self, component_id: str, new_state, queue: EventQueue, now: SimTime) -> list[Event]
```

### Gating tests (6 tests)
1. Single node: 8 GPUs, correct links, all paths exist
2. Fat-tree: 4 nodes, correct switch/link counts, full connectivity
3. GPU state machine: valid transitions work, invalid raise errors
4. Switch failure cascade: ToR fails → GPUs on that node lose external connectivity
5. Link failure: one spine link down → reduced bandwidth, connectivity maintained
6. Engine integration: schedule switch_fail event → cascade events fire in correct priority order

---

# PHASE 3: ORCHESTRATION LAYER

**Package**: `dcsim.scheduler` — `scheduler.py`, `allocation.py`
**Depends on**: Phase 1 (engine), Phase 2 (hardware interfaces)
**Tests**: `tests/test_scheduler.py`

### What to build

**`allocation.py`** — Resource pool
```python
@dataclass
class GPUAllocation:
    job_id: str
    gpu_ids: list[str]
    allocated_at: SimTime
    released_at: SimTime | None = None

class ResourcePool:
    def allocate(self, job_id: str, gpu_ids: list[str], now: SimTime) -> GPUAllocation
    def release(self, job_id: str, now: SimTime) -> list[str]
    def get_job_for_gpu(self, gpu_id: str) -> str | None
    def get_allocation(self, job_id: str) -> GPUAllocation | None
    def get_free_gpu_ids(self, graph: HardwareGraph) -> list[str]
```

**`scheduler.py`** — Event-driven scheduler
```python
@dataclass
class JobRequest:
    job_id: str
    workload_type: str        # "allreduce_training", "inference"
    gpu_count: int
    priority: int = 0
    submitted_at: SimTime = 0
    workload_config: dict = field(default_factory=dict)

class SchedulerStrategy(Protocol):
    def select_gpus(self, job: JobRequest, pool: ResourcePool, graph: HardwareGraph) -> list[str] | None

class SimpleScheduler(SchedulerStrategy):
    # FIFO, best-fit: prefer GPUs on same node, then same rack

class Scheduler:
    # Registers these event handlers with SimulationLoop:
    # "scheduler.job.submit"     → try allocate → emit "scheduler.job.start" or queue
    # "scheduler.gpu.available"  → try schedule pending
    # "cascade.gpu.job_interrupted" → release allocation, re-queue job
    # "workload.job.complete"    → release GPUs, try schedule pending
```

The scheduler emits `scheduler.gpu.allocate` events (priority 20) that the hardware layer uses to transition GPUs IDLE→IN_USE. It emits `scheduler.job.start` (priority 20) that the workload layer listens for.

### Gating tests (5 tests)
1. Submit 4-GPU job on 8-GPU node → 4 GPUs become IN_USE
2. Two 8-GPU jobs on 8-GPU system → first starts, second queues, first completes → second starts
3. GPU failure during job → job_interrupted fires, GPUs released
4. After interruption → job re-queued and starts when GPUs available
5. Multi-node: scheduler prefers co-located GPUs

---

# PHASE 4: WORKLOADS

**Package**: `dcsim.workloads` — `base.py`, `allreduce.py`, `inference.py`
**Depends on**: Phase 1 (engine), Phase 2 (hardware), Phase 3 (scheduler)
**Tests**: `tests/test_workloads.py`

### What to build

**`base.py`** — Workload ABC and WorkloadManager
```python
class WorkloadPhase(Enum):
    COMPUTE = "compute"
    COMMUNICATE = "communicate"
    CHECKPOINT = "checkpoint"

@dataclass
class PhaseSpec:
    phase_type: WorkloadPhase
    duration_us: SimTime          # base duration before adjustments
    gpu_ids: list[str] | None = None
    payload_bytes: int = 0

class Workload(ABC):
    job_id: str
    gpu_ids: list[str]
    current_step: int = 0
    total_steps: int
    state: str  # "pending", "running", "interrupted", "completed", "failed"

    @abstractmethod
    def get_next_phase(self, graph: HardwareGraph, now: SimTime) -> PhaseSpec | None
    @abstractmethod
    def on_gpu_failed(self, gpu_id: str, graph: HardwareGraph, now: SimTime) -> WorkloadReaction
    @abstractmethod
    def compute_phase_duration(self, phase: PhaseSpec, graph: HardwareGraph) -> SimTime

@dataclass
class WorkloadReaction:
    action: str  # "abort", "degrade"
    events: list[EventPayload] = field(default_factory=list)

class WorkloadManager:
    # Event handlers registered with SimulationLoop:
    # "scheduler.job.start"      → create Workload, schedule first phase
    # "workload.phase.complete"  → advance step, schedule next phase or emit job_complete
    # "cascade.gpu.job_interrupted" → cancel pending phase, call workload.on_gpu_failed
```

**`allreduce.py`** — Training workload

Each training step = COMPUTE (forward) → COMMUNICATE (allreduce) → COMPUTE (backward). Every K steps: CHECKPOINT.

```python
class AllReduceTraining(Workload):
    base_compute_us: SimTime        # User-configurable per-step compute time
    gradient_size_bytes: int
    checkpoint_interval: int = 0    # 0 = no checkpointing
    checkpoint_duration_us: SimTime = 0
    last_checkpoint_step: int = 0   # resume point after failure

    # compute_phase_duration:
    #   COMPUTE → base_compute_us / min(throttle_factor) across all GPUs
    #   COMMUNICATE → 2 * (N-1)/N * gradient_size_bytes * 8 / bottleneck_bandwidth_gbps / 1000
    #   CHECKPOINT → checkpoint_duration_us
    #
    # on_gpu_failed → abort (training needs all GPUs)
    # On reschedule: resume from last_checkpoint_step
```

**`inference.py`** — Inference workload
```python
class InferenceWorkload(Workload):
    base_compute_us: SimTime
    active_replicas: int
    min_replicas: int = 1
    # Runs until simulation time limit
    # on_gpu_failed → degrade if active_replicas >= min_replicas, else abort
```

### Key timing formulas

**Compute phase** (training, per-step):
```
actual_duration = base_compute_us / min(gpu.throttle_factor for gpu in allocated_gpus)
```
Training is synchronous — slowest GPU dominates. A GPU with throttle_factor=0.33 makes compute take 3x longer.

**AllReduce communication phase**:
```
actual_duration = 2 * (N-1)/N * gradient_size_bytes * 8 / bottleneck_bandwidth_gbps / 1000
```
Where bottleneck_bandwidth = minimum bandwidth along any path between GPU pairs. Units: gradient in bytes, bandwidth in Gbps, result in microseconds.

### Gating tests (6 tests)
1. AllReduce timing: 8 GPUs, known bandwidth → duration matches formula
2. 10-step training on healthy hardware → 10 steps complete, progress 0→1
3. GPU failure mid-compute → job aborts, pending phase cancelled
4. Throttled GPU (0.5x) → compute phases take 2x longer
5. Inference: 4 replicas, fail 1 GPU → degrades to 3, continues
6. Inference: 4 replicas, min=3, fail 2 → abort

---

# PHASE 5: CHAOS MONKEY & OBSERVER

**Package**: `dcsim.chaos` — `injector.py`, `distributions.py`; `dcsim.observer` — `logger.py`, `metrics.py`
**Depends on**: Phase 1 (engine), Phase 2 (hardware)
**Tests**: `tests/test_chaos.py`, `tests/test_observer.py`

### What to build

**`injector.py`** — Chaos monkey
```python
@dataclass
class ChaosEvent:
    target_id: str           # Component ID (e.g., "node-2/gpu-4")
    event_type: str          # "hardware.gpu.fail", "hardware.gpu.throttle", etc.
    time: SimTime            # When to inject
    duration: SimTime | None = None  # Auto-repair after duration (None = permanent)
    properties: dict = field(default_factory=dict)  # e.g., {"throttle_factor": 0.33}

class ChaosInjector:
    def inject_manual(self, events: list[ChaosEvent], queue: EventQueue) -> list[Event]
        # Schedule each ChaosEvent into the sim queue
        # If duration set, also schedule the corresponding repair event
    def inject_from_distribution(self, dist, components, time_range, queue, rng) -> list[Event]
```

**`distributions.py`** — Failure distributions
```python
class FailureDistribution(Protocol):
    def sample_next_failure_time(self, rng: random.Random, now: SimTime) -> SimTime
    def sample_repair_time(self, rng: random.Random) -> SimTime

class ExponentialFailure(FailureDistribution):
    mtbf_us: SimTime
    mttr_us: SimTime
```

**`logger.py`** — Event logger with causal chain tracking
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
    category: str  # "hardware", "scheduler", "workload", "chaos", "system"

class EventLogger:
    def log(self, entry: LogEntry) -> None
    def get_timeline(self, start: SimTime = 0, end: SimTime | None = None) -> list[LogEntry]
    def get_causal_chain(self, event_id: str) -> list[LogEntry]
    def get_by_job(self, job_id: str) -> list[LogEntry]
    def get_by_component(self, component_id: str) -> list[LogEntry]
    def export_json(self) -> str
```

**`metrics.py`** — Time-series metrics
```python
class MetricsCollector:
    def record(self, name: str, time: SimTime, value: float) -> None
    def get_series(self, name: str) -> list[tuple[SimTime, float]]
    # Built-in: gpu_utilization, network_health, queue_depth, training_progress.<job_id>
```

The EventLogger and MetricsCollector register as event handlers. The logger captures every event. The MetricsCollector samples periodically via `system.metrics.sample` events.

### Gating tests (7 tests)
1. Manual injection: GPU fail at t=1000 → fires at t=1000 with cascade
2. Auto-repair: fail with duration=5000 → fail at t=1000, repair at t=6000
3. Distribution: ExponentialFailure(mtbf=1e6) sampled 1000 times → mean ≈ 1e6
4. Determinism: same seed twice → identical event logs
5. Logging completeness: every event logged with correct timestamp/category
6. Causal chain: switch failure → trace includes link failures, GPU isolation, job interruption
7. Metrics: GPU failures → gpu_utilization drops at failure times

---

# PHASE 6: API LAYER

**Package**: `dcsim.api` — `server.py`, `routes.py`, `ws.py`, `models.py`
**Depends on**: All phases above
**Tests**: `tests/test_api.py`

### REST endpoints
```
POST   /api/sessions                    Create session
GET    /api/sessions                    List sessions
GET    /api/sessions/{id}               Session details
DELETE /api/sessions/{id}               Delete session
POST   /api/sessions/{id}/configure     Set topology + workload + chaos
POST   /api/sessions/{id}/run           Start simulation
POST   /api/sessions/{id}/step          Single event step
POST   /api/sessions/{id}/pause         Pause
POST   /api/sessions/{id}/resume        Resume
GET    /api/sessions/{id}/state         Hardware graph snapshot
GET    /api/sessions/{id}/timeline      Event timeline (paginated)
GET    /api/sessions/{id}/metrics       Metrics time-series
POST   /api/sessions/{id}/chaos         Inject chaos mid-run
GET    /api/topologies                  Topology presets
GET    /api/workloads                   Workload types
```

### WebSocket — `WS /api/sessions/{id}/ws`
Server → Client: `event`, `metrics_snapshot`, `state_snapshot`, `simulation_status`
Client → Server: `subscribe`, `inject_chaos`

---

# EVENT TYPE REGISTRY

| Event Type | Priority | Emitted by | Consumed by |
|---|---|---|---|
| `hardware.gpu.fail` | 0 | Chaos | Hardware, Scheduler |
| `hardware.gpu.repair` | 0 | Chaos | Hardware, Scheduler |
| `hardware.gpu.throttle` | 0 | Chaos | Hardware, Workloads |
| `hardware.gpu.unthrottle` | 0 | Chaos | Hardware, Workloads |
| `hardware.switch.fail` | 0 | Chaos | Hardware |
| `hardware.switch.repair` | 0 | Chaos | Hardware |
| `hardware.link.fail` | 0 | Chaos | Hardware |
| `hardware.link.repair` | 0 | Chaos | Hardware |
| `cascade.link.down` | 10 | Hardware | Observer |
| `cascade.gpu.isolated` | 10 | Hardware | Scheduler |
| `cascade.gpu.job_interrupted` | 10 | Hardware | Scheduler, Workloads |
| `cascade.gpu.throttled` | 10 | Hardware | Workloads |
| `scheduler.job.submit` | 20 | API/Config | Scheduler |
| `scheduler.job.start` | 20 | Scheduler | Workloads |
| `scheduler.job.interrupt` | 20 | Scheduler | Workloads, Observer |
| `scheduler.gpu.allocate` | 20 | Scheduler | Hardware |
| `scheduler.gpu.release` | 20 | Scheduler | Hardware |
| `scheduler.gpu.available` | 20 | Hardware | Scheduler |
| `workload.phase.start` | 30 | Workloads | Observer |
| `workload.phase.complete` | 30 | Workloads | Workloads (self) |
| `workload.step.complete` | 30 | Workloads | Observer |
| `workload.job.complete` | 30 | Workloads | Scheduler, Observer |
| `workload.job.failed` | 30 | Workloads | Scheduler, Observer |
| `system.metrics.sample` | 40 | Metrics | Metrics (self) |

---

# DEMO SCENARIOS & E2E TESTS

These are the target demo scenarios. The integration test file (`tests/test_integration.py`) must validate all of them.

## Baseline Configuration
- 32 GPUs (4 nodes x 8 GPUs)
- 10 training iterations
- Each iteration: 100ms compute + 50ms allreduce comms = 150ms/iteration
- Baseline total: 1500ms (1.5s) for 10 iterations with no failures

## Test Case 1: Thermal Throttle
- Config: 32 GPUs, 10 iterations, 100ms compute, 50ms comms
- Chaos: at t=320ms (during compute phase of iteration 3), GPU `node-1/gpu-4` gets thermal throttle with throttle_factor=0.33 (3x slowdown)
- Expected: GPU 12 (node-1/gpu-4) compute takes 300ms instead of 100ms. Other 31 GPUs finish their compute at t=400ms and idle waiting for GPU 12 until ~t=480ms. The allreduce for iteration 3 starts late. Total time > 1500ms baseline
- Validate: logs show 31 GPUs idle-waiting between t=400ms and t≈480ms

## Test Case 2: Link Flap
- Config: 32 GPUs, 10 iterations, 100ms compute, 50ms comms
- Chaos: at t=160ms (during comms phase of iteration 1), rack-0 (node-0, node-1) uplink fails. Recovers at t=200ms
- Expected: During t=160-200ms, rack-0 is isolated — allreduce cannot complete. After recovery, routing recalculation adds overhead. Comms phase takes longer than 50ms baseline
- Validate: iteration 1 total > 150ms baseline

## Test Case 3: XID 79 (GPU Hard Failure)
- Config: 32 GPUs, 10 iterations, 100ms compute, 50ms comms
- Chaos: at t=460ms, GPU `node-3/gpu-1` (G25) has XID error → hardware.gpu.fail
- Expected: GPU fails → job interrupted → scheduler must find replacement GPU → 10,000ms (10s) penalty for detection + reboot + reload. Training resumes from last checkpoint after penalty
- Validate: logs show ~10,000ms gap between failure and training resumption

## Integration Test Structure
```python
# tests/test_integration.py

class TestBaseline:
    """32 GPUs, 10 iterations, no failures. Total ≈ 1500ms."""

class TestThermalThrottle:
    """Thermal throttle at t=320ms. Validate GPU idle-wait behavior."""

class TestLinkFlap:
    """Link flap at t=160ms. Validate degraded comms phase."""

class TestXID79:
    """GPU hard failure at t=460ms. Validate 10s penalty and reschedule."""
```

---

# SIMCONTEXT EXTENSION PATTERN

`SimulationContext` currently has `clock` and `queue`. Each phase extends it:

Phase 2 adds: `hardware: HardwareGraph`
Phase 3 adds: `scheduler: Scheduler`
Phase 4 adds: `workload_manager: WorkloadManager`
Phase 5 adds: `logger: EventLogger`, `metrics: MetricsCollector`

Each phase should modify `SimulationContext` to add its attribute. Use simple attribute assignment — no need for subclassing. Update `__slots__` or convert to a regular class if needed.

---

# CONVENTIONS

- All source in `src/dcsim/`, tests in `tests/`
- Run tests: `source .venv/bin/activate && pytest tests/ -v`
- GPU naming: `"node-{n}/gpu-{g}"` (e.g., `"node-0/gpu-3"`, `"node-3/gpu-7"`)
- Switch naming: `"tor-{n}"` for ToR, `"spine-{s}"` for spine
- Link naming: `"link-{source}-{target}"`
- Event types use dotted namespace: `"hardware.gpu.fail"`, `"scheduler.job.start"`
- All randomness via seeded `random.Random` (never `random.random()`)
- Every handler receives `(event: Event, ctx: SimulationContext)` and returns `list[EventPayload] | None`
