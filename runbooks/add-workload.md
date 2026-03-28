# Adding a New Workload Type

## Overview
How to add a new AI workload (e.g., inference, pipeline parallelism, data loading) to the simulator. Workloads define the phase pattern (compute/communicate cycle) and how they react to hardware failures.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/workloads/new_workload.py` | New file subclassing `Workload` ABC | Defines phase cycle and failure behavior |
| `src/dcsim/workloads/base.py` | Possibly new `WorkloadPhase` enum values | If the workload has phases beyond COMPUTE/COMMUNICATE |
| `src/dcsim/workloads/base.py` | WorkloadManager changes (if failure semantics differ) | E.g., inference degrades instead of aborting |
| `src/dcsim/workloads/__init__.py` | Export the new class | So it's importable |

## Step-by-Step

### 1. Create the workload file

Follow the `allreduce.py` pattern:

```python
# src/dcsim/workloads/inference.py
from dataclasses import dataclass, field
from dcsim.engine.clock import SimTime
from dcsim.workloads.base import Workload, WorkloadPhase

@dataclass
class InferenceWorkload(Workload):
    """Inference serving: N replicas, each GPU handles requests independently."""

    base_compute_us: SimTime = 50_000   # 50ms per request
    active_replicas: int = 0
    min_replicas: int = 1

    def __post_init__(self):
        self.active_replicas = len(self.gpu_ids)

    def get_next_phase(self, gpu_states, now):
        # Inference is continuous COMPUTE phases (no allreduce)
        if self.state != "running":
            return None
        # Each "step" is one batch of requests served
        if self.current_step >= self.total_steps:
            return None
        min_throttle = min(
            s.get("throttle_factor", 1.0) for s in gpu_states.values()
            if not s.get("failed", False)
        )
        duration = int(self.base_compute_us / min_throttle)
        return (WorkloadPhase.COMPUTE, duration)

    def on_gpu_failed(self, gpu_id):
        self.active_replicas -= 1
        if self.active_replicas >= self.min_replicas:
            return "continue"  # Degraded but running
        return "abort"
```

### 2. Implement the two required abstract methods

Every `Workload` subclass MUST implement:

```python
def get_next_phase(self, gpu_states: dict, now: SimTime) -> tuple[WorkloadPhase, SimTime] | None:
    """Return (phase_type, duration_us) or None if done."""

def on_gpu_failed(self, gpu_id: str) -> str:
    """Return 'abort' or 'continue'."""
```

### 3. Add new WorkloadPhase values if needed (`base.py`)

If your workload has a phase that isn't COMPUTE or COMMUNICATE:

```python
class WorkloadPhase(Enum):
    COMPUTE = "compute"
    COMMUNICATE = "communicate"
    PIPELINE_BUBBLE = "pipeline_bubble"  # ← new
```

### 4. Handle different failure semantics in WorkloadManager (`base.py`)

Currently, `_handle_gpu_interrupted` always aborts. If your workload returns "continue":

```python
def _handle_gpu_interrupted(self, event, ctx):
    reaction = self._workload.on_gpu_failed(gpu_id)
    if reaction == "abort":
        # Cancel pending, set interrupted
        ...
    elif reaction == "continue":
        # Recalculate with fewer active GPUs, keep running
        # Remove failed GPU from active set
        # Reschedule current phase with adjusted timing
        ...
```

This is the main place where WorkloadManager behavior varies by workload type.

### 5. Wire into demo scenarios

Add the workload as an option in `demo.py` and add a test in `test_integration.py`.

## Phase Cycle Patterns

| Workload | Phase cycle | Notes |
|----------|------------|-------|
| AllReduce Training | COMPUTE → COMMUNICATE → step++ → repeat | Sync barrier: slowest GPU dominates |
| Inference | COMPUTE → step++ → repeat | No communication. Independent replicas. |
| Pipeline Parallel | COMPUTE → COMMUNICATE → PIPELINE_BUBBLE → step++ | Bubble time = pipeline depth overhead |

## Checklist
- [ ] New file subclasses `Workload` ABC
- [ ] `get_next_phase()` returns correct phase/duration cycle
- [ ] `on_gpu_failed()` returns appropriate reaction
- [ ] WorkloadManager handles the reaction (if different from default abort)
- [ ] Exported in `__init__.py`
- [ ] Tests: baseline completion, failure behavior, step counting
- [ ] `pytest tests/ -v` passes
