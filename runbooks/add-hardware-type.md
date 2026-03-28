# Adding a New Hardware Component Type

## Overview
How to add a new hardware component (e.g., TPU, DPU, Storage Node, NIC) to the simulator. A hardware type is anything that has state, can fail, and participates in the datacenter topology.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/hardware/components.py` | Add state enum + dataclass | Defines the component's states and properties |
| `src/dcsim/hardware/topology.py` | Add component to topology builders | So clusters include the new hardware |
| `src/dcsim/hardware/graph.py` | Add cascade logic for failures | Defines what happens when this component fails |
| `src/dcsim/workloads/base.py` | Add event handlers in WorkloadManager | So workloads react to failures of this component |
| `src/dcsim/chaos/injector.py` | Add repair-event mapping | So chaos injector can auto-schedule repairs |
| `CLAUDE.md` | Add to event type registry table | Documents the new event types |

## Step-by-Step

### 1. Define the state enum and dataclass (`hardware/components.py`)

Follow the existing GPU/Switch/Link pattern:

```python
class TPUState(Enum):
    IDLE = "idle"
    IN_USE = "in_use"
    FAILED = "failed"

@dataclass
class TPU(HardwareComponent):
    node_id: str
    tpu_index: int
    component_type: str = "tpu"
    state: TPUState = TPUState.IDLE
    compute_tflops: float = 275.0
    memory_gb: float = 32.0
```

### 2. Add to topology builders (`hardware/topology.py`)

Add the component to whichever topology builders should include it:

```python
def build_standard_cluster() -> HardwareGraph:
    # ... existing GPU/Switch/Link creation ...

    # Add TPUs to each node
    for n in range(num_nodes):
        tpu = TPU(
            id=f"node-{n}/tpu-0",
            node_id=f"node-{n}",
            tpu_index=0,
        )
        graph.add_component(tpu)
        # Add links connecting TPU to the node's switch
```

### 3. Add cascade logic (`hardware/graph.py`)

Register event handlers for the new hardware event types:

```python
# In HardwareGraph.__init__ or setup method:
sim.register_handler("hardware.tpu.fail", self._handle_tpu_fail)
sim.register_handler("hardware.tpu.repair", self._handle_tpu_repair)

def _handle_tpu_fail(self, event, ctx):
    component = self.get_component(event.payload.data["component_id"])
    component.state = TPUState.FAILED
    # If TPU was IN_USE, emit cascade event
    if old_state == TPUState.IN_USE:
        return [EventPayload(
            event_type="cascade.tpu.job_interrupted",
            data={"tpu_id": component.id},
        )]
```

### 4. Add workload reaction (`workloads/base.py`)

Register a handler in WorkloadManager so workloads react to the new failure:

```python
# In WorkloadManager.setup():
sim.register_handler("hardware.tpu.fail", self._handle_tpu_fail)
sim.register_handler("hardware.tpu.repair", self._handle_tpu_repair)
sim.register_handler("cascade.tpu.job_interrupted", self._handle_tpu_interrupted)
```

### 5. Add repair mapping (`chaos/injector.py`)

So `ChaosInjector.inject()` knows how to auto-schedule repairs:

```python
REPAIR_EVENT_MAP = {
    "hardware.gpu.fail": "hardware.gpu.repair",
    "hardware.gpu.throttle": "hardware.gpu.unthrottle",
    "hardware.link.fail": "hardware.link.repair",
    "hardware.switch.fail": "hardware.switch.repair",
    "hardware.tpu.fail": "hardware.tpu.repair",        # ← add this
}
```

### 6. Update CLAUDE.md event registry

Add new rows to the event type registry table.

## Checklist
- [ ] State enum has all valid states
- [ ] Dataclass has all relevant performance properties
- [ ] Topology builder creates and connects the component
- [ ] Cascade logic handles failure propagation
- [ ] WorkloadManager has handlers for the new events
- [ ] ChaosInjector repair mapping is updated
- [ ] CLAUDE.md event registry is updated
- [ ] Tests added: state transitions, cascade, workload reaction
- [ ] `pytest tests/ -v` passes with no regressions
