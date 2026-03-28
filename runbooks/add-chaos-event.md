# Adding a New Chaos/Failure Event Type

## Overview
How to add a new kind of hardware failure or degradation (e.g., ECC memory error, NVLink CRC error, power supply failure) that the chaos monkey can inject.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/chaos/injector.py` | Add to `REPAIR_EVENT_MAP` | So auto-repair scheduling works |
| `src/dcsim/hardware/graph.py` | Register handler for the new event | So hardware graph updates state on this event |
| `src/dcsim/workloads/base.py` | Add handler in WorkloadManager (if it affects running jobs) | So workloads react to this failure |
| `CLAUDE.md` | Add to event type registry | Documentation |

## Step-by-Step

### 1. Choose event type names

Follow the naming convention: `hardware.<component>.<action>`

```
hardware.gpu.memory_error     → hardware.gpu.memory_recover
hardware.link.crc_error       → hardware.link.crc_recover
hardware.switch.power_fail    → hardware.switch.power_restore
```

### 2. Add repair mapping (`chaos/injector.py`)

```python
REPAIR_EVENT_MAP = {
    # ... existing mappings ...
    "hardware.gpu.memory_error": "hardware.gpu.memory_recover",
}
```

This allows `ChaosInjector.inject()` to auto-schedule the repair event when `duration` is set on a `ChaosEvent`.

### 3. Define what happens to hardware state (`hardware/graph.py`)

Register a handler that updates the component's state:

```python
sim.register_handler("hardware.gpu.memory_error", self._handle_gpu_memory_error)

def _handle_gpu_memory_error(self, event, ctx):
    gpu = self.get_component(event.payload.data["component_id"])
    # Memory errors might degrade performance rather than full failure
    gpu.state = GPUState.THROTTLED
    gpu.throttle_factor = 0.5  # 50% throughput with ECC corrections
    if old_state == GPUState.IN_USE:
        return [EventPayload(
            event_type="cascade.gpu.throttled",
            data={"gpu_id": gpu.id, "throttle_factor": 0.5},
        )]
```

### 4. Add workload reaction if needed (`workloads/base.py`)

If the new event affects running workloads differently from existing events:

```python
sim.register_handler("hardware.gpu.memory_error", self._handle_gpu_memory_error)

def _handle_gpu_memory_error(self, event, ctx):
    # For memory errors, we might want to checkpoint immediately
    # rather than just recalculate timing
    ...
```

If the new event produces the same cascade events as existing ones (e.g., `cascade.gpu.throttled`), the WorkloadManager already handles it — no changes needed.

### 5. Create chaos events using the new type

```python
ChaosEvent(
    target_id="node-2/gpu-3",
    event_type="hardware.gpu.memory_error",
    time=500 * MILLISECOND,
    duration=2 * SECOND,  # Auto-recovers after 2s
    properties={"throttle_factor": 0.5},
)
```

## Event Type Design Principles

- **A failure event should produce exactly one cascade event** that downstream consumers (WorkloadManager) listen for. Don't make the WorkloadManager listen for every hardware event variant.
- **Cascade events are the abstraction boundary**: hardware details stay in `hardware/`, workloads only see `cascade.gpu.job_interrupted`, `cascade.gpu.throttled`, `cascade.gpu.isolated`.
- **If your new event's effect maps to an existing cascade event**, you don't need to change WorkloadManager at all.

```
hardware.gpu.memory_error  ─┐
hardware.gpu.throttle      ─┼─→  cascade.gpu.throttled  →  WorkloadManager
hardware.gpu.overheat      ─┘
```

## Checklist
- [ ] Event type names follow `hardware.<component>.<action>` convention
- [ ] Repair mapping added in `chaos/injector.py`
- [ ] Hardware handler registered and updates component state
- [ ] Appropriate cascade event emitted (reuse existing if possible)
- [ ] WorkloadManager handles the cascade (or already does via existing handler)
- [ ] CLAUDE.md event registry updated
- [ ] Tests: inject event → verify state change → verify cascade → verify workload reaction
- [ ] `pytest tests/ -v` passes
