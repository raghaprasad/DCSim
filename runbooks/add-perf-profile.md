# Adding a Hardware Performance Profile

## Overview
How to model different hardware SKUs (e.g., H100 vs A100 vs B200 GPUs) with distinct performance characteristics that affect workload timing.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/hardware/components.py` | Add/modify properties on component dataclass | Define SKU-specific specs (TFLOPS, bandwidth, memory) |
| `src/dcsim/hardware/topology.py` | Accept GPU model as parameter | So builders create the right GPU type |
| `src/dcsim/workloads/base.py` | Expand `_gpu_throttle_factors` to richer profile dict | So WorkloadManager can pass GPU specs to workloads |
| `src/dcsim/workloads/allreduce.py` | Use GPU specs in `get_next_phase()` | So compute duration reflects actual GPU capability |

## Step-by-Step

### 1. Define performance constants (`hardware/components.py`)

Add a profile dict or enum for known SKUs:

```python
GPU_PROFILES = {
    "H100": {"compute_tflops": 312.0, "memory_gb": 80.0, "memory_bw_gbps": 3350},
    "A100": {"compute_tflops": 156.0, "memory_gb": 80.0, "memory_bw_gbps": 2039},
    "B200": {"compute_tflops": 500.0, "memory_gb": 192.0, "memory_bw_gbps": 8000},
}

@dataclass
class GPU(HardwareComponent):
    model: str = "H100"
    compute_tflops: float = 312.0  # Populated from GPU_PROFILES
    memory_gb: float = 80.0
    memory_bw_gbps: float = 3350.0
    throttle_factor: float = 1.0
```

### 2. Parameterize topology builders (`hardware/topology.py`)

```python
def build_standard_cluster(gpu_model: str = "H100") -> HardwareGraph:
    profile = GPU_PROFILES[gpu_model]
    for n in range(num_nodes):
        for g in range(gpus_per_node):
            gpu = GPU(
                id=f"node-{n}/gpu-{g}",
                node_id=f"node-{n}",
                gpu_index=g,
                model=gpu_model,
                compute_tflops=profile["compute_tflops"],
                memory_gb=profile["memory_gb"],
            )
```

### 3. Pass GPU specs through WorkloadManager (`workloads/base.py`)

Currently `_gpu_throttle_factors` is `dict[str, float]`. Expand to include specs:

```python
# In _get_gpu_states():
states[gpu_id] = {
    "throttle_factor": self._gpu_throttle_factors.get(gpu_id, 1.0),
    "failed": gpu_id in self._gpu_failed,
    "compute_tflops": self._gpu_profiles[gpu_id].get("compute_tflops", 312.0),
}
```

### 4. Use specs in workload timing (`workloads/allreduce.py`)

Currently `base_compute_us` is a flat user-configured number. To make it GPU-aware:

```python
def get_next_phase(self, gpu_states, now):
    if not self._in_communicate:
        # Option A: Scale base_compute_us by relative GPU capability
        min_tflops = min(s.get("compute_tflops", 312.0) for s in gpu_states.values())
        reference_tflops = 312.0  # H100 baseline
        scale_factor = reference_tflops / min_tflops  # A100 = 2x slower
        min_throttle = min(s.get("throttle_factor", 1.0) for s in gpu_states.values())
        duration = int(self.base_compute_us * scale_factor / min_throttle)
        return (WorkloadPhase.COMPUTE, duration)
```

## Key Principle

`base_compute_us` is always the **reference duration on the baseline GPU** (H100). Other GPUs scale relative to that. This keeps the workload config simple — you don't need to recalculate base times per GPU model.

## Checklist
- [ ] GPU_PROFILES dict has all target SKUs
- [ ] GPU dataclass has the new properties
- [ ] Topology builder accepts gpu_model parameter
- [ ] WorkloadManager passes GPU specs to workloads
- [ ] Workload timing formulas use the specs
- [ ] Tests: same workload on H100 vs A100 produces different completion times
- [ ] `pytest tests/ -v` passes
