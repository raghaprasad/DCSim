# Adding a New Network Topology

## Overview
How to add a new datacenter network topology (e.g., torus, dragonfly, custom import) to the simulator.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/hardware/topology.py` | Add new builder function | Creates the topology's nodes, switches, and links |
| `src/dcsim/hardware/components.py` | Possibly new `LinkType` or switch tier values | If the topology uses connection types not yet modeled |

## Step-by-Step

### 1. Add a builder function (`hardware/topology.py`)

Follow the existing pattern. Every builder returns a `HardwareGraph`:

```python
def build_torus_cluster(
    num_nodes: int = 16,
    gpus_per_node: int = 8,
    dimensions: int = 2,
) -> HardwareGraph:
    """Build a 2D/3D torus topology.

    Each node connects to its neighbors in each dimension.
    Wraps around at edges (torus vs mesh).
    """
    graph = HardwareGraph()

    # 1. Create GPUs for each node
    for n in range(num_nodes):
        for g in range(gpus_per_node):
            gpu = GPU(
                id=f"node-{n}/gpu-{g}",
                node_id=f"node-{n}",
                gpu_index=g,
            )
            graph.add_component(gpu)

        # 2. Create intra-node NVLink connections (all-to-all within node)
        for g1 in range(gpus_per_node):
            for g2 in range(g1 + 1, gpus_per_node):
                link = Link(
                    id=f"link-node-{n}/gpu-{g1}-node-{n}/gpu-{g2}",
                    source_id=f"node-{n}/gpu-{g1}",
                    target_id=f"node-{n}/gpu-{g2}",
                    link_type=LinkType.NVLINK,
                    bandwidth_gbps=7200,
                    latency_us=0.1,
                )
                graph.add_link(link)

    # 3. Create inter-node torus connections
    side = int(num_nodes ** (1.0 / dimensions))
    for n in range(num_nodes):
        coords = _index_to_coords(n, side, dimensions)
        for dim in range(dimensions):
            neighbor_coords = list(coords)
            neighbor_coords[dim] = (coords[dim] + 1) % side  # wrap
            neighbor = _coords_to_index(neighbor_coords, side)
            if neighbor > n:  # avoid duplicate links
                link = Link(
                    id=f"link-node-{n}-node-{neighbor}",
                    source_id=f"node-{n}/gpu-0",  # representative GPU
                    target_id=f"node-{neighbor}/gpu-0",
                    link_type=LinkType.INFINIBAND,
                    bandwidth_gbps=400,
                    latency_us=1.0,
                )
                graph.add_link(link)

    return graph
```

### 2. Add new link types if needed (`hardware/components.py`)

If the topology uses connections not covered by NVLINK/INFINIBAND:

```python
class LinkType(Enum):
    NVLINK = "nvlink"
    INFINIBAND = "infiniband"
    ETHERNET = "ethernet"          # ← new
    OPTICAL_CIRCUIT = "optical"    # ← new
```

### 3. Add switch tiers if needed

Some topologies have different switching layers:

```python
# Fat-tree: tier 0=ToR, 1=spine
# Dragonfly: tier 0=local, 1=global
# Torus: no switches (direct connections)
```

## Topology Design Patterns

| Topology | Switches? | Link pattern | Failure characteristics |
|----------|-----------|-------------|----------------------|
| Fat-tree | Yes (ToR + spine) | Hierarchical, redundant paths | Switch failure partitions a rack |
| Torus | No | Direct neighbor connections, wrap-around | Link failure has limited blast radius |
| Dragonfly | Yes (local + global) | Two-level, all-to-all within groups | Group switch failure isolates a group |
| Custom | Varies | Imported from JSON/YAML | Depends on design |

## Testing Approach

For any new topology, test:
1. **Component counts**: correct number of GPUs, switches, links
2. **Full connectivity**: any GPU can reach any other GPU
3. **Failure isolation**: a single failure doesn't partition the entire cluster
4. **Bandwidth**: bottleneck bandwidth between distant GPUs is correct

## Checklist
- [ ] Builder function returns a valid `HardwareGraph`
- [ ] All GPUs have unique IDs following `node-{n}/gpu-{g}` convention
- [ ] Intra-node links use NVLink, inter-node links use InfiniBand (or new type)
- [ ] New LinkType values added if needed
- [ ] Tests: component counts, full connectivity, failure isolation
- [ ] `pytest tests/ -v` passes
