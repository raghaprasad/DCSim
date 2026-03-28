"""Topology builder: fixed 32-GPU cluster."""

from __future__ import annotations

from dcsim.hardware.components import GPU, Link, LinkType, Switch
from dcsim.hardware.graph import HardwareGraph


def build_standard_cluster() -> HardwareGraph:
    """Build a 4-node x 8-GPU cluster with 2 ToR and 2 spine switches.

    Node 0,1 -> ToR tor-0 (rack 0)
    Node 2,3 -> ToR tor-1 (rack 1)
    tor-0, tor-1 -> spine-0, spine-1 (full mesh)

    Intra-node: NVLink all-to-all (7200 Gbps, 0.1us)
    Node-to-ToR: InfiniBand (400 Gbps, 1.0us)
    ToR-to-Spine: InfiniBand (400 Gbps, 1.0us)
    """
    graph = HardwareGraph()

    # GPUs: 4 nodes x 8 GPUs
    for node_idx in range(4):
        for gpu_idx in range(8):
            graph.add_component(
                GPU(
                    id=f"node-{node_idx}/gpu-{gpu_idx}",
                    node_id=f"node-{node_idx}",
                    gpu_index=gpu_idx,
                )
            )

    # Switches: 2 ToR (tier 0), 2 spine (tier 1)
    for tor_idx in range(2):
        graph.add_component(Switch(id=f"tor-{tor_idx}", tier=0))
    for spine_idx in range(2):
        graph.add_component(Switch(id=f"spine-{spine_idx}", tier=1))

    # Intra-node NVLink: all-to-all within each node
    for node_idx in range(4):
        gpu_ids = [f"node-{node_idx}/gpu-{g}" for g in range(8)]
        for i in range(len(gpu_ids)):
            for j in range(i + 1, len(gpu_ids)):
                graph.add_link(
                    Link(
                        id=f"link-{gpu_ids[i]}-{gpu_ids[j]}",
                        source_id=gpu_ids[i],
                        target_id=gpu_ids[j],
                        link_type=LinkType.NVLINK,
                        bandwidth_gbps=7200.0,
                        latency_us=0.1,
                    )
                )

    # GPU-to-ToR InfiniBand
    node_to_tor = {0: "tor-0", 1: "tor-0", 2: "tor-1", 3: "tor-1"}
    for node_idx in range(4):
        tor_id = node_to_tor[node_idx]
        for gpu_idx in range(8):
            gpu_id = f"node-{node_idx}/gpu-{gpu_idx}"
            graph.add_link(
                Link(
                    id=f"link-{gpu_id}-{tor_id}",
                    source_id=gpu_id,
                    target_id=tor_id,
                    link_type=LinkType.INFINIBAND,
                    bandwidth_gbps=400.0,
                    latency_us=1.0,
                )
            )

    # ToR-to-Spine InfiniBand (full mesh)
    for tor_idx in range(2):
        for spine_idx in range(2):
            tor_id = f"tor-{tor_idx}"
            spine_id = f"spine-{spine_idx}"
            graph.add_link(
                Link(
                    id=f"link-{tor_id}-{spine_id}",
                    source_id=tor_id,
                    target_id=spine_id,
                    link_type=LinkType.INFINIBAND,
                    bandwidth_gbps=400.0,
                    latency_us=1.0,
                )
            )

    return graph
