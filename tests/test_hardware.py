"""Phase 2 gating tests for the hardware graph."""

import pytest

from dcsim.engine.event import Event, EventPayload, EventQueue
from dcsim.engine.loop import SimulationLoop
from dcsim.hardware.components import GPUState, LinkState, SwitchState
from dcsim.hardware.graph import HardwareGraph
from dcsim.hardware.topology import build_standard_cluster


class TestClusterComponentCounts:
    """Test 1: build_standard_cluster produces correct counts and naming."""

    def test_gpu_count(self):
        graph = build_standard_cluster()
        assert len(graph.get_gpus()) == 32

    def test_switch_count(self):
        graph = build_standard_cluster()
        switches = [c for c in graph._components.values() if c.component_type == "switch"]
        assert len(switches) == 4

    def test_link_count(self):
        graph = build_standard_cluster()
        links = [c for c in graph._components.values() if c.component_type == "link"]
        # 112 NVLink + 32 GPU-to-ToR + 4 ToR-to-Spine = 148
        assert len(links) == 148

    def test_naming_conventions(self):
        graph = build_standard_cluster()
        assert graph.get_component("node-0/gpu-0").component_type == "gpu"
        assert graph.get_component("node-3/gpu-7").component_type == "gpu"
        assert graph.get_component("tor-0").component_type == "switch"
        assert graph.get_component("tor-1").component_type == "switch"
        assert graph.get_component("spine-0").component_type == "switch"
        assert graph.get_component("spine-1").component_type == "switch"


class TestFullConnectivity:
    """Test 2: Any GPU can reach any other GPU."""

    def test_all_gpu_pairs_reachable(self):
        graph = build_standard_cluster()
        gpu_ids = [g.id for g in graph.get_gpus()]
        for i in range(len(gpu_ids)):
            for j in range(i + 1, len(gpu_ids)):
                bw = graph.get_bandwidth_between(gpu_ids[i], gpu_ids[j])
                assert bw > 0, f"No path between {gpu_ids[i]} and {gpu_ids[j]}"


class TestGPUStateTransitions:
    """Test 3: Valid transitions work; invalid transitions raise."""

    def test_valid_transition_chain(self):
        graph = build_standard_cluster()
        gpu_id = "node-0/gpu-0"
        queue = EventQueue()
        dummy = Event(time=0, priority=0, sequence=0, event_id="test", payload=EventPayload("test"))

        # IDLE -> IN_USE
        graph.apply_state_change(gpu_id, GPUState.IN_USE, dummy, queue, 0)
        assert graph.get_component(gpu_id).state == GPUState.IN_USE

        # IN_USE -> THROTTLED
        dummy.payload = EventPayload("test", data={"throttle_factor": 0.33})
        graph.apply_state_change(gpu_id, GPUState.THROTTLED, dummy, queue, 0)
        assert graph.get_component(gpu_id).state == GPUState.THROTTLED

        # THROTTLED -> FAILED
        graph.apply_state_change(gpu_id, GPUState.FAILED, dummy, queue, 0)
        assert graph.get_component(gpu_id).state == GPUState.FAILED

        # FAILED -> IDLE
        graph.apply_state_change(gpu_id, GPUState.IDLE, dummy, queue, 0)
        assert graph.get_component(gpu_id).state == GPUState.IDLE

    def test_invalid_transition_raises(self):
        graph = build_standard_cluster()
        gpu_id = "node-0/gpu-0"
        queue = EventQueue()
        dummy = Event(time=0, priority=0, sequence=0, event_id="test", payload=EventPayload("test"))

        # IDLE -> THROTTLED is invalid
        with pytest.raises(ValueError):
            graph.apply_state_change(gpu_id, GPUState.THROTTLED, dummy, queue, 0)


class TestToRSwitchFailure:
    """Test 4: ToR switch failure cascades and isolates rack."""

    def test_tor_failure_cascade(self):
        graph = build_standard_cluster()
        queue = EventQueue()
        dummy = Event(time=0, priority=0, sequence=0, event_id="test", payload=EventPayload("test"))

        # Before: cross-rack connectivity exists
        assert graph.get_bandwidth_between("node-0/gpu-0", "node-2/gpu-0") > 0

        # Fail tor-0
        cascade = graph.apply_state_change("tor-0", SwitchState.FAILED, dummy, queue, 0)

        # 16 GPU-to-ToR links (nodes 0,1 x 8 GPUs) + 2 ToR-to-Spine links = 18
        cascade_types = [e.payload.event_type for e in cascade]
        assert cascade_types.count("cascade.link.down") == 18

        # Rack 0 GPUs lose cross-rack connectivity
        assert graph.get_bandwidth_between("node-0/gpu-0", "node-2/gpu-0") == 0.0
        assert graph.get_bandwidth_between("node-1/gpu-0", "node-3/gpu-0") == 0.0

        # Intra-node NVLink still works
        assert graph.get_bandwidth_between("node-0/gpu-0", "node-0/gpu-1") > 0

        # Rack 1 cross-node connectivity still works
        assert graph.get_bandwidth_between("node-2/gpu-0", "node-3/gpu-0") > 0


class TestSpineLinkFailure:
    """Test 5: Spine link failure reduces redundancy but maintains connectivity."""

    def test_spine_link_down_keeps_connectivity(self):
        graph = build_standard_cluster()
        queue = EventQueue()
        dummy = Event(time=0, priority=0, sequence=0, event_id="test", payload=EventPayload("test"))

        # Fail one spine link
        graph.apply_state_change("link-tor-0-spine-0", LinkState.DOWN, dummy, queue, 0)

        # Connectivity maintained via spine-1
        bw = graph.get_bandwidth_between("node-0/gpu-0", "node-2/gpu-0")
        assert bw > 0
        assert bw == 400.0


class TestEngineIntegration:
    """Test 6: Hardware events + cascade fire in correct priority order."""

    def test_gpu_fail_cascade_priority(self):
        sim = SimulationLoop()
        graph = build_standard_cluster()
        graph.setup(sim)

        # Set GPU to IN_USE so cascade fires
        gpu_id = "node-0/gpu-0"
        graph.get_component(gpu_id).state = GPUState.IN_USE

        event_log: list[tuple[str, int]] = []

        original_fail_handler = graph._handle_gpu_fail

        def logging_fail_handler(event, ctx):
            event_log.append((event.payload.event_type, event.priority))
            return original_fail_handler(event, ctx)

        def cascade_handler(event, ctx):
            event_log.append((event.payload.event_type, event.priority))

        # Replace the fail handler with a logging wrapper
        sim._handlers["hardware.gpu.fail"] = [logging_fail_handler]
        sim.register_handler("cascade.gpu.job_interrupted", cascade_handler)

        sim.schedule(
            1000,
            EventPayload(event_type="hardware.gpu.fail", data={"target_id": gpu_id}),
            priority=0,
        )

        sim.run()

        assert len(event_log) == 2
        assert event_log[0] == ("hardware.gpu.fail", 0)
        assert event_log[1] == ("cascade.gpu.job_interrupted", 10)
