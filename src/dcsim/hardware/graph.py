"""HardwareGraph: NetworkX-backed hardware topology with cascade logic."""

from __future__ import annotations

import networkx as nx

from dcsim.engine.clock import SimTime
from dcsim.engine.event import Event, EventPayload, EventQueue
from dcsim.engine.loop import EventHandler, SimulationContext, SimulationLoop
from dcsim.hardware.components import (
    GPU,
    GPUState,
    HardwareComponent,
    Link,
    LinkState,
    Switch,
    SwitchState,
    validate_transition,
)


class HardwareGraph:
    """Wraps a NetworkX graph representing the datacenter hardware topology."""

    @staticmethod
    def _extract_target_id(data: dict, *fallback_keys: str) -> str:
        """Extract a target component ID from event data with fallback chain.

        Tries ``target_id`` first, then each key in *fallback_keys*, then
        ``component_id`` as the last resort.
        """
        if "target_id" in data:
            return data["target_id"]
        for key in fallback_keys:
            if key in data:
                return data[key]
        if "component_id" in data:
            return data["component_id"]
        raise KeyError(
            f"Cannot find target ID in event data. Tried: target_id, {', '.join(fallback_keys)}, component_id. "
            f"Available keys: {list(data.keys())}"
        )

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._components: dict[str, HardwareComponent] = {}
        self._links: dict[str, Link] = {}

    def add_component(self, c: HardwareComponent) -> None:
        self._components[c.id] = c
        if isinstance(c, (GPU, Switch)):
            self._graph.add_node(c.id, component=c, active=True)

    def add_link(self, link: Link) -> None:
        self._components[link.id] = link
        self._links[link.id] = link
        self._graph.add_edge(
            link.source_id,
            link.target_id,
            link_id=link.id,
            bandwidth_gbps=link.bandwidth_gbps,
            active=True,
        )

    def get_component(self, component_id: str) -> HardwareComponent:
        return self._components[component_id]

    def get_gpus(self, state: GPUState | None = None) -> list[GPU]:
        gpus = [c for c in self._components.values() if isinstance(c, GPU)]
        if state is not None:
            gpus = [g for g in gpus if g.state == state]
        return sorted(gpus, key=lambda g: g.id)

    def get_links_for_component(self, component_id: str) -> list[Link]:
        result: list[Link] = []
        if component_id not in self._graph:
            return result
        for _, neighbor, data in self._graph.edges(component_id, data=True):
            link = self._links.get(data["link_id"])
            if link is not None:
                result.append(link)
        return result

    def _active_view(self) -> nx.classes.graphviews.SubGraph:
        return nx.subgraph_view(
            self._graph,
            filter_node=lambda n: self._graph.nodes[n].get("active", True),
            filter_edge=lambda u, v: self._graph.edges[u, v].get("active", True),
        )

    def get_bandwidth_between(self, src: str, dst: str) -> float:
        view = self._active_view()
        if src not in view or dst not in view:
            return 0.0
        if not nx.has_path(view, src, dst):
            return 0.0
        path = nx.shortest_path(view, src, dst)
        bandwidths = []
        for i in range(len(path) - 1):
            edge_data = self._graph.edges[path[i], path[i + 1]]
            bandwidths.append(edge_data["bandwidth_gbps"])
        return min(bandwidths) if bandwidths else 0.0

    def apply_state_change(
        self,
        component_id: str,
        new_state: GPUState | LinkState | SwitchState,
        event: Event,
        queue: EventQueue,
        now: SimTime,
    ) -> list[Event]:
        component = self._components[component_id]
        validate_transition(component, new_state)
        old_state = component.state
        component.state = new_state

        cascade_events: list[Event] = []

        if isinstance(component, Switch) and new_state == SwitchState.FAILED:
            self._graph.nodes[component_id]["active"] = False
            for link in self.get_links_for_component(component_id):
                link.state = LinkState.DOWN
                self._graph.edges[link.source_id, link.target_id]["active"] = False
                evt = queue.schedule(
                    now,
                    EventPayload(
                        event_type="cascade.link.down",
                        parent_event_id=event.event_id,
                        data={"link_id": link.id, "cause": "switch_failure", "switch_id": component_id},
                    ),
                    priority=10,
                )
                cascade_events.append(evt)

        elif isinstance(component, Switch) and new_state == SwitchState.ACTIVE:
            self._graph.nodes[component_id]["active"] = True
            for link in self.get_links_for_component(component_id):
                link.state = LinkState.UP
                self._graph.edges[link.source_id, link.target_id]["active"] = True

        elif isinstance(component, GPU):
            if new_state == GPUState.FAILED and old_state == GPUState.IN_USE:
                evt = queue.schedule(
                    now,
                    EventPayload(
                        event_type="cascade.gpu.job_interrupted",
                        parent_event_id=event.event_id,
                        data={"gpu_id": component_id},
                    ),
                    priority=10,
                )
                cascade_events.append(evt)

            elif new_state == GPUState.THROTTLED and old_state == GPUState.IN_USE:
                component.throttle_factor = event.payload.data.get("throttle_factor", 0.33)
                evt = queue.schedule(
                    now,
                    EventPayload(
                        event_type="cascade.gpu.throttled",
                        parent_event_id=event.event_id,
                        data={"gpu_id": component_id, "throttle_factor": component.throttle_factor},
                    ),
                    priority=10,
                )
                cascade_events.append(evt)

            elif new_state == GPUState.IN_USE and old_state == GPUState.THROTTLED:
                component.throttle_factor = 1.0

        elif isinstance(component, Link):
            if new_state == LinkState.DOWN:
                self._graph.edges[component.source_id, component.target_id]["active"] = False
                evt = queue.schedule(
                    now,
                    EventPayload(
                        event_type="cascade.link.down",
                        parent_event_id=event.event_id,
                        data={"link_id": component_id},
                    ),
                    priority=10,
                )
                cascade_events.append(evt)

            elif new_state == LinkState.UP:
                self._graph.edges[component.source_id, component.target_id]["active"] = True

        return cascade_events

    # -- Event handlers --

    def _handle_gpu_fail(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        gpu_id = self._extract_target_id(event.payload.data, "gpu_id")
        self.apply_state_change(gpu_id, GPUState.FAILED, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_gpu_repair(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        gpu_id = self._extract_target_id(event.payload.data, "gpu_id")
        self.apply_state_change(gpu_id, GPUState.IDLE, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_gpu_throttle(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        gpu_id = self._extract_target_id(event.payload.data, "gpu_id")
        self.apply_state_change(gpu_id, GPUState.THROTTLED, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_gpu_unthrottle(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        gpu_id = self._extract_target_id(event.payload.data, "gpu_id")
        self.apply_state_change(gpu_id, GPUState.IN_USE, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_link_fail(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        link_id = self._extract_target_id(event.payload.data, "link_id")
        self.apply_state_change(link_id, LinkState.DOWN, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_link_repair(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        link_id = self._extract_target_id(event.payload.data, "link_id")
        self.apply_state_change(link_id, LinkState.UP, event, ctx.queue, ctx.clock.now())
        return None

    def _handle_switch_fail(self, event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
        switch_id = self._extract_target_id(event.payload.data, "switch_id")
        self.apply_state_change(switch_id, SwitchState.FAILED, event, ctx.queue, ctx.clock.now())
        return None

    def setup(self, sim: SimulationLoop) -> None:
        """Register all hardware event handlers with the simulation loop."""
        sim.context.hardware = self  # type: ignore[attr-defined]
        sim.register_handler("hardware.gpu.fail", self._handle_gpu_fail)
        sim.register_handler("hardware.gpu.repair", self._handle_gpu_repair)
        sim.register_handler("hardware.gpu.throttle", self._handle_gpu_throttle)
        sim.register_handler("hardware.gpu.unthrottle", self._handle_gpu_unthrottle)
        sim.register_handler("hardware.link.fail", self._handle_link_fail)
        sim.register_handler("hardware.link.repair", self._handle_link_repair)
        sim.register_handler("hardware.switch.fail", self._handle_switch_fail)
