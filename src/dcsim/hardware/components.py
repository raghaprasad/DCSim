"""Hardware components: state enums, transition validation, and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GPUState(Enum):
    IDLE = "idle"
    IN_USE = "in_use"
    THROTTLED = "throttled"
    FAILED = "failed"


class LinkState(Enum):
    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"


class SwitchState(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"


class LinkType(Enum):
    NVLINK = "nvlink"
    INFINIBAND = "infiniband"


VALID_GPU_TRANSITIONS: set[tuple[GPUState, GPUState]] = {
    (GPUState.IDLE, GPUState.IN_USE),
    (GPUState.IDLE, GPUState.FAILED),
    (GPUState.IN_USE, GPUState.THROTTLED),
    (GPUState.IN_USE, GPUState.FAILED),
    (GPUState.THROTTLED, GPUState.IN_USE),
    (GPUState.THROTTLED, GPUState.FAILED),
    (GPUState.FAILED, GPUState.IDLE),
}

VALID_LINK_TRANSITIONS: set[tuple[LinkState, LinkState]] = {
    (LinkState.UP, LinkState.DEGRADED),
    (LinkState.UP, LinkState.DOWN),
    (LinkState.DEGRADED, LinkState.DOWN),
    (LinkState.DEGRADED, LinkState.UP),
    (LinkState.DOWN, LinkState.UP),
}

VALID_SWITCH_TRANSITIONS: set[tuple[SwitchState, SwitchState]] = {
    (SwitchState.ACTIVE, SwitchState.DEGRADED),
    (SwitchState.ACTIVE, SwitchState.FAILED),
    (SwitchState.DEGRADED, SwitchState.FAILED),
    (SwitchState.FAILED, SwitchState.ACTIVE),
}


def validate_transition(
    component: HardwareComponent,
    new_state: GPUState | LinkState | SwitchState,
) -> None:
    """Raise ValueError if the state transition is not valid."""
    old = component.state
    pair = (old, new_state)

    if isinstance(old, GPUState):
        if pair not in VALID_GPU_TRANSITIONS:
            raise ValueError(f"Invalid GPU transition: {old.value} -> {new_state.value}")
    elif isinstance(old, LinkState):
        if pair not in VALID_LINK_TRANSITIONS:
            raise ValueError(f"Invalid Link transition: {old.value} -> {new_state.value}")
    elif isinstance(old, SwitchState):
        if pair not in VALID_SWITCH_TRANSITIONS:
            raise ValueError(f"Invalid Switch transition: {old.value} -> {new_state.value}")
    else:
        raise TypeError(f"Unknown state type: {type(old)}")


@dataclass
class HardwareComponent:
    id: str = ""
    component_type: str = ""
    state: GPUState | LinkState | SwitchState = field(default=GPUState.IDLE)


@dataclass
class GPU(HardwareComponent):
    node_id: str = ""
    gpu_index: int = 0
    state: GPUState = GPUState.IDLE
    throttle_factor: float = 1.0

    def __post_init__(self) -> None:
        self.component_type = "gpu"


@dataclass
class Switch(HardwareComponent):
    tier: int = 0
    state: SwitchState = SwitchState.ACTIVE

    def __post_init__(self) -> None:
        self.component_type = "switch"


@dataclass
class Link(HardwareComponent):
    source_id: str = ""
    target_id: str = ""
    link_type: LinkType = LinkType.INFINIBAND
    bandwidth_gbps: float = 400.0
    latency_us: float = 1.0
    state: LinkState = LinkState.UP

    def __post_init__(self) -> None:
        self.component_type = "link"
