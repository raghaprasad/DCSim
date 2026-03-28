"""Chaos injection: schedule failure and auto-repair events."""

from __future__ import annotations

from dataclasses import dataclass, field

from dcsim.engine.clock import SimTime
from dcsim.engine.event import Event, EventPayload, EventQueue

REPAIR_EVENT_TYPES: dict[str, str] = {
    "hardware.gpu.fail": "hardware.gpu.repair",
    "hardware.gpu.throttle": "hardware.gpu.unthrottle",
    "hardware.link.fail": "hardware.link.repair",
    "hardware.switch.fail": "hardware.switch.repair",
}


@dataclass
class ChaosEvent:
    target_id: str
    event_type: str
    time: SimTime
    duration: SimTime | None
    properties: dict = field(default_factory=dict)


class ChaosInjector:
    def inject(self, events: list[ChaosEvent], queue: EventQueue) -> list[Event]:
        scheduled: list[Event] = []
        for ce in events:
            data = self._build_data(ce.target_id, ce.event_type)
            data.update(ce.properties)

            fail_event = queue.schedule(
                ce.time,
                EventPayload(event_type=ce.event_type, data=data),
                priority=0,
            )
            scheduled.append(fail_event)

            if ce.duration is not None:
                repair_type = REPAIR_EVENT_TYPES.get(ce.event_type)
                if repair_type is None:
                    raise ValueError(
                        f"No repair mapping for event type {ce.event_type!r}. "
                        f"Supported: {list(REPAIR_EVENT_TYPES)}"
                    )
                repair_data = self._build_data(ce.target_id, ce.event_type)
                repair_event = queue.schedule(
                    ce.time + ce.duration,
                    EventPayload(event_type=repair_type, data=repair_data),
                    priority=0,
                )
                scheduled.append(repair_event)

        return scheduled

    @staticmethod
    def _build_data(target_id: str, event_type: str) -> dict:
        data: dict = {"component_id": target_id}
        if "gpu" in event_type:
            data["gpu_id"] = target_id
        elif "link" in event_type:
            data["link_id"] = target_id
        elif "switch" in event_type:
            data["switch_id"] = target_id
        return data
