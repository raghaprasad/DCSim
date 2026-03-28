"""Event logger: captures simulation events for timeline display and export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from dcsim.engine.clock import SimTime
from dcsim.engine.event import Event, EventPayload
from dcsim.engine.loop import EventHandler, SimulationContext


@dataclass
class LogEntry:
    timestamp: SimTime
    event_id: str
    parent_event_id: str | None
    event_type: str
    component_id: str | None
    job_id: str | None
    description: str
    data: dict


class EventLogger:
    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

    def make_handler(self) -> EventHandler:
        def _log_handler(event: Event, ctx: SimulationContext) -> list[EventPayload] | None:
            data = event.payload.data
            entry = LogEntry(
                timestamp=event.time,
                event_id=event.event_id,
                parent_event_id=event.payload.parent_event_id,
                event_type=event.payload.event_type,
                component_id=(
                    data.get("component_id")
                    or data.get("gpu_id")
                    or data.get("link_id")
                    or data.get("switch_id")
                ),
                job_id=data.get("job_id"),
                description=event.payload.describe(),
                data=dict(data),
            )
            self.entries.append(entry)
            return None

        return _log_handler

    def get_timeline(self) -> list[LogEntry]:
        return sorted(self.entries, key=lambda e: e.timestamp)

    def export_json(self) -> str:
        return json.dumps([asdict(entry) for entry in self.entries], indent=2)

    def export_dicts(self) -> list[dict]:
        return [asdict(entry) for entry in self.entries]
