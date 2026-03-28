"""Gating tests for the event logger."""

from dcsim.engine.event import EventPayload
from dcsim.engine.loop import SimulationLoop
from dcsim.observer.logger import EventLogger, LogEntry


class TestEventLoggerTimeline:
    """Log 5 entries -> get_timeline returns them in order."""

    def test_get_timeline_returns_entries_in_order(self):
        sim = SimulationLoop()
        logger = EventLogger()
        handler = logger.make_handler()

        sim.register_handler("test.event", handler)

        for i, t in enumerate([500, 100, 300, 200, 400]):
            sim.schedule(
                t,
                EventPayload(
                    event_type="test.event",
                    data={"seq": i, "component_id": f"comp-{i}"},
                ),
            )

        sim.run()

        timeline = logger.get_timeline()

        assert len(timeline) == 5
        assert [e.timestamp for e in timeline] == [100, 200, 300, 400, 500]
        assert all(isinstance(e, LogEntry) for e in timeline)
        assert all(e.component_id is not None for e in timeline)


class TestEventLoggerExport:
    """export_dicts returns list of dicts suitable for DataFrame construction."""

    def test_export_dicts_returns_list_of_dicts(self):
        sim = SimulationLoop()
        logger = EventLogger()
        handler = logger.make_handler()

        sim.register_handler("hardware.gpu.fail", handler)
        sim.register_handler("workload.step.complete", handler)

        sim.schedule(
            1000,
            EventPayload(
                event_type="hardware.gpu.fail",
                data={"component_id": "node-0/gpu-0", "gpu_id": "node-0/gpu-0"},
            ),
        )
        sim.schedule(
            2000,
            EventPayload(
                event_type="workload.step.complete",
                data={"job_id": "train-1", "step": 1},
            ),
        )

        sim.run()

        dicts = logger.export_dicts()

        assert isinstance(dicts, list)
        assert len(dicts) == 2
        assert all(isinstance(d, dict) for d in dicts)

        required_keys = {
            "timestamp", "event_id", "parent_event_id",
            "event_type", "component_id", "job_id",
            "description", "data",
        }
        for d in dicts:
            assert required_keys.issubset(d.keys())

        assert dicts[0]["timestamp"] == 1000
        assert isinstance(dicts[0]["timestamp"], int)
        assert dicts[0]["event_type"] == "hardware.gpu.fail"
        assert dicts[1]["event_type"] == "workload.step.complete"
        assert dicts[0]["component_id"] == "node-0/gpu-0"
        assert dicts[1]["job_id"] == "train-1"
