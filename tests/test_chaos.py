"""Gating tests for chaos injection."""

from dcsim.engine.event import Event, EventPayload
from dcsim.engine.loop import SimulationLoop
from dcsim.chaos.injector import ChaosEvent, ChaosInjector


class TestChaosInjectorScheduling:
    """Schedule GPU fail at t=1000 -> event fires at t=1000."""

    def test_gpu_fail_fires_at_correct_time(self):
        sim = SimulationLoop()
        injector = ChaosInjector()

        scheduled = injector.inject(
            [ChaosEvent(
                target_id="node-1/gpu-4",
                event_type="hardware.gpu.fail",
                time=1000,
                duration=None,
            )],
            sim.queue,
        )

        assert len(scheduled) == 1
        assert scheduled[0].time == 1000
        assert scheduled[0].payload.data["component_id"] == "node-1/gpu-4"
        assert scheduled[0].payload.data["gpu_id"] == "node-1/gpu-4"

        fired: list[tuple[int, str]] = []

        def handler(event: Event, ctx):
            fired.append((ctx.clock.now(), event.payload.event_type))

        sim.register_handler("hardware.gpu.fail", handler)
        sim.run()

        assert fired == [(1000, "hardware.gpu.fail")]


class TestChaosInjectorAutoRepair:
    """Fail with duration=5000 -> fail event + repair event scheduled."""

    def test_fail_and_repair_events_scheduled(self):
        sim = SimulationLoop()
        injector = ChaosInjector()

        scheduled = injector.inject(
            [ChaosEvent(
                target_id="node-1/gpu-4",
                event_type="hardware.gpu.fail",
                time=1000,
                duration=5000,
            )],
            sim.queue,
        )

        assert len(scheduled) == 2
        assert scheduled[0].time == 1000
        assert scheduled[0].payload.event_type == "hardware.gpu.fail"
        assert scheduled[1].time == 6000
        assert scheduled[1].payload.event_type == "hardware.gpu.repair"

        event_log: list[tuple[int, str]] = []

        def handler(event: Event, ctx):
            event_log.append((ctx.clock.now(), event.payload.event_type))

        sim.register_handler("hardware.gpu.fail", handler)
        sim.register_handler("hardware.gpu.repair", handler)
        sim.run()

        assert event_log == [
            (1000, "hardware.gpu.fail"),
            (6000, "hardware.gpu.repair"),
        ]
