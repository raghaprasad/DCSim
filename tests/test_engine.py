"""Phase 1 gating tests for the core simulation engine."""

import time

from dcsim.engine.clock import SimulationClock, MILLISECOND, SECOND
from dcsim.engine.event import Event, EventPayload, EventQueue
from dcsim.engine.loop import SimulationLoop


class TestDeterministicOrdering:
    """Test 1: Events at same timestamp execute in priority then insertion order."""

    def test_priority_ordering(self):
        execution_order: list[int] = []
        sim = SimulationLoop()

        # Schedule 100 events at t=0 with priorities 99 down to 0 (inserted in reverse)
        for i in range(100):
            priority = 99 - i
            sim.schedule(0, EventPayload(event_type="test", data={"p": priority}), priority=priority)

        def handler(event: Event, ctx):
            execution_order.append(event.payload.data["p"])

        sim.register_handler("test", handler)
        sim.run()

        # Should execute in priority order: 0, 1, 2, ..., 99
        assert execution_order == list(range(100))

    def test_deterministic_across_runs(self):
        """Same schedule produces identical execution both times."""
        def run_once() -> list[int]:
            order: list[int] = []
            sim = SimulationLoop()
            for i in range(50):
                p = (i * 7) % 10  # Scrambled priorities with duplicates
                sim.schedule(0, EventPayload(event_type="t", data={"i": i}), priority=p)

            def handler(event: Event, ctx):
                order.append(event.payload.data["i"])

            sim.register_handler("t", handler)
            sim.run()
            return order

        assert run_once() == run_once()


class TestTimeAdvancement:
    """Test 2: Clock reads correct time at each handler invocation."""

    def test_clock_at_handler_time(self):
        observed_times: list[int] = []
        sim = SimulationLoop()

        sim.schedule(0, EventPayload(event_type="tick"))
        sim.schedule(1000, EventPayload(event_type="tick"))
        sim.schedule(5000, EventPayload(event_type="tick"))

        def handler(event: Event, ctx):
            observed_times.append(ctx.clock.now())

        sim.register_handler("tick", handler)
        sim.run()

        assert observed_times == [0, 1000, 5000]

    def test_clock_cannot_go_backward(self):
        clock = SimulationClock()
        clock.advance_to(100)
        try:
            clock.advance_to(50)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestCancellation:
    """Test 3: Cancelled events do not fire."""

    def test_cancel_middle_event(self):
        fired: list[str] = []
        sim = SimulationLoop()

        sim.schedule(100, EventPayload(event_type="a", data={"name": "first"}))
        middle = sim.schedule(200, EventPayload(event_type="a", data={"name": "middle"}))
        sim.schedule(300, EventPayload(event_type="a", data={"name": "last"}))

        sim.queue.cancel(middle)

        def handler(event: Event, ctx):
            fired.append(event.payload.data["name"])

        sim.register_handler("a", handler)
        sim.run()

        assert fired == ["first", "last"]

    def test_double_cancel_returns_false(self):
        q = EventQueue()
        e = q.schedule(0, EventPayload(event_type="x"))
        assert q.cancel(e) is True
        assert q.cancel(e) is False


class TestHandlerChaining:
    """Test 4: Handlers can schedule future events that schedule further events."""

    def test_chain_of_three(self):
        trace: list[str] = []
        sim = SimulationLoop()

        def handler_a(event: Event, ctx):
            trace.append("A")
            ctx.queue.schedule(ctx.clock.now() + 100, EventPayload(event_type="b"))

        def handler_b(event: Event, ctx):
            trace.append("B")
            ctx.queue.schedule(ctx.clock.now() + 200, EventPayload(event_type="c"))

        def handler_c(event: Event, ctx):
            trace.append("C")

        sim.register_handler("a", handler_a)
        sim.register_handler("b", handler_b)
        sim.register_handler("c", handler_c)

        sim.schedule(0, EventPayload(event_type="a"))
        result = sim.run()

        assert trace == ["A", "B", "C"]
        assert result.final_time == 300

    def test_returned_payloads_scheduled_at_current_time(self):
        """Payloads returned from handler are scheduled at the current sim time."""
        times: list[int] = []
        sim = SimulationLoop()

        def handler_a(event: Event, ctx):
            times.append(ctx.clock.now())
            return [EventPayload(event_type="b")]

        def handler_b(event: Event, ctx):
            times.append(ctx.clock.now())

        sim.register_handler("a", handler_a)
        sim.register_handler("b", handler_b)
        sim.schedule(500, EventPayload(event_type="a"))
        sim.run()

        assert times == [500, 500]


class TestTermination:
    """Test 5: Empty queue terminates the loop cleanly."""

    def test_empty_queue(self):
        sim = SimulationLoop()
        result = sim.run()
        assert result.events_processed == 0
        assert result.stopped_by_empty_queue is True

    def test_step_on_empty_returns_false(self):
        sim = SimulationLoop()
        assert sim.step() is False


class TestMaxTimeBound:
    """Test 6: Events beyond max time do not fire."""

    def test_until_stops_before_event(self):
        fired = []
        sim = SimulationLoop()
        sim.schedule(1000, EventPayload(event_type="late"))

        def handler(event: Event, ctx):
            fired.append(True)

        sim.register_handler("late", handler)
        result = sim.run(until=500)

        assert fired == []
        assert result.stopped_by_max_time is True
        assert result.events_processed == 0

    def test_until_allows_earlier_events(self):
        fired = []
        sim = SimulationLoop()
        sim.schedule(100, EventPayload(event_type="early"))
        sim.schedule(1000, EventPayload(event_type="late"))

        def handler(event: Event, ctx):
            fired.append(event.payload.event_type)

        sim.register_handler("early", handler)
        sim.register_handler("late", handler)
        result = sim.run(until=500)

        assert fired == ["early"]
        assert result.stopped_by_max_time is True


class TestPerformance:
    """Test 7: 100,000 events processed in under 1 second."""

    def test_100k_events(self):
        sim = SimulationLoop()
        counter = {"n": 0}

        for i in range(100_000):
            sim.schedule(i, EventPayload(event_type="perf"))

        def handler(event: Event, ctx):
            counter["n"] += 1

        sim.register_handler("perf", handler)

        start = time.perf_counter()
        result = sim.run()
        elapsed = time.perf_counter() - start

        assert counter["n"] == 100_000
        assert result.events_processed == 100_000
        assert elapsed < 1.0, f"Took {elapsed:.3f}s, expected <1s"


class TestCausalChain:
    """Bonus: Verify parent_event_id is set on returned payloads."""

    def test_parent_event_id_propagation(self):
        parent_ids: list[str | None] = []
        sim = SimulationLoop()

        def handler_a(event: Event, ctx):
            return [EventPayload(event_type="child")]

        def handler_child(event: Event, ctx):
            parent_ids.append(event.payload.parent_event_id)

        sim.register_handler("root", handler_a)
        sim.register_handler("child", handler_child)

        root_event = sim.schedule(0, EventPayload(event_type="root"))
        sim.run()

        assert len(parent_ids) == 1
        assert parent_ids[0] == root_event.event_id
