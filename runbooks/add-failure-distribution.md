# Adding a New Failure Distribution

## Overview
How to add a new probabilistic failure model (e.g., Weibull bathtub curve, correlated failures, bursty failures) for auto-generating chaos events.

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/chaos/distributions.py` | New class implementing `FailureDistribution` protocol | Defines the statistical model |

That's it — one file. Distributions are self-contained.

## Step-by-Step

### 1. Implement the protocol (`chaos/distributions.py`)

The protocol requires two methods:

```python
class FailureDistribution(Protocol):
    def sample_next_failure_time(self, rng: random.Random, now: SimTime) -> SimTime:
        """Return the absolute time of the next failure."""
        ...

    def sample_repair_time(self, rng: random.Random) -> SimTime:
        """Return the duration of repair (added to failure time)."""
        ...
```

### 2. Example: Weibull distribution (bathtub curve)

```python
import math
import random
from dataclasses import dataclass
from dcsim.engine.clock import SimTime

@dataclass
class WeibullFailure:
    """Weibull distribution models the 'bathtub curve':
    - shape < 1: early-life failures (infant mortality)
    - shape = 1: constant failure rate (same as exponential)
    - shape > 1: wear-out failures (aging)
    """
    shape: float          # Weibull shape parameter (k)
    scale_us: SimTime     # Weibull scale parameter (lambda) in microseconds
    mttr_us: SimTime      # Mean time to repair

    def sample_next_failure_time(self, rng: random.Random, now: SimTime) -> SimTime:
        # Weibull: t = scale * (-ln(U))^(1/shape)
        u = rng.random()
        t = self.scale_us * (-math.log(u)) ** (1.0 / self.shape)
        return now + int(t)

    def sample_repair_time(self, rng: random.Random) -> SimTime:
        # Exponential repair time
        return int(rng.expovariate(1.0 / self.mttr_us))
```

### 3. Example: Correlated failures

```python
@dataclass
class CorrelatedFailure:
    """When one component fails, nearby components have elevated failure probability."""
    base_distribution: FailureDistribution
    correlation_factor: float = 5.0    # Failure rate multiplier for neighbors
    correlation_window_us: SimTime = 60 * SECOND  # Time window for correlation

    def sample_next_failure_time(self, rng: random.Random, now: SimTime) -> SimTime:
        return self.base_distribution.sample_next_failure_time(rng, now)

    def sample_repair_time(self, rng: random.Random) -> SimTime:
        return self.base_distribution.sample_repair_time(rng)

    def sample_correlated_failures(
        self, rng: random.Random, primary_failure_time: SimTime, neighbor_ids: list[str]
    ) -> list[tuple[str, SimTime]]:
        """Returns (component_id, failure_time) for neighbors that also fail."""
        correlated = []
        for nid in neighbor_ids:
            if rng.random() < (1.0 / self.correlation_factor):
                delay = int(rng.expovariate(1.0 / self.correlation_window_us))
                correlated.append((nid, primary_failure_time + delay))
        return correlated
```

### 4. Use with ChaosInjector

```python
injector = ChaosInjector()
dist = WeibullFailure(shape=0.5, scale_us=1_000 * SECOND, mttr_us=10 * SECOND)

# Generate failures for all GPUs over a 1-hour simulation
injector.inject_from_distribution(
    dist=dist,
    components=[gpu.id for gpu in graph.get_gpus()],
    time_range=(0, 3600 * SECOND),
    queue=sim.queue,
    rng=random.Random(42),  # Seeded for determinism!
)
```

## Key Rules

1. **Always use the `rng` parameter** — never call `random.random()` directly. The seeded `random.Random` instance ensures deterministic simulation replay.
2. **Return absolute times** from `sample_next_failure_time`, not deltas. The `now` parameter is provided so you can compute `now + delta`.
3. **Repair times are durations**, not absolute times. The caller adds them to the failure time.

## Checklist
- [ ] Implements both `sample_next_failure_time` and `sample_repair_time`
- [ ] Uses only the provided `rng`, never global `random`
- [ ] Returns integer microseconds (`SimTime`)
- [ ] Tests: statistical validation (sample 1000+ times, verify mean/shape)
- [ ] Tests: determinism (same seed → same sequence)
- [ ] `pytest tests/ -v` passes
