# Adding a New Visualization

## Overview
How to add a new chart type to the demo HTML report (e.g., network topology diagram, failure cascade tree, throughput over time).

## Files to Modify

| File | What to change | Why |
|------|---------------|-----|
| `src/dcsim/visualize.py` | Add new chart generation function | Creates the Plotly figure |
| `src/dcsim/demo.py` | Call the new function and include in report | Wires it into the HTML output |

## Step-by-Step

### 1. Add a chart function (`visualize.py`)

Follow the existing pattern. Each function takes log data and returns a Plotly figure:

```python
import plotly.graph_objects as go

def create_throughput_chart(log_entries: list[dict], title: str = "") -> go.Figure:
    """Line chart showing training throughput (steps/second) over time."""
    step_events = [e for e in log_entries if e["event_type"] == "workload.step.complete"]

    times_ms = [e["timestamp"] / 1000 for e in step_events]
    steps = list(range(1, len(step_events) + 1))

    # Calculate instantaneous throughput
    throughput = []
    for i in range(1, len(times_ms)):
        dt = times_ms[i] - times_ms[i-1]
        throughput.append(1000.0 / dt if dt > 0 else 0)  # steps/sec

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_ms[1:],
        y=throughput,
        mode="lines+markers",
        name="Throughput",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Steps/second",
    )
    return fig
```

### 2. Include in the HTML report (`demo.py`)

The demo runner assembles figures into a single HTML file:

```python
from dcsim.visualize import create_gpu_timeline, create_throughput_chart

def generate_report(scenarios: dict[str, list[dict]]) -> str:
    figures = []
    for name, logs in scenarios.items():
        figures.append(create_gpu_timeline(logs, title=f"{name}: GPU States"))
        figures.append(create_throughput_chart(logs, title=f"{name}: Throughput"))

    # Combine into single HTML
    html = "<html><body>"
    for fig in figures:
        html += fig.to_html(full_html=False, include_plotlyjs="cdn")
    html += "</body></html>"
    return html
```

## Available Data for Charts

The event logger produces `list[dict]` with these fields:

```python
{
    "timestamp": 320000,            # SimTime (microseconds)
    "event_id": "a1b2c3d4e5f6",
    "parent_event_id": "f6e5d4c3b2a1",  # Causal parent (or None)
    "event_type": "workload.phase.complete",
    "component_id": "node-1/gpu-4",  # Or None
    "job_id": "train-1",            # Or None
    "description": "...",
    "data": { ... },                # Event-specific payload
}
```

## Chart Ideas

| Chart | Data source | Value |
|-------|------------|-------|
| GPU State Timeline (heatmap) | All events with `component_id` matching `gpu-*` | Shows idle/compute/throttled/failed per GPU over time |
| Iteration Duration (bar) | `workload.step.complete` timestamps | Shows which iterations were slow |
| Failure Cascade Tree | `parent_event_id` chains | Shows root cause → effects |
| Network Health | `hardware.link.*` events | Shows link up/down over time |
| Throughput Over Time | `workload.step.complete` deltas | Shows steps/second degradation |
| Comparison (multi-bar) | Baseline vs chaos scenario step times | Side-by-side impact |

## Checklist
- [ ] Function takes `list[dict]` (log entries) and returns `go.Figure`
- [ ] Chart has clear title, axis labels, and legend
- [ ] Added to the HTML report assembly in `demo.py`
- [ ] Tested: generates valid HTML that opens in browser
- [ ] Chaos events are visually distinguishable (red markers, annotations)
