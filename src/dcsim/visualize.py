"""Visualization: Plotly HTML generation for simulation results.

Generates:
1. GPU state timeline (horizontal bar chart / Gantt-style)
2. Iteration durations (grouped bar chart)
3. Event log table

Combines all into a single HTML file with include_plotlyjs='cdn'.
"""

from __future__ import annotations

import io
from typing import Any

import plotly.graph_objects as go

from dcsim.observer.logger import LogEntry


def _build_gpu_timeline(timeline: list[LogEntry]) -> go.Figure:
    """Build a Gantt-style horizontal bar chart of GPU states over time.

    Tracks phase start/complete events for each GPU to show compute vs communicate
    intervals, plus failure/throttle intervals.
    """
    # Collect phase intervals from workload events
    phase_intervals: list[dict[str, Any]] = []

    # Track active phases by job_id
    active_phases: dict[str, dict[str, Any]] = {}

    # Track GPU-level state changes
    gpu_state_intervals: list[dict[str, Any]] = []
    gpu_active_state: dict[str, dict[str, Any]] = {}

    for entry in timeline:
        t_ms = entry.timestamp / 1_000.0

        if entry.event_type == "workload.phase.start":
            phase = entry.data.get("phase", "")
            step = entry.data.get("step", 0)
            job_id = entry.data.get("job_id", "")
            active_phases[job_id] = {
                "phase": phase,
                "step": step,
                "start_ms": t_ms,
            }

        elif entry.event_type == "workload.phase.complete":
            job_id = entry.data.get("job_id", "")
            if job_id in active_phases:
                info = active_phases.pop(job_id)
                phase_intervals.append({
                    "phase": info["phase"],
                    "step": info["step"],
                    "start_ms": info["start_ms"],
                    "end_ms": t_ms,
                })

        elif entry.event_type in ("hardware.gpu.fail", "hardware.gpu.throttle"):
            gpu_id = entry.component_id or ""
            state = "failed" if "fail" in entry.event_type else "throttled"
            gpu_active_state[gpu_id] = {"state": state, "start_ms": t_ms}

        elif entry.event_type in ("hardware.gpu.repair", "hardware.gpu.unthrottle"):
            gpu_id = entry.component_id or ""
            if gpu_id in gpu_active_state:
                info = gpu_active_state.pop(gpu_id)
                gpu_state_intervals.append({
                    "gpu_id": gpu_id,
                    "state": info["state"],
                    "start_ms": info["start_ms"],
                    "end_ms": t_ms,
                })

    fig = go.Figure()

    # Color mapping for phases
    phase_colors = {
        "compute": "rgba(55, 128, 191, 0.7)",
        "communicate": "rgba(50, 171, 96, 0.7)",
    }

    # Plot workload phases as bars at y="All GPUs" (since AllReduce is synchronous)
    for interval in phase_intervals:
        duration = interval["end_ms"] - interval["start_ms"]
        color = phase_colors.get(interval["phase"], "rgba(128, 128, 128, 0.7)")
        fig.add_trace(go.Bar(
            y=["All GPUs"],
            x=[duration],
            base=[interval["start_ms"]],
            orientation="h",
            marker_color=color,
            name=f"Step {interval['step']} {interval['phase']}",
            showlegend=False,
            hovertext=f"Step {interval['step']}: {interval['phase']}<br>"
                       f"{interval['start_ms']:.1f}ms - {interval['end_ms']:.1f}ms<br>"
                       f"Duration: {duration:.1f}ms",
            hoverinfo="text",
        ))

    # Plot GPU state intervals (failures, throttles)
    state_colors = {
        "failed": "rgba(219, 64, 82, 0.9)",
        "throttled": "rgba(255, 165, 0, 0.9)",
    }

    for interval in gpu_state_intervals:
        duration = interval["end_ms"] - interval["start_ms"]
        color = state_colors.get(interval["state"], "rgba(128, 128, 128, 0.9)")
        fig.add_trace(go.Bar(
            y=[interval["gpu_id"]],
            x=[duration],
            base=[interval["start_ms"]],
            orientation="h",
            marker_color=color,
            name=f"{interval['gpu_id']} {interval['state']}",
            showlegend=False,
            hovertext=f"{interval['gpu_id']}: {interval['state']}<br>"
                       f"{interval['start_ms']:.1f}ms - {interval['end_ms']:.1f}ms",
            hoverinfo="text",
        ))

    # Add legend entries for phase types
    for phase, color in phase_colors.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None], orientation="h",
            marker_color=color, name=phase.capitalize(), showlegend=True,
        ))
    for state, color in state_colors.items():
        fig.add_trace(go.Bar(
            y=[None], x=[None], orientation="h",
            marker_color=color, name=state.capitalize(), showlegend=True,
        ))

    fig.update_layout(
        title="GPU State Timeline",
        xaxis_title="Time (ms)",
        yaxis_title="Component",
        barmode="overlay",
        height=max(300, 50 * (len(gpu_state_intervals) + 2)),
        showlegend=True,
    )

    return fig


def _build_iteration_durations(timeline: list[LogEntry]) -> go.Figure:
    """Build a grouped bar chart of compute vs communicate durations per step."""
    # Collect phase durations per step
    step_durations: dict[int, dict[str, float]] = {}
    active_phases: dict[str, dict[str, Any]] = {}

    for entry in timeline:
        t_ms = entry.timestamp / 1_000.0

        if entry.event_type == "workload.phase.start":
            job_id = entry.data.get("job_id", "")
            active_phases[job_id] = {
                "phase": entry.data.get("phase", ""),
                "step": entry.data.get("step", 0),
                "start_ms": t_ms,
            }

        elif entry.event_type == "workload.phase.complete":
            job_id = entry.data.get("job_id", "")
            if job_id in active_phases:
                info = active_phases.pop(job_id)
                step = info["step"]
                phase = info["phase"]
                duration = t_ms - info["start_ms"]

                if step not in step_durations:
                    step_durations[step] = {}
                step_durations[step][phase] = duration

    if not step_durations:
        fig = go.Figure()
        fig.update_layout(title="Iteration Durations (no data)")
        return fig

    steps = sorted(step_durations.keys())
    compute_durations = [step_durations[s].get("compute", 0) for s in steps]
    comms_durations = [step_durations[s].get("communicate", 0) for s in steps]
    total_durations = [c + m for c, m in zip(compute_durations, comms_durations)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"Step {s}" for s in steps],
        y=compute_durations,
        name="Compute",
        marker_color="rgba(55, 128, 191, 0.7)",
    ))

    fig.add_trace(go.Bar(
        x=[f"Step {s}" for s in steps],
        y=comms_durations,
        name="Communicate",
        marker_color="rgba(50, 171, 96, 0.7)",
    ))

    fig.add_trace(go.Bar(
        x=[f"Step {s}" for s in steps],
        y=total_durations,
        name="Total",
        marker_color="rgba(128, 128, 128, 0.3)",
    ))

    fig.update_layout(
        title="Iteration Durations (ms)",
        xaxis_title="Training Step",
        yaxis_title="Duration (ms)",
        barmode="group",
        height=400,
    )

    return fig


def _build_event_log_table(timeline: list[LogEntry]) -> go.Figure:
    """Build a Plotly table figure from the event log."""
    timestamps = []
    event_types = []
    component_ids = []
    job_ids = []
    descriptions = []

    for entry in timeline:
        t_ms = entry.timestamp / 1_000.0
        timestamps.append(f"{t_ms:.3f}")
        event_types.append(entry.event_type)
        component_ids.append(entry.component_id or "")
        job_ids.append(entry.job_id or "")
        descriptions.append(entry.description[:80] if entry.description else "")

    # Color-code rows by event type category
    row_colors = []
    for et in event_types:
        if "fail" in et or "interrupted" in et:
            row_colors.append("rgba(219, 64, 82, 0.15)")
        elif "throttle" in et:
            row_colors.append("rgba(255, 165, 0, 0.15)")
        elif "repair" in et or "unthrottle" in et:
            row_colors.append("rgba(50, 171, 96, 0.15)")
        elif "workload" in et:
            row_colors.append("rgba(55, 128, 191, 0.10)")
        else:
            row_colors.append("white")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Time (ms)", "Event Type", "Component", "Job", "Description"],
            fill_color="rgba(55, 128, 191, 0.8)",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[timestamps, event_types, component_ids, job_ids, descriptions],
            fill_color=[row_colors],
            align="left",
            font=dict(size=11),
            height=25,
        ),
    )])

    fig.update_layout(
        title="Event Log",
        height=max(400, 30 * len(timestamps) + 100),
    )

    return fig


def generate_html(
    timeline: list[LogEntry],
    title: str = "DCSim Simulation Report",
) -> str:
    """Generate a single HTML page with all visualization figures.

    Uses include_plotlyjs='cdn' so the HTML is self-contained (loads plotly from CDN).
    """
    fig_timeline = _build_gpu_timeline(timeline)
    fig_iterations = _build_iteration_durations(timeline)
    fig_table = _build_event_log_table(timeline)

    # Build combined HTML
    buf = io.StringIO()
    buf.write(f"<html><head><title>{title}</title></head><body>\n")
    buf.write(f"<h1>{title}</h1>\n")

    # First figure includes plotly.js from CDN
    buf.write(fig_timeline.to_html(full_html=False, include_plotlyjs="cdn"))
    buf.write("<hr>\n")
    buf.write(fig_iterations.to_html(full_html=False, include_plotlyjs=False))
    buf.write("<hr>\n")
    buf.write(fig_table.to_html(full_html=False, include_plotlyjs=False))

    buf.write("\n</body></html>")
    return buf.getvalue()


def generate_report(
    timeline: list[LogEntry],
    output_path: str = "dcsim_report.html",
    title: str = "DCSim Simulation Report",
) -> str:
    """Generate HTML report and write to file. Returns the output path."""
    html = generate_html(timeline, title=title)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path
