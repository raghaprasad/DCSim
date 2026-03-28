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
    """Build a Gantt-style chart showing per-GPU-group behavior.

    For scenarios with throttling or failure, shows three rows:
      - Affected GPU:   compute (slow/orange) for the full degraded duration
      - Healthy GPUs:   compute (green, normal speed) then IDLE (gray, waiting)
      - All 32 GPUs:    communicate (blue, after sync barrier)

    For the baseline (no chaos), shows two rows:
      - All GPUs: compute (green)
      - All GPUs: communicate (blue)
    """
    # --- Extract phase intervals ---
    phase_intervals: list[dict[str, Any]] = []
    active_phases: dict[str, dict[str, Any]] = {}

    for entry in timeline:
        t_ms = entry.timestamp / 1_000.0
        if entry.event_type == "workload.phase.start":
            job_id = entry.data.get("job_id", "")
            active_phases[job_id] = {
                "phase": entry.data.get("phase", ""),
                "step": entry.data.get("step", 0),
                "start_ms": t_ms,
                "duration_us": entry.data.get("duration_us", 0),
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
                    "duration_us": info["duration_us"],
                })

    # --- Detect affected GPUs and timing ---
    affected_gpus: dict[str, dict[str, Any]] = {}  # gpu_id -> {state, time, throttle_factor}
    for entry in timeline:
        if entry.event_type == "hardware.gpu.throttle":
            gid = entry.component_id or ""
            affected_gpus[gid] = {
                "state": "throttled",
                "time_ms": entry.timestamp / 1_000.0,
                "throttle_factor": entry.data.get("throttle_factor", 0.33),
            }
        elif entry.event_type == "hardware.gpu.fail":
            gid = entry.component_id or ""
            affected_gpus[gid] = {
                "state": "failed",
                "time_ms": entry.timestamp / 1_000.0,
            }

    # Derive baseline compute duration from the first compute phase (before chaos)
    baseline_compute_ms = 100.0  # default
    for iv in phase_intervals:
        if iv["phase"] == "compute":
            baseline_compute_ms = iv["end_ms"] - iv["start_ms"]
            break

    # --- Build figure ---
    fig = go.Figure()

    COLORS = {
        "compute": "#3780bf",        # blue
        "communicate": "#32ab60",    # green
        "idle_wait": "#95a5a6",      # gray — waiting at sync barrier
        "throttled": "#f39c12",      # orange
        "failed": "#e74c3c",         # red
    }

    has_affected = len(affected_gpus) > 0
    affected_label = ""
    healthy_count = 32

    if has_affected:
        gid = next(iter(affected_gpus))
        info = affected_gpus[gid]
        state = info["state"]
        affected_label = f"{gid} ({state})"
        healthy_count = 31

        if state == "throttled":
            _build_throttle_timeline(
                fig, phase_intervals, gid, info,
                baseline_compute_ms, healthy_count,
                affected_label, COLORS,
            )
        elif state == "failed":
            _build_failure_timeline(
                fig, phase_intervals, gid, info,
                baseline_compute_ms, healthy_count,
                affected_label, COLORS, timeline,
            )
    else:
        # Baseline — simple two-row view
        for iv in phase_intervals:
            dur = iv["end_ms"] - iv["start_ms"]
            phase = iv["phase"]
            color = COLORS.get(phase, COLORS["idle_wait"])
            label = "All 32 GPUs"
            fig.add_trace(go.Bar(
                y=[label], x=[dur], base=[iv["start_ms"]], orientation="h",
                marker_color=color, showlegend=False,
                hovertext=f"Step {iv['step']}: {phase}<br>"
                          f"{iv['start_ms']:.1f} - {iv['end_ms']:.1f}ms ({dur:.1f}ms)",
                hoverinfo="text",
            ))

    # --- Legend ---
    legend_items = [
        ("Compute", COLORS["compute"]),
        ("Communicate", COLORS["communicate"]),
        ("Idle (waiting)", COLORS["idle_wait"]),
    ]
    if has_affected:
        state = next(iter(affected_gpus.values()))["state"]
        legend_items.append((state.capitalize(), COLORS[state]))

    for name, color in legend_items:
        fig.add_trace(go.Bar(
            y=[None], x=[None], orientation="h",
            marker_color=color, name=name, showlegend=True,
        ))

    row_count = 3 if has_affected else 1
    fig.update_layout(
        title="GPU State Timeline — Synchronous Training Barrier",
        xaxis_title="Time (ms)",
        yaxis_title="",
        barmode="overlay",
        height=max(300, 120 * row_count + 100),
        showlegend=True,
        yaxis=dict(autorange="reversed"),
    )

    return fig


def _build_throttle_timeline(
    fig: go.Figure,
    phase_intervals: list[dict[str, Any]],
    gpu_id: str,
    gpu_info: dict[str, Any],
    baseline_compute_ms: float,
    healthy_count: int,
    affected_label: str,
    colors: dict[str, str],
) -> None:
    """Add bars for a throttle scenario: affected GPU, healthy GPUs, all-GPU comms."""
    throttle_time_ms = gpu_info["time_ms"]

    row_affected = f"  {affected_label}"
    row_healthy = f"  {healthy_count} Healthy GPUs"
    row_comms = f"  All 32 GPUs (comms)"

    for iv in phase_intervals:
        start = iv["start_ms"]
        end = iv["end_ms"]
        dur = end - start
        step = iv["step"]

        if iv["phase"] == "communicate":
            # Comms: all GPUs together
            fig.add_trace(go.Bar(
                y=[row_comms], x=[dur], base=[start], orientation="h",
                marker_color=colors["communicate"], showlegend=False,
                hovertext=f"Step {step}: allreduce<br>"
                          f"{start:.1f} - {end:.1f}ms ({dur:.1f}ms)",
                hoverinfo="text",
            ))

        elif iv["phase"] == "compute":
            is_degraded = start >= throttle_time_ms or (start < throttle_time_ms < end)

            if not is_degraded:
                # Normal compute — all GPUs same speed
                fig.add_trace(go.Bar(
                    y=[row_healthy], x=[dur], base=[start], orientation="h",
                    marker_color=colors["compute"], showlegend=False,
                    hovertext=f"Step {step}: compute<br>"
                              f"{start:.1f} - {end:.1f}ms ({dur:.1f}ms)",
                    hoverinfo="text",
                ))
                fig.add_trace(go.Bar(
                    y=[row_affected], x=[dur], base=[start], orientation="h",
                    marker_color=colors["compute"], showlegend=False,
                    hovertext=f"Step {step}: compute<br>"
                              f"{start:.1f} - {end:.1f}ms ({dur:.1f}ms)",
                    hoverinfo="text",
                ))
            else:
                # Degraded: healthy GPUs finish early then wait
                healthy_end = start + baseline_compute_ms
                if healthy_end > end:
                    healthy_end = end  # clamp
                wait_dur = end - healthy_end

                # Healthy GPUs: compute (normal speed)
                fig.add_trace(go.Bar(
                    y=[row_healthy], x=[baseline_compute_ms], base=[start],
                    orientation="h", marker_color=colors["compute"], showlegend=False,
                    hovertext=f"Step {step}: compute (healthy)<br>"
                              f"{start:.1f} - {healthy_end:.1f}ms ({baseline_compute_ms:.1f}ms)",
                    hoverinfo="text",
                ))

                # Healthy GPUs: idle/waiting for slow GPU
                if wait_dur > 0.5:
                    fig.add_trace(go.Bar(
                        y=[row_healthy], x=[wait_dur], base=[healthy_end],
                        orientation="h", marker_color=colors["idle_wait"], showlegend=False,
                        hovertext=f"Step {step}: IDLE — waiting for {gpu_id}<br>"
                                  f"{healthy_end:.1f} - {end:.1f}ms ({wait_dur:.1f}ms wasted)",
                        hoverinfo="text",
                    ))

                # Affected GPU: full throttled compute
                fig.add_trace(go.Bar(
                    y=[row_affected], x=[dur], base=[start], orientation="h",
                    marker_color=colors["throttled"], showlegend=False,
                    hovertext=f"Step {step}: compute (throttled)<br>"
                              f"{start:.1f} - {end:.1f}ms ({dur:.1f}ms — "
                              f"{dur/baseline_compute_ms:.1f}x baseline)",
                    hoverinfo="text",
                ))


def _build_failure_timeline(
    fig: go.Figure,
    phase_intervals: list[dict[str, Any]],
    gpu_id: str,
    gpu_info: dict[str, Any],
    baseline_compute_ms: float,
    healthy_count: int,
    affected_label: str,
    colors: dict[str, str],
    timeline: list[LogEntry],
) -> None:
    """Add bars for a GPU failure scenario: failed GPU, healthy GPUs, comms."""
    fail_time_ms = gpu_info["time_ms"]

    # Find repair time
    repair_time_ms = None
    for entry in timeline:
        if entry.event_type == "hardware.gpu.repair" and (entry.component_id or "") == gpu_id:
            repair_time_ms = entry.timestamp / 1_000.0
            break

    row_affected = f"  {affected_label}"
    row_healthy = f"  {healthy_count} Healthy GPUs"
    row_comms = f"  All 32 GPUs (comms)"

    # Show the failure gap on the affected GPU row
    if repair_time_ms is not None:
        gap = repair_time_ms - fail_time_ms
        fig.add_trace(go.Bar(
            y=[row_affected], x=[gap], base=[fail_time_ms], orientation="h",
            marker_color=colors["failed"], showlegend=False,
            hovertext=f"{gpu_id}: FAILED<br>"
                      f"{fail_time_ms:.1f} - {repair_time_ms:.1f}ms "
                      f"({gap:.0f}ms downtime — detection + reboot + reload)",
            hoverinfo="text",
        ))

    # All GPUs idle during the failure gap
    if repair_time_ms is not None:
        gap = repair_time_ms - fail_time_ms
        fig.add_trace(go.Bar(
            y=[row_healthy], x=[gap], base=[fail_time_ms], orientation="h",
            marker_color=colors["idle_wait"], showlegend=False,
            hovertext=f"{healthy_count} GPUs: IDLE — waiting for {gpu_id} repair<br>"
                      f"{fail_time_ms:.1f} - {repair_time_ms:.1f}ms ({gap:.0f}ms wasted)",
            hoverinfo="text",
        ))

    for iv in phase_intervals:
        start = iv["start_ms"]
        end = iv["end_ms"]
        dur = end - start
        step = iv["step"]

        if iv["phase"] == "communicate":
            fig.add_trace(go.Bar(
                y=[row_comms], x=[dur], base=[start], orientation="h",
                marker_color=colors["communicate"], showlegend=False,
                hovertext=f"Step {step}: allreduce<br>{start:.1f} - {end:.1f}ms",
                hoverinfo="text",
            ))
        elif iv["phase"] == "compute":
            # Show compute on both rows (all same speed for non-throttle)
            fig.add_trace(go.Bar(
                y=[row_healthy], x=[dur], base=[start], orientation="h",
                marker_color=colors["compute"], showlegend=False,
                hovertext=f"Step {step}: compute<br>{start:.1f} - {end:.1f}ms",
                hoverinfo="text",
            ))
            fig.add_trace(go.Bar(
                y=[row_affected], x=[dur], base=[start], orientation="h",
                marker_color=colors["compute"], showlegend=False,
                hovertext=f"Step {step}: compute<br>{start:.1f} - {end:.1f}ms",
                hoverinfo="text",
            ))


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


def _build_datacenter_svg(timeline: list[LogEntry]) -> str:
    """Build an SVG diagram of the datacenter with affected components color-coded.

    Layout:
        spine-0  spine-1          (top, spine switches)
         |  \\  /  |
        tor-0    tor-1            (ToR switches)
         |         |
      +------+ +------+   +------+ +------+
      |node-0| |node-1|   |node-2| |node-3|
      | 8GPU | | 8GPU |   | 8GPU | | 8GPU |
      +------+ +------+   +------+ +------+
       \\__ rack 0 __/      \\__ rack 1 __/
    """
    # Determine which components were affected
    failed_gpus: set[str] = set()
    throttled_gpus: set[str] = set()
    failed_links: set[str] = set()
    failed_switches: set[str] = set()

    for entry in timeline:
        cid = entry.component_id or ""
        if entry.event_type == "hardware.gpu.fail":
            failed_gpus.add(cid)
        elif entry.event_type == "hardware.gpu.throttle":
            throttled_gpus.add(cid)
        elif entry.event_type in ("hardware.link.fail", "cascade.link.down"):
            failed_links.add(cid)
        elif entry.event_type == "hardware.switch.fail":
            failed_switches.add(cid)

    def _gpu_color(node_idx: int, gpu_idx: int) -> str:
        gid = f"node-{node_idx}/gpu-{gpu_idx}"
        if gid in failed_gpus:
            return "#e74c3c"  # red
        if gid in throttled_gpus:
            return "#f39c12"  # orange
        return "#2ecc71"  # green

    def _switch_color(switch_id: str) -> str:
        if switch_id in failed_switches:
            return "#e74c3c"
        return "#3498db"

    def _link_affected(link_id: str) -> bool:
        return any(link_id in lid or lid in link_id for lid in failed_links)

    # SVG dimensions
    svg_w, svg_h = 720, 460
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
                 f'style="max-width:{svg_w}px; font-family:-apple-system,sans-serif;">')

    # Background
    lines.append(f'<rect width="{svg_w}" height="{svg_h}" fill="#1a1a2e" rx="8"/>')

    # --- Spine switches (top) ---
    spine_y = 40
    spine_positions = {"spine-0": 260, "spine-1": 460}
    for sid, sx in spine_positions.items():
        col = _switch_color(sid)
        lines.append(f'<rect x="{sx-35}" y="{spine_y-15}" width="70" height="30" rx="4" fill="{col}" opacity="0.9"/>')
        lines.append(f'<text x="{sx}" y="{spine_y+5}" text-anchor="middle" fill="white" font-size="11" font-weight="600">{sid}</text>')

    # --- ToR switches ---
    tor_y = 130
    tor_positions = {"tor-0": 180, "tor-1": 540}
    for tid, tx in tor_positions.items():
        col = _switch_color(tid)
        lines.append(f'<rect x="{tx-30}" y="{tor_y-15}" width="60" height="30" rx="4" fill="{col}" opacity="0.9"/>')
        lines.append(f'<text x="{tx}" y="{tor_y+5}" text-anchor="middle" fill="white" font-size="11" font-weight="600">{tid}</text>')

    # --- Spine-to-ToR links ---
    for tid, tx in tor_positions.items():
        for sid, sx in spine_positions.items():
            lid = f"link-{tid}-{sid}"
            stroke = "#e74c3c" if _link_affected(lid) else "#555"
            sw = "3" if _link_affected(lid) else "1.5"
            dash = 'stroke-dasharray="6,4"' if _link_affected(lid) else ""
            lines.append(f'<line x1="{sx}" y1="{spine_y+15}" x2="{tx}" y2="{tor_y-15}" '
                         f'stroke="{stroke}" stroke-width="{sw}" {dash}/>')

    # --- Rack backgrounds ---
    rack_y = 175
    rack_h = 240
    lines.append(f'<rect x="30" y="{rack_y}" width="300" height="{rack_h}" rx="6" fill="#16213e" stroke="#2c3e50" stroke-width="1.5"/>')
    lines.append(f'<text x="180" y="{rack_y+20}" text-anchor="middle" fill="#7f8c8d" font-size="12" font-weight="600">RACK 0</text>')
    lines.append(f'<rect x="390" y="{rack_y}" width="300" height="{rack_h}" rx="6" fill="#16213e" stroke="#2c3e50" stroke-width="1.5"/>')
    lines.append(f'<text x="540" y="{rack_y+20}" text-anchor="middle" fill="#7f8c8d" font-size="12" font-weight="600">RACK 1</text>')

    # --- Nodes with GPUs ---
    # rack 0: node-0, node-1 ; rack 1: node-2, node-3
    node_layout = [
        (0, 55, rack_y + 35),    # node-0 in rack 0
        (1, 185, rack_y + 35),   # node-1 in rack 0
        (2, 415, rack_y + 35),   # node-2 in rack 1
        (3, 545, rack_y + 35),   # node-3 in rack 1
    ]

    for node_idx, nx_, ny in node_layout:
        # Node box
        nw, nh = 120, 195
        lines.append(f'<rect x="{nx_}" y="{ny}" width="{nw}" height="{nh}" rx="4" fill="#0f3460" stroke="#2c3e50"/>')
        lines.append(f'<text x="{nx_+nw//2}" y="{ny+18}" text-anchor="middle" fill="#ecf0f1" font-size="11" font-weight="600">node-{node_idx}</text>')

        # 8 GPUs in a 4x2 grid
        gpu_start_x = nx_ + 10
        gpu_start_y = ny + 28
        gpu_w, gpu_h = 22, 18
        gpu_gap_x, gpu_gap_y = 4, 4

        for g in range(8):
            row, col = divmod(g, 4)
            gx = gpu_start_x + col * (gpu_w + gpu_gap_x)
            gy = gpu_start_y + row * (gpu_h + gpu_gap_y)
            color = _gpu_color(node_idx, g)
            lines.append(f'<rect x="{gx}" y="{gy}" width="{gpu_w}" height="{gpu_h}" rx="2" fill="{color}" opacity="0.85"/>')
            lines.append(f'<text x="{gx+gpu_w//2}" y="{gy+13}" text-anchor="middle" fill="white" font-size="8">G{g}</text>')

        # ToR link
        tor_id = "tor-0" if node_idx < 2 else "tor-1"
        tx = tor_positions[tor_id]
        link_lid = f"link-node-{node_idx}/gpu-0-{tor_id}"
        stroke = "#e74c3c" if _link_affected(link_lid) else "#444"
        lines.append(f'<line x1="{nx_+nw//2}" y1="{ny}" x2="{tx}" y2="{tor_y+15}" '
                     f'stroke="{stroke}" stroke-width="1" opacity="0.6"/>')

    # --- Legend ---
    legend_y = svg_h - 30
    legend_items = [
        ("#2ecc71", "Healthy"),
        ("#f39c12", "Throttled"),
        ("#e74c3c", "Failed"),
        ("#3498db", "Switch OK"),
    ]
    lx = 30
    for color, label in legend_items:
        lines.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" rx="2" fill="{color}"/>')
        lines.append(f'<text x="{lx+20}" y="{legend_y+11}" fill="#bdc3c7" font-size="11">{label}</text>')
        lx += 100

    lines.append("</svg>")
    return "\n".join(lines)


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
    dc_svg = _build_datacenter_svg(timeline)

    # Build combined HTML
    buf = io.StringIO()
    buf.write(f"<html><head><title>{title}</title>")
    buf.write("<style>body { font-family: -apple-system, sans-serif; margin: 20px; }</style>")
    buf.write("</head><body>\n")
    buf.write(f"<h1>{title}</h1>\n")

    # Datacenter diagram first
    buf.write("<h2>Datacenter Topology</h2>\n")
    buf.write(dc_svg)
    buf.write("\n<hr>\n")

    # Plotly charts
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
