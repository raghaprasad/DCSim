"""Entry point for ``python -m dcsim``.

Runs all 4 demo scenarios, prints a summary, and generates one HTML per scenario.

Usage:
    python -m dcsim                       # run all, print summary
    python -m dcsim --html output_dir     # also write per-scenario HTML reports
    python -m dcsim --scenario baseline   # run a single scenario
"""

from __future__ import annotations

import argparse
import os
import webbrowser

from dcsim.demo import (
    ScenarioResult,
    load_chaos_file,
    parse_chaos_string,
    print_summary,
    run_all_scenarios,
    run_scenario,
    scenario_baseline,
    scenario_gpu_failure,
    scenario_link_flap,
    scenario_thermal_throttle,
)


SCENARIOS = {
    "baseline": scenario_baseline,
    "gpu-failure": scenario_gpu_failure,
    "thermal-throttle": scenario_thermal_throttle,
    "link-flap": scenario_link_flap,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dcsim",
        description="DCSim: Datacenter Chaos Engineering Simulator demo runner.",
    )
    parser.add_argument(
        "--html",
        metavar="DIR",
        default="reports/",
        help="Write per-scenario HTML reports to DIR (default: reports/)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        default=False,
        help="Skip HTML report generation, terminal summary only",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default=None,
        help="Run a named scenario (default: run all if no --chaos given)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        default=True,
        help="Open HTML reports in browser (default: on)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        default=False,
        help="Don't open HTML reports in browser",
    )
    parser.add_argument(
        "--name",
        metavar="NAME",
        default=None,
        help="Name for the scenario (used in reports). Defaults to 'Custom chaos'.",
    )
    parser.add_argument(
        "--chaos",
        action="append",
        metavar="EVENT",
        help=(
            'Inject a custom chaos event. Format: "<type> <target> <time> [duration] [key=val ...]". '
            'Example: "gpu.throttle node-1/gpu-4 320ms 5s throttle_factor=0.33". '
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--chaos-file",
        metavar="PATH",
        help="Load chaos events from a JSON file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps (default: 10)",
    )
    args = parser.parse_args()

    if args.no_html:
        args.html = None
    if args.no_open:
        args.open = False

    # Run scenarios
    results: list[ScenarioResult]
    if args.chaos or args.chaos_file:
        # Custom chaos mode
        from dcsim.chaos.injector import ChaosEvent
        chaos_events: list[ChaosEvent] = []
        if args.chaos:
            for s in args.chaos:
                chaos_events.append(parse_chaos_string(s))
        if args.chaos_file:
            chaos_events.extend(load_chaos_file(args.chaos_file))
        results = [run_scenario(
            name=args.name or "Custom chaos",
            chaos_events=chaos_events,
            total_steps=args.steps,
        )]
    elif args.scenario and args.scenario != "all":
        results = [SCENARIOS[args.scenario]()]
    else:
        results = run_all_scenarios()

    print_summary(results)

    # Generate one HTML per scenario
    if args.html:
        from dcsim.visualize import generate_report

        out_dir = args.html
        os.makedirs(out_dir, exist_ok=True)

        paths: list[str] = []
        for r in results:
            slug = r.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            slug = slug.replace("/", "_").replace(",", "")
            filename = f"{slug}.html"
            path = os.path.join(out_dir, filename)
            generate_report(r.timeline, output_path=path, title=f"DCSim: {r.name}")
            paths.append(path)
            print(f"  {r.name:<35} -> {path}")

        # Generate an index page linking to all scenario pages
        index_path = os.path.join(out_dir, "index.html")
        _write_index(index_path, results, paths)
        print(f"\n  Index page -> {index_path}")

        if args.open:
            webbrowser.open(f"file://{os.path.abspath(index_path)}")


def _write_index(path: str, results: list[ScenarioResult], html_paths: list[str]) -> None:
    """Write a simple index.html that links to each scenario report."""
    lines = [
        "<html><head><title>DCSim: Chaos Engineering Demo</title>",
        "<style>",
        "  body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }",
        "  h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }",
        "  table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "  th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #ddd; }",
        "  th { background: #2c3e50; color: white; }",
        "  tr:hover { background: #f5f5f5; }",
        "  a { color: #2980b9; text-decoration: none; font-weight: 600; }",
        "  a:hover { text-decoration: underline; }",
        "  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }",
        "  .ok { background: #d4edda; color: #155724; }",
        "  .warn { background: #fff3cd; color: #856404; }",
        "</style></head><body>",
        "<h1>DCSim: Datacenter Chaos Engineering</h1>",
        "<p>32 GPUs (4 nodes x 8 GPUs) running 10 AllReduce training iterations. "
        "Baseline iteration = 100ms compute + 50ms comms = 150ms.</p>",
        "<table>",
        "<tr><th>Scenario</th><th>Result</th><th>Steps</th><th>Time</th><th>Impact</th></tr>",
    ]

    baseline_time = results[0].final_time_us if results else 1_500_000
    for r, hp in zip(results, html_paths):
        fname = os.path.basename(hp)
        time_ms = r.final_time_us / 1_000
        overhead = ((r.final_time_us - baseline_time) / baseline_time * 100) if baseline_time else 0
        tag = '<span class="tag ok">clean</span>' if overhead == 0 else f'<span class="tag warn">+{overhead:.0f}%</span>'
        lines.append(
            f'<tr><td><a href="{fname}">{r.name}</a></td>'
            f"<td>{r.workload_state}</td>"
            f"<td>{r.steps_completed}/{r.total_steps}</td>"
            f"<td>{time_ms:.0f} ms</td>"
            f"<td>{tag}</td></tr>"
        )

    lines.append("</table>")
    lines.append("<p><em>Click a scenario name to see the detailed GPU timeline, iteration durations, and event log.</em></p>")
    lines.append("</body></html>")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
