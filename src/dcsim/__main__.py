"""Entry point for ``python -m dcsim``.

Runs all 4 demo scenarios, prints a summary, and optionally generates an HTML report.

Usage:
    python -m dcsim                   # run and print summary
    python -m dcsim --html report.html  # also write Plotly HTML report
    python -m dcsim --scenario baseline # run a single scenario
"""

from __future__ import annotations

import argparse
import sys

from dcsim.demo import (
    ScenarioResult,
    print_summary,
    run_all_scenarios,
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
        metavar="PATH",
        default=None,
        help="Write Plotly HTML report to PATH",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Run a specific scenario or all (default: all)",
    )
    args = parser.parse_args()

    # Run scenarios
    results: list[ScenarioResult]
    if args.scenario == "all":
        results = run_all_scenarios()
    else:
        results = [SCENARIOS[args.scenario]()]

    # Print summary
    print_summary(results)

    # Generate HTML report if requested
    if args.html:
        from dcsim.visualize import generate_report

        # Combine all timelines for the report
        all_entries = []
        for r in results:
            all_entries.extend(r.timeline)

        # Sort by timestamp
        all_entries.sort(key=lambda e: e.timestamp)

        title = "DCSim Simulation Report"
        if args.scenario != "all":
            title = f"DCSim: {results[0].name}"

        output = generate_report(all_entries, output_path=args.html, title=title)
        print(f"HTML report written to: {output}")


if __name__ == "__main__":
    main()
