#!/usr/bin/env python3
"""
Quick Label Propagation crossover analysis.

Uses existing validation results when available, otherwise falls back to
MEASURED_POINTS.

Saves:
  crossover_data.json
  crossover_analysis.png
"""
import json
import os
import sys


# Format: (nodes, standalone_exec_ms, burst_span_ms, burst_total_ms|None, standalone_total_ms|None)
MEASURED_POINTS = [
    (1_000_000, 2900.0, 6740.0, 15610.0, None),
    (10_000_000, 35180.0, 5410.0, 14910.0, None),
]


def interpolate_crossover(
    p1: tuple[int, float, float],
    p2: tuple[int, float, float],
) -> float | None:
    n1, sa1, bs1 = p1
    n2, sa2, bs2 = p2

    m_sa = (sa2 - sa1) / (n2 - n1)
    b_sa = sa1 - m_sa * n1
    m_bs = (bs2 - bs1) / (n2 - n1)
    b_bs = bs1 - m_bs * n1

    denom = m_sa - m_bs
    if abs(denom) < 1e-10:
        return None

    return (b_bs - b_sa) / denom


def plot_crossover_analysis(points: list, crossover_n: float | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot generation")
        return

    nodes_arr = np.array([p[0] for p in points]) / 1e6
    sa_exec = np.array([p[1] for p in points])
    bs_span = np.array([p[2] for p in points])
    has_totals = all(p[3] is not None and p[4] is not None for p in points)
    bs_total = np.array([p[3] for p in points], dtype=float) if has_totals else None
    sa_total = np.array([p[4] for p in points], dtype=float) if has_totals else None

    n_interp = np.linspace(nodes_arr[0], nodes_arr[-1] * 1.2, 200)
    n_raw = n_interp * 1e6

    m_sa, b_sa = np.polyfit([p[0] for p in points], sa_exec, 1)
    m_bs, b_bs = np.polyfit([p[0] for p in points], bs_span, 1)
    sa_interp = m_sa * n_raw + b_sa
    bs_interp = m_bs * n_raw + b_bs

    speedup_algo = sa_exec / bs_span
    speedup_interp = sa_interp / np.maximum(bs_interp, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Label Propagation Crossover Analysis: Standalone vs Burst", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(nodes_arr, sa_exec / 1000, "o-", color="steelblue", label="Standalone (exec)")
    ax.plot(nodes_arr, bs_span / 1000, "s-", color="tomato", label="Burst span")
    ax.plot(n_interp, sa_interp / 1000, "--", color="steelblue", alpha=0.5, label="Standalone (fit)")
    ax.plot(n_interp, bs_interp / 1000, "--", color="tomato", alpha=0.5, label="Burst span (fit)")
    if crossover_n is not None:
        ax.axvline(crossover_n / 1e6, color="green", linestyle=":", alpha=0.8,
                   label=f"Crossover ~{crossover_n/1e6:.1f}M")
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Execution Times")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(nodes_arr, speedup_algo, "D-", color="purple", label="Algorithmic speedup")
    ax.plot(n_interp, speedup_interp, "--", color="purple", alpha=0.5, label="Speedup (fit)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.8, label="Speedup = 1 (crossover)")
    if crossover_n is not None:
        ax.axvline(crossover_n / 1e6, color="green", linestyle=":", alpha=0.8)
        ax.fill_betweenx([0, max(speedup_algo.max(), 2)], 0, crossover_n / 1e6,
                         color="blue", alpha=0.05, label="Standalone faster region")
        ax.fill_betweenx([0, max(speedup_algo.max(), 2)], crossover_n / 1e6, n_interp[-1],
                         color="red", alpha=0.05, label="Burst faster region")
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Speedup (standalone_exec / burst_span)")
    ax.set_title("Algorithmic Speedup")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if has_totals:
        ax.bar(nodes_arr - 0.1, sa_total / 1000, width=0.18, color="steelblue", label="Standalone (total)")
        ax.bar(nodes_arr + 0.1, bs_total / 1000, width=0.18, color="tomato", label="Burst (total)")
        ax.set_xlabel("Graph size (M nodes)")
        ax.set_ylabel("Total time (s)")
        ax.set_title("Total End-to-End Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No measured total metrics\n(panel hidden)", ha="center", va="center", fontsize=10)
        ax.set_title("Total End-to-End Time")

    ax = axes[1, 1]
    if has_totals:
        overhead_ms = bs_total - bs_span
        overhead_pct = overhead_ms / bs_total * 100
        bars = ax.bar(nodes_arr, overhead_pct, color="orange", alpha=0.8)
        for bar, pct, ov in zip(bars, overhead_pct, overhead_ms):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{ov/1000:.1f}s", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Graph size (M nodes)")
        ax.set_ylabel("Overhead (%)")
        ax.set_title("OpenWhisk Infrastructure Overhead")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No measured burst total metrics\n(panel hidden)", ha="center", va="center", fontsize=10)
        ax.set_title("OpenWhisk Infrastructure Overhead")

    plt.tight_layout()
    plt.savefig("crossover_analysis.png", dpi=150, bbox_inches="tight")
    print("📊 Plot saved: crossover_analysis.png")


def generate_report(points: list, crossover_n: float | None) -> None:
    print("\n" + "=" * 70)
    print("LABEL PROPAGATION CROSSOVER ANALYSIS")
    print("=" * 70)
    print(f"{'Nodes':>12} {'SA exec':>10} {'Burst span':>12} {'Speedup':>10}")
    print("-" * 70)
    for nodes, sa_exec, bs_span, *_ in points:
        speedup = sa_exec / bs_span if bs_span > 0 else float("inf")
        winner = "Burst ✓" if speedup > 1.0 else "Standalone"
        print(f"{nodes / 1e6:>10.1f}M {sa_exec:>10.0f}ms {bs_span:>10.0f}ms  {speedup:>8.2f}x  {winner}")
    print("=" * 70)
    if crossover_n is not None:
        print(f"\n📍 Estimated crossover: {crossover_n / 1e6:.2f}M nodes")
    else:
        print("\n⚠️  No crossover detected in the measured range")


def main() -> None:
    if os.path.exists("crossover_validation_results.json"):
        with open("crossover_validation_results.json") as f:
            data = json.load(f)
        results = data.get("results", [])
        if len(results) >= 2:
            points = [
                (
                    r["nodes"],
                    r["standalone_ms"],
                    r["burst_ms"],
                    r.get("burst_total_ms"),
                    r.get("standalone_total_ms"),
                )
                for r in results
            ]
            print("Loaded data from crossover_validation_results.json")
        else:
            points = MEASURED_POINTS
            print("Using hardcoded MEASURED_POINTS (not enough data in JSON)")
    else:
        points = MEASURED_POINTS
        print("Using hardcoded MEASURED_POINTS (no JSON file found)")

    if len(points) < 2:
        print("Need at least 2 data points. Edit MEASURED_POINTS and re-run.")
        sys.exit(1)

    crossover_n = None
    for i in range(len(points) - 1):
        p1 = (points[i][0], points[i][1], points[i][2])
        p2 = (points[i + 1][0], points[i + 1][1], points[i + 1][2])
        cx = interpolate_crossover(p1, p2)
        if cx is not None and p1[0] <= cx <= p2[0] * 1.5:
            crossover_n = cx
            break

    if crossover_n is None:
        p_first = (points[0][0], points[0][1], points[0][2])
        p_last = (points[-1][0], points[-1][1], points[-1][2])
        crossover_n = interpolate_crossover(p_first, p_last)

    generate_report(points, crossover_n)
    plot_crossover_analysis(points, crossover_n)

    output = {
        "data_points": [
            {
                "nodes": p[0],
                "standalone_exec_ms": p[1],
                "burst_span_ms": p[2],
                "burst_total_ms": p[3],
                "standalone_total_ms": p[4],
                "speedup": p[1] / p[2] if p[2] > 0 else None,
            }
            for p in points
        ],
        "crossover_estimate_nodes": crossover_n,
        "crossover_estimate_millions": crossover_n / 1e6 if crossover_n else None,
    }

    with open("crossover_data.json", "w") as f:
        json.dump(output, f, indent=2)
    print("💾 Saved: crossover_data.json")


if __name__ == "__main__":
    main()
