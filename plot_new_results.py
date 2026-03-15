#!/usr/bin/env python3
"""
Generate comprehensive Label Propagation benchmark analysis plots.

Reads data from crossover_validation_results.json.

Outputs:
  lp_comprehensive_analysis.png   – 3×2 multi-panel figure
  lp_crossover_analysis.png       – dedicated crossover figure
"""
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib / numpy not available. Install with: pip install matplotlib numpy")
        sys.exit(1)


def load_results(json_path: str = "crossover_validation_results.json") -> tuple[list, dict, float | None]:
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        print("Run the crossover validation first to generate the data.")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("Error: no results in JSON file.")
        sys.exit(1)

    config = data.get("configuration", {})
    crossover_estimate = data.get("crossover_estimate")
    return results, config, crossover_estimate


def _fmt_node_count(n):
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K" if n >= 1000 else str(n)


def plot_comprehensive(results: list, crossover_estimate: float | None, config: dict) -> None:
    nodes = np.array([r["nodes"] for r in results])
    sa_exec = np.array([r["standalone_ms"] for r in results], dtype=float)
    bs_span = np.array([r["burst_ms"] for r in results], dtype=float)

    has_burst_total = all(r.get("burst_total_ms") is not None for r in results)
    has_sa_total = all(r.get("standalone_total_ms") is not None for r in results)
    has_total_speedup = has_burst_total and has_sa_total

    bs_total = np.array([r["burst_total_ms"] for r in results], dtype=float) if has_burst_total else None
    sa_total = np.array([r["standalone_total_ms"] for r in results], dtype=float) if has_sa_total else None

    sa_std = np.array([r.get("standalone_std_ms", 0.0) for r in results], dtype=float)
    bs_std = np.array([r.get("burst_std_ms", 0.0) for r in results], dtype=float)

    speedup_algo = sa_exec / bs_span
    speedup_total = (sa_total / bs_total) if has_total_speedup else None
    overhead_ms = (bs_total - bs_span) if has_burst_total else None
    overhead_pct = (overhead_ms / bs_total * 100) if has_burst_total else None

    nodes_m = nodes / 1e6
    sa_throughput = nodes / sa_exec * 1000
    bs_throughput = nodes / bs_span * 1000

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Label Propagation Benchmark: Standalone vs Burst\n"
        f"Config: {config.get('partitions', '?')} partitions, "
        f"max_iter={config.get('max_iter', '?')}, "
        f"memory={config.get('memory_mb', '?')}MB",
        fontsize=13,
        fontweight="bold",
    )

    colours = {"standalone": "#2196F3", "burst": "#F44336", "overhead": "#FF9800", "speedup": "#9C27B0"}

    ax = axes[0, 0]
    sa_yerr = sa_std / 1000 if sa_std.any() else None
    bs_yerr = bs_std / 1000 if bs_std.any() else None
    ax.errorbar(nodes_m, sa_exec / 1000, yerr=sa_yerr, fmt="o-", color=colours["standalone"], lw=2, capsize=4, label="Standalone execution")
    ax.errorbar(nodes_m, bs_span / 1000, yerr=bs_yerr, fmt="s-", color=colours["burst"], lw=2, capsize=4, label="Burst distributed span")
    if crossover_estimate:
        ax.axvline(crossover_estimate / 1e6, color="green", ls=":", lw=1.5, label=f"Crossover ~{crossover_estimate/1e6:.1f}M")
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Processing Time (Algorithmic)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(nodes_m, speedup_algo, "D-", color=colours["speedup"], lw=2, label="Algorithmic speedup")
    if speedup_total is not None:
        ax.plot(nodes_m, speedup_total, "^--", color="gray", lw=1.5, label="Total speedup")
    ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.6)
    if crossover_estimate:
        ax.axvline(crossover_estimate / 1e6, color="green", ls=":", lw=1.5)
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Speedup (Standalone / Burst)")
    ax.set_title("Speedup")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if overhead_pct is not None:
        bars = ax.bar(nodes_m, overhead_pct, color=colours["overhead"], alpha=0.85, width=float(nodes_m[0]) * 0.35 if len(nodes_m) > 0 else 0.1)
        for bar, pct, ov in zip(bars, overhead_pct, overhead_ms):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{ov/1000:.0f}s\n{pct:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xlabel("Graph size (M nodes)")
        ax.set_ylabel("Infrastructure overhead (%)")
        ax.set_title("OpenWhisk Overhead")
        ax.set_xticks(nodes_m)
        ax.set_xticklabels([_fmt_node_count(n) for n in nodes], fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No measured burst total metrics\n(overhead panel hidden)", ha="center", va="center", fontsize=10)
        ax.set_title("OpenWhisk Overhead")

    ax = axes[1, 0]
    ax.plot(nodes_m, sa_throughput / 1e6, "o-", color=colours["standalone"], lw=2, label="Standalone (M nodes/s)")
    ax.plot(nodes_m, bs_throughput / 1e6, "s-", color=colours["burst"], lw=2, label="Burst span (M nodes/s)")
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Throughput (M nodes/s)")
    ax.set_title("Throughput")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(nodes_m, sa_exec / 1000, color=colours["standalone"], s=60, zorder=5)
    if HAS_SCIPY and len(nodes) >= 2:
        slope, intercept, r2, *_ = sp_stats.linregress(nodes, sa_exec)
        x_fit = np.linspace(nodes[0], nodes[-1] * 1.15, 100)
        y_fit = (slope * x_fit + intercept) / 1000
        ax.plot(x_fit / 1e6, y_fit, "--", color=colours["standalone"], alpha=0.7,
                label=f"Linear fit (R²={r2**2:.4f})\n{slope/1000*1e6:.2f} s/M nodes")
        ax.legend(fontsize=9)
    elif len(nodes) >= 2:
        m = (sa_exec[-1] - sa_exec[0]) / (nodes[-1] - nodes[0])
        b = sa_exec[0] - m * nodes[0]
        x_fit = np.linspace(nodes[0], nodes[-1] * 1.15, 100)
        ax.plot(x_fit / 1e6, (m * x_fit + b) / 1000, "--", color=colours["standalone"], alpha=0.7)
    ax.set_xlabel("Graph size (M nodes)")
    ax.set_ylabel("Standalone execution time (s)")
    ax.set_title("Standalone Scaling (Linear)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    headers = ["Nodes", "SA exec (s)", "Burst span (s)", "Speedup", "Winner"]
    for r in results:
        sa = r["standalone_ms"] / 1000
        bs = r["burst_ms"] / 1000
        sp = sa / bs if bs > 0 else 0
        table_data.append([
            _fmt_node_count(r["nodes"]),
            f"{sa:.2f}",
            f"{bs:.2f}",
            f"{sp:.2f}x",
            "Burst ✓" if sp > 1.0 else "Standalone",
        ])
    if crossover_estimate:
        table_data.append([f"~{crossover_estimate/1e6:.2f}M", "≈", "≈", "1.00x", "CROSSOVER"])

    tbl = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.6)
    ax.set_title("Results Summary", fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "lp_comprehensive_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out}")
    plt.close()


def plot_crossover(results: list, crossover_estimate: float | None) -> None:
    nodes = np.array([r["nodes"] for r in results])
    sa_exec = np.array([r["standalone_ms"] for r in results], dtype=float)
    bs_span = np.array([r["burst_ms"] for r in results], dtype=float)
    sa_std = np.array([r.get("standalone_std_ms", 0.0) for r in results], dtype=float)
    bs_std = np.array([r.get("burst_std_ms", 0.0) for r in results], dtype=float)
    speedup = sa_exec / bs_span
    nodes_m = nodes / 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Label Propagation Crossover Point Analysis", fontsize=13, fontweight="bold")

    sa_yerr = sa_std / 1000 if sa_std.any() else None
    bs_yerr = bs_std / 1000 if bs_std.any() else None
    ax1.errorbar(nodes_m, sa_exec / 1000, yerr=sa_yerr, fmt="o-", color="#2196F3", lw=2, ms=8, capsize=5, label="Standalone execution")
    ax1.errorbar(nodes_m, bs_span / 1000, yerr=bs_yerr, fmt="s-", color="#F44336", lw=2, ms=8, capsize=5, label="Burst distributed span")
    if crossover_estimate:
        cx = crossover_estimate / 1e6
        ax1.axvline(cx, color="green", ls="-.", lw=2, label=f"Crossover: {cx:.2f}M nodes")
        ax1.axvspan(0, cx, alpha=0.04, color="blue")
        ax1.axvspan(cx, nodes_m[-1] * 1.05, alpha=0.04, color="red")
    ax1.set_xlabel("Graph size (M nodes)")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Execution Time vs Graph Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(nodes_m, speedup, "D-", color="#9C27B0", lw=2, ms=8, label="Algorithmic speedup")
    ax2.axhline(1.0, color="black", ls="--", alpha=0.7, label="Speedup = 1.0 (crossover)")
    if crossover_estimate:
        cx = crossover_estimate / 1e6
        ax2.axvline(cx, color="green", ls="-.", lw=2)
        ax2.axvspan(0, cx, alpha=0.06, color="blue", label="Standalone region")
        ax2.axvspan(cx, nodes_m[-1] * 1.05, alpha=0.06, color="red", label="Burst region")
    ax2.set_xlabel("Graph size (M nodes)")
    ax2.set_ylabel("Speedup (Standalone exec / Burst span)")
    ax2.set_title("Algorithmic Speedup Trend")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "lp_crossover_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out}")
    plt.close()


def main():
    json_path = "crossover_validation_results.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    results, config, crossover_estimate = load_results(json_path)

    print(f"Loaded {len(results)} data points from {json_path}")
    if crossover_estimate:
        print(f"Crossover estimate: {crossover_estimate / 1e6:.2f}M nodes")

    plot_comprehensive(results, crossover_estimate, config)
    plot_crossover(results, crossover_estimate)
    print("✅ All plots generated.")


if __name__ == "__main__":
    main()
