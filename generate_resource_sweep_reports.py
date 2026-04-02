#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SWEEP_ROOT = HERE / "resource_sweep"
SIZE_DIR = SWEEP_ROOT / "size_sweep"
OUT_ROOT = SWEEP_ROOT / "reports"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"
DOC_DIR = OUT_ROOT / "docs"


ALGORITHM_TITLES = {
    "bfs": "BFS",
    "sssp": "SSSP",
    "labelpropagation": "Label Propagation",
    "wcc": "WCC",
}


def ensure_dirs() -> None:
    for path in (FIG_DIR, TABLE_DIR, DOC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = load_payload(path)
    return [row for row in payload.get("results", []) if row.get("status") == "passed"]


def fmt_nodes(nodes: int) -> str:
    if nodes >= 1_000_000:
        value = nodes / 1_000_000
        return f"{value:.1f}M" if value % 1 else f"{int(value)}M"
    if nodes >= 1_000:
        return f"{int(nodes / 1_000)}k"
    return str(nodes)


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/d"
    return f"{value:.2f}"


def resource_label(row: dict[str, Any]) -> str:
    if row["framework"] == "burst":
        return f"p={row['partitions']} g={row['granularity']} mem={row['memory_mb']}MB"
    return f"p={row['partitions']} e={row['executors']} mem={row['executor_memory']}"


def plot_profile_curves(algorithm: str, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['framework']}::{resource_label(row)}"
        grouped[key].append(row)

    plt.figure(figsize=(10, 5.5))
    colors = {"burst": "#1f77b4", "spark": "#d62728"}
    linestyles = {"burst": "-", "spark": "--"}

    for label, points in sorted(grouped.items()):
        points.sort(key=lambda item: int(item["nodes"]))
        framework = points[0]["framework"]
        xs = [int(point["nodes"]) for point in points]
        ys = [float(point["primary_metric_ms"]) for point in points]
        display = f"{framework}: {resource_label(points[0])}"
        plt.plot(
            xs,
            ys,
            marker="o",
            label=display,
            color=colors[framework],
            linestyle=linestyles[framework],
            linewidth=1.8,
        )

    x_values = sorted({int(row["nodes"]) for row in rows})
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.xticks(x_values, [fmt_nodes(node) for node in x_values])
    plt.xlabel("Nodes")
    plt.ylabel("Primary metric (ms)")
    plt.title(f"{ALGORITHM_TITLES.get(algorithm, algorithm)} size sweep by resource profile")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend(fontsize=8, ncol=2)

    out_path = FIG_DIR / f"{algorithm}_profiles.svg"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["primary_metric_ms"]))


def extract_burst_warm(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    warm = row.get("warm_total_ms")
    if warm is not None:
        return float(warm)
    summary = row.get("summary") or {}
    burst = summary.get("burst") or {}
    total = burst.get("total_time_ms")
    return float(total) if total is not None else None


def plot_best_curves(
    algorithm: str,
    burst_rows: list[dict[str, Any]],
    spark_rows: list[dict[str, Any]],
) -> Path | None:
    nodes = sorted({int(row["nodes"]) for row in burst_rows + spark_rows})
    if not nodes:
        return None

    burst_best = {node: best_row([row for row in burst_rows if int(row["nodes"]) == node]) for node in nodes}
    spark_best = {node: best_row([row for row in spark_rows if int(row["nodes"]) == node]) for node in nodes}
    warm_values = [extract_burst_warm(burst_best[node]) for node in nodes]

    plt.figure(figsize=(8.5, 5))
    plt.plot(nodes, [float(burst_best[node]["primary_metric_ms"]) for node in nodes], marker="o", label="Burst cold", color="#1f77b4")
    plt.plot(nodes, [float(spark_best[node]["primary_metric_ms"]) for node in nodes], marker="o", label="Spark total", color="#d62728")
    if any(value is not None for value in warm_values):
        plt.plot(
            nodes,
            [value if value is not None else float("nan") for value in warm_values],
            marker="o",
            label="Burst warm",
            color="#2ca02c",
        )
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.xticks(nodes, [fmt_nodes(node) for node in nodes])
    plt.xlabel("Nodes")
    plt.ylabel("Time (ms)")
    plt.title(f"{ALGORITHM_TITLES.get(algorithm, algorithm)} best configuration per framework")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()

    out_path = FIG_DIR / f"{algorithm}_best.svg"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def build_best_rows(
    burst_rows: list[dict[str, Any]],
    spark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    algorithms = sorted({row["algorithm"] for row in burst_rows + spark_rows})
    for algorithm in algorithms:
        nodes = sorted({int(row["nodes"]) for row in burst_rows + spark_rows if row["algorithm"] == algorithm})
        for node in nodes:
            burst_best = best_row([row for row in burst_rows if row["algorithm"] == algorithm and int(row["nodes"]) == node])
            spark_best = best_row([row for row in spark_rows if row["algorithm"] == algorithm and int(row["nodes"]) == node])
            burst_cold = float(burst_best["primary_metric_ms"]) if burst_best else None
            burst_warm = extract_burst_warm(burst_best)
            spark_total = float(spark_best["primary_metric_ms"]) if spark_best else None
            cold_winner = None
            if burst_cold is not None and spark_total is not None:
                cold_winner = "burst" if burst_cold < spark_total else "spark"
            rows.append(
                {
                    "algorithm": algorithm,
                    "nodes": node,
                    "burst_cold_ms": burst_cold,
                    "burst_warm_ms": burst_warm,
                    "spark_total_ms": spark_total,
                    "burst_profile": resource_label(burst_best) if burst_best else None,
                    "spark_profile": resource_label(spark_best) if spark_best else None,
                    "cold_winner": cold_winner,
                    "spark_over_burst_cold": (spark_total / burst_cold) if burst_cold and spark_total else None,
                }
            )
    return rows


def write_best_table(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    json_path = TABLE_DIR / "best_size_sweep_rows.json"
    json_path.write_text(json.dumps({"results": rows}, indent=2), encoding="utf-8")

    csv_path = TABLE_DIR / "best_size_sweep_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm",
                "nodes",
                "burst_cold_ms",
                "burst_warm_ms",
                "spark_total_ms",
                "burst_profile",
                "spark_profile",
                "cold_winner",
                "spark_over_burst_cold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_summary(
    best_rows: list[dict[str, Any]],
    figure_map: dict[str, dict[str, Path | None]],
    burst_count: int,
    spark_count: int,
) -> Path:
    def doc_rel(path: Path) -> str:
        return (Path("..") / path.relative_to(DOC_DIR.parent)).as_posix()

    lines: list[str] = []
    lines.append("# Resource Sweep Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Burst size-sweep rows: `{burst_count}` passed.")
    lines.append(f"- Spark size-sweep rows: `{spark_count}` passed.")
    lines.append("- Comparison uses the best passed configuration per framework at each input size.")
    lines.append("")

    algorithms = sorted({row["algorithm"] for row in best_rows})
    for algorithm in algorithms:
        title = ALGORITHM_TITLES.get(algorithm, algorithm)
        lines.append(f"## {title}")
        lines.append("")
        best_fig = figure_map.get(algorithm, {}).get("best")
        profile_fig = figure_map.get(algorithm, {}).get("profiles")
        if best_fig:
            lines.append(f"![{title} best]({doc_rel(best_fig)})")
            lines.append("")
        if profile_fig:
            lines.append(f"![{title} profiles]({doc_rel(profile_fig)})")
            lines.append("")
        lines.append("| Nodes | Best Burst cold (ms) | Best Burst warm (ms) | Best Spark total (ms) | Burst profile | Spark profile | Faster cold |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- | --- |")
        for row in [item for item in best_rows if item["algorithm"] == algorithm]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        fmt_nodes(int(row["nodes"])),
                        fmt_ms(row["burst_cold_ms"]),
                        fmt_ms(row["burst_warm_ms"]),
                        fmt_ms(row["spark_total_ms"]),
                        row["burst_profile"] or "n/d",
                        row["spark_profile"] or "n/d",
                        row["cold_winner"] or "n/d",
                    ]
                )
                + " |"
            )
        lines.append("")

    path = DOC_DIR / "summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    burst_rows = load_results(SIZE_DIR / "burst_size_sweep.json")
    spark_rows = load_results(SIZE_DIR / "spark_size_sweep.json")

    figure_map: dict[str, dict[str, Path | None]] = {}
    for algorithm in sorted({row["algorithm"] for row in burst_rows + spark_rows}):
        algorithm_burst = [row for row in burst_rows if row["algorithm"] == algorithm]
        algorithm_spark = [row for row in spark_rows if row["algorithm"] == algorithm]
        figure_map[algorithm] = {
            "profiles": plot_profile_curves(algorithm, algorithm_burst + algorithm_spark),
            "best": plot_best_curves(algorithm, algorithm_burst, algorithm_spark),
        }

    best_rows = build_best_rows(burst_rows, spark_rows)
    json_path, csv_path = write_best_table(best_rows)
    summary_path = write_summary(best_rows, figure_map, len(burst_rows), len(spark_rows))

    index = {
        "summary": str(summary_path),
        "tables": {
            "json": str(json_path),
            "csv": str(csv_path),
        },
        "figures": {
            algorithm: {name: str(path) for name, path in outputs.items() if path is not None}
            for algorithm, outputs in figure_map.items()
        },
    }
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
