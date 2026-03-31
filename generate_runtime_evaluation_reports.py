#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "runtime_evaluation" / "results"
FIG_ROOT = HERE / "runtime_evaluation" / "figures"
TABLE_ROOT = HERE / "runtime_evaluation" / "tables"


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def generate_startup_plot(rows: list[dict]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    grouped: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        config = row["configuration"]
        metrics = row["metrics"]
        grouped.setdefault(config["workers"], []).append((config["granularity"], metrics["startup_median_ms"]))
    for workers, points in sorted(grouped.items()):
        points.sort()
        plt.plot([granularity for granularity, _ in points], [lat for _, lat in points], marker="o", label=f"{workers} workers")
    plt.xlabel("Granularity")
    plt.ylabel("Median startup latency (ms)")
    plt.title("Startup latency vs granularity")
    plt.legend()
    save_figure(FIG_ROOT / "startup_latency.svg")


def generate_load_plot(rows: list[dict]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    grouped: dict[int, list[tuple[int, float]]] = {}
    for row in rows:
        config = row["configuration"]
        metrics = row["metrics"]
        load_ms = metrics.get("load_ms")
        if load_ms is None:
            continue
        grouped.setdefault(config["workers"], []).append((config["granularity"], load_ms))
    for workers, points in sorted(grouped.items()):
        points.sort()
        plt.plot(
            [granularity for granularity, _ in points],
            [lat for _, lat in points],
            marker="o",
            label=f"{workers} workers",
        )
    plt.xlabel("Granularity")
    plt.ylabel("Load time (ms)")
    plt.title("Partition load time vs granularity")
    plt.legend()
    save_figure(FIG_ROOT / "load_latency.svg")


def generate_collective_plot(rows: list[dict]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        mode = row["probe"]
        config = row["configuration"]
        latency = row["metrics"]["latency_median_ms"]
        if latency is None:
            continue
        key = f"{mode}:{config['workers']}"
        grouped.setdefault(key, []).append((config["granularity"], latency))
    for label, points in sorted(grouped.items()):
        points.sort()
        plt.plot([granularity for granularity, _ in points], [lat for _, lat in points], marker="o", label=label)
    plt.xlabel("Granularity")
    plt.ylabel("Median latency (ms)")
    plt.title("Burst collectives vs granularity")
    plt.legend()
    save_figure(FIG_ROOT / "collectives_latency.svg")


def generate_collective_mode_plots(rows: list[dict]) -> None:
    if not rows:
        return
    for mode in ("broadcast", "all_to_all"):
        mode_rows = [row for row in rows if row.get("probe") == mode]
        if not mode_rows:
            continue
        plt.figure(figsize=(8, 5))
        grouped: dict[int, list[tuple[int, float]]] = {}
        for row in mode_rows:
            config = row["configuration"]
            latency = row["metrics"]["latency_median_ms"]
            if latency is None:
                continue
            grouped.setdefault(config["workers"], []).append((config["granularity"], latency))
        for workers, points in sorted(grouped.items()):
            points.sort()
            plt.plot(
                [granularity for granularity, _ in points],
                [lat for _, lat in points],
                marker="o",
                label=f"{workers} workers",
            )
        plt.xlabel("Granularity")
        plt.ylabel("Median latency (ms)")
        plt.title(f"{mode} latency vs granularity")
        plt.legend()
        save_figure(FIG_ROOT / f"{mode}_latency.svg")


def generate_startup_timeline(rows: list[dict]) -> None:
    if not rows:
        return
    target_workers = max(row["configuration"]["workers"] for row in rows)
    timeline_rows = [
        row for row in rows
        if row["configuration"]["workers"] == target_workers
        and row["configuration"]["granularity"] in {1, 2, 4}
    ]
    if not timeline_rows:
        return

    timeline_rows.sort(key=lambda row: row["configuration"]["granularity"])
    fig, axes = plt.subplots(len(timeline_rows), 1, figsize=(10, 2.2 * len(timeline_rows)), sharex=True)
    if len(timeline_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, timeline_rows):
        config = row["configuration"]
        starts = []
        for worker in row.get("workers", []):
            worker_start = None
            for ts in worker.get("timestamps", []):
                if ts.get("key") == "worker_start":
                    worker_start = int(ts["value"])
                    break
            if worker_start is not None:
                starts.append((worker.get("worker_id", len(starts)), worker_start))
        if not starts:
            continue
        base = min(ts for _, ts in starts)
        starts.sort(key=lambda item: item[1])
        offsets = [ts - base for _, ts in starts]
        labels = [wid for wid, _ in starts]
        ax.barh(range(len(offsets)), offsets, color="#4C78A8")
        ax.set_yticks(range(len(offsets)))
        ax.set_yticklabels(labels)
        ax.set_ylabel(f"g={config['granularity']}")
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("Worker start offset (ms)")
    fig.suptitle(f"Startup simultaneity timeline ({target_workers} workers)")
    save_figure(FIG_ROOT / "startup_timeline.svg")


def generate_ptp_plot(rows: list[dict]) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        config = row["configuration"]
        throughput = row["metrics"]["throughput_mb_s"]
        if throughput is None:
            continue
        grouped.setdefault(config["backend"], []).append((config["chunk_size"], throughput))
    for backend, points in sorted(grouped.items()):
        points.sort()
        plt.plot([chunk for chunk, _ in points], [throughput for _, throughput in points], marker="o", label=backend)
    plt.xlabel("Chunk size (KB)")
    plt.ylabel("Throughput (MiB/s)")
    plt.title("Point-to-point throughput vs chunk size")
    plt.legend()
    save_figure(FIG_ROOT / "ptp_throughput.svg")


def generate_ptp_burst_size_plot(rows: list[dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: row["configuration"]["workers"])
    workers = []
    throughputs = []
    for row in rows:
        throughput = row.get("metrics", {}).get("throughput_mb_s")
        if throughput is None:
            continue
        workers.append(row["configuration"]["workers"])
        throughputs.append(throughput)
    if not workers:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(workers, throughputs, marker="o")
    plt.xlabel("Workers")
    plt.ylabel("Aggregate throughput (MiB/s)")
    plt.title("Point-to-point throughput vs burst size")
    save_figure(FIG_ROOT / "ptp_throughput_vs_burst_size.svg")


def generate_application_breakdown(rows: list[dict]) -> None:
    if not rows:
        return
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    for algorithm, points in grouped.items():
        points.sort(key=lambda item: item["configuration"]["granularity"])
        granularities = [row["configuration"]["granularity"] for row in points]
        loads = []
        computes = []
        comms = []
        for row in points:
            phase_metrics = row.get("burst", {}).get("phase_metrics") or {}
            loads.append(phase_metrics.get("load_ms") or 0)
            computes.append(phase_metrics.get("compute_ms") or 0)
            comms.append(phase_metrics.get("communication_ms") or 0)

        plt.figure(figsize=(8, 5))
        plt.stackplot(granularities, loads, computes, comms, labels=["load", "compute", "communication"])
        plt.xlabel("Granularity")
        plt.ylabel("Time (ms)")
        plt.title(f"{algorithm} phase breakdown vs granularity")
        plt.legend(loc="upper left")
        save_figure(FIG_ROOT / f"{algorithm}_phase_breakdown.svg")


def generate_load_component_plots(rows: list[dict]) -> None:
    if not rows:
        return
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)

    for algorithm, points in grouped.items():
        points.sort(key=lambda item: item["configuration"]["granularity"])
        granularities = [row["configuration"]["granularity"] for row in points]
        loads = []
        for row in points:
            phase_metrics = row.get("burst", {}).get("phase_metrics") or {}
            loads.append(phase_metrics.get("load_ms") or 0)
        plt.figure(figsize=(8, 5))
        plt.plot(granularities, loads, marker="o")
        plt.xlabel("Granularity")
        plt.ylabel("Load time (ms)")
        plt.title(f"{algorithm} load time vs granularity")
        save_figure(FIG_ROOT / f"{algorithm}_load_vs_granularity.svg")


def generate_traffic_table(rows: list[dict]) -> None:
    if not rows:
        return
    table_rows = []
    for row in rows:
        traffic = row.get("burst", {}).get("logical_traffic_bytes")
        if not traffic:
            continue
        table_rows.append(
            {
                "algorithm": row["algorithm"],
                "granularity": row["configuration"]["granularity"],
                "partitions": row["configuration"]["partitions"],
                "backend": row["configuration"].get("backend"),
                "chunk_size": row["configuration"].get("chunk_size"),
                "reduce_bytes": traffic.get("reduce_bytes"),
                "broadcast_bytes": traffic.get("broadcast_bytes"),
                "total_bytes": traffic.get("total_bytes"),
            }
        )
    if table_rows:
        (TABLE_ROOT / "logical_traffic.json").write_text(
            json.dumps({"results": table_rows}, indent=2),
            encoding="utf-8",
        )


def generate_system_comparison_plots() -> None:
    raw_root = HERE / "benchmark_reports" / "raw"
    spark_root = HERE / "spark_baseline" / "results"
    for algorithm in ("bfs", "sssp", "wcc"):
        raw_path = raw_root / f"{algorithm}.json"
        spark_path = spark_root / f"{algorithm}.json"
        if not raw_path.exists() or not spark_path.exists():
            continue

        raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
        spark_payload = json.loads(spark_path.read_text(encoding="utf-8"))
        raw_rows = {row["nodes"]: row for row in raw_payload.get("results", [])}
        spark_rows = {row["nodes"]: row for row in spark_payload.get("results", [])}
        nodes = sorted(set(raw_rows) & set(spark_rows))
        if not nodes:
            continue

        standalone = [raw_rows[node].get("standalone_total_ms") for node in nodes]
        burst = [raw_rows[node].get("burst_total_ms") for node in nodes]
        spark = [spark_rows[node].get("spark_total_ms") for node in nodes]

        plt.figure(figsize=(8, 5))
        plt.plot(nodes, standalone, marker="o", label="Standalone")
        plt.plot(nodes, burst, marker="o", label="Burst")
        plt.plot(nodes, spark, marker="o", label="Spark")
        plt.xlabel("Nodes")
        plt.ylabel("End-to-end time (ms)")
        plt.title(f"{algorithm} end-to-end comparison")
        plt.legend()
        save_figure(FIG_ROOT / f"{algorithm}_end_to_end_comparison.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ATC25-style plots from characterization/application outputs.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    parse_args()
    ensure_dirs()
    startup_rows = load_results(RESULTS_ROOT / "characterization" / "startup.json")
    load_rows = load_results(RESULTS_ROOT / "characterization" / "load.json")
    collective_rows = load_results(RESULTS_ROOT / "characterization" / "collectives.json")
    ptp_rows = load_results(RESULTS_ROOT / "characterization" / "ptp.json")
    ptp_burst_size_rows = load_results(RESULTS_ROOT / "characterization" / "ptp_burst_size.json")
    generate_startup_plot(startup_rows)
    generate_load_plot(load_rows)
    generate_startup_timeline(startup_rows)
    generate_collective_plot(collective_rows)
    generate_collective_mode_plots(collective_rows)
    generate_ptp_plot(ptp_rows)
    generate_ptp_burst_size_plot(ptp_burst_size_rows)
    app_rows = load_results(RESULTS_ROOT / "applications" / "phase_breakdown.json")
    generate_application_breakdown(app_rows)
    generate_load_component_plots(app_rows)
    generate_traffic_table(app_rows)
    generate_system_comparison_plots()


if __name__ == "__main__":
    main()
