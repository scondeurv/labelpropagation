#!/usr/bin/env python3
"""
Crossover validation for Label Propagation: Standalone vs Burst.

Runs strategic graph sizes around the expected crossover point and detects where
the burst span crosses the standalone execution time.

Each test point is run RUNS times (graph generated only once per size) to
assess measurement consistency (mean ± std).
"""
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime


TEST_POINTS = [
    3_000_000,
    4_000_000,
    4_500_000,
    5_000_000,
    6_000_000,
]

RUNS = 5

PARTITIONS = 4
MAX_ITER = 10
MEMORY = 4096
DENSITY = 20
WORKER_S3_ENDPOINT = os.environ.get("S3_WORKER_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000")
BUCKET = os.environ.get("S3_BUCKET", "test-bucket")
HOST_S3_ENDPOINT = os.environ.get("S3_HOST_ENDPOINT", "http://localhost:9000")
PYTHON_CMD = os.environ.get("VALIDATION_PYTHON", sys.executable)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def generate_graph(nodes: int) -> bool:
    log(f"Generating {nodes / 1e6:.1f}M node graph...")
    result = subprocess.run(
        [
            PYTHON_CMD, "setup_large_lp_data.py",
            "--nodes", str(nodes),
            "--partitions", str(PARTITIONS),
            "--bucket", BUCKET,
            "--endpoint", HOST_S3_ENDPOINT,
            "--density", str(DENSITY),
            "--prefix", f"graphs/large-{nodes}",
            "--output", f"large_{nodes}.txt",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        log(f"❌ Failed to generate graph: {result.stderr}")
        return False
    log("✅ Graph generated successfully")
    return True


def run_benchmark(nodes: int, skip_generate: bool = False) -> dict | None:
    log(f"\n{'=' * 80}")
    log(f"BENCHMARKING LP: {nodes / 1e6:.1f}M nodes")
    log(f"{'=' * 80}")

    if not skip_generate and not generate_graph(nodes):
        return None

    log("Running benchmark (Standalone + Burst)...")
    result = subprocess.run(
        [
            PYTHON_CMD, "benchmark_lp.py",
            "--nodes", str(nodes),
            "--partitions", str(PARTITIONS),
            "--iter", str(MAX_ITER),
            "--memory", str(MEMORY),
            "--s3-endpoint", WORKER_S3_ENDPOINT,
            "--validation-endpoint", HOST_S3_ENDPOINT,
            "--bucket", BUCKET,
            "--key-prefix", "graphs",
            "--validate",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        log(f"❌ Benchmark failed: {result.stderr}")
        return None

    output = result.stdout
    print(output)

    standalone_time = None
    burst_time = None
    for line in output.split("\n"):
        if "Standalone Processing Time (Execution):" in line:
            try:
                standalone_time = float(line.split(":")[1].strip().split()[0])
            except Exception:
                pass
        if "Burst Processing Time (Distributed Span):" in line:
            try:
                burst_time = float(line.split(":")[1].strip().split()[0])
            except Exception:
                pass

    if standalone_time is not None and burst_time is not None:
        speedup = standalone_time / burst_time if burst_time > 0 else 0.0
        winner = "Burst" if speedup > 1.0 else "Standalone"
        log(f"\n📊 Results for {nodes / 1e6:.1f}M nodes:")
        log(f"   Standalone: {standalone_time:.2f} ms")
        log(f"   Burst span: {burst_time:.2f} ms")
        log(f"   Speedup:    {speedup:.2f}x  →  {winner} ✅")
        return {
            "nodes": nodes,
            "standalone_ms": standalone_time,
            "burst_ms": burst_time,
            "speedup": speedup,
            "winner": winner,
        }

    log("⚠️  Could not parse results")
    return None


def main() -> None:
    log("=" * 80)
    log("LABEL PROPAGATION CROSSOVER VALIDATION BENCHMARK")
    log("=" * 80)
    log(f"Test points: {[f'{n / 1e6:.1f}M' for n in TEST_POINTS]}")
    log(f"Runs per point: {RUNS}")
    log(f"Config: partitions={PARTITIONS}, max_iter={MAX_ITER}, density={DENSITY}, memory={MEMORY}MB")
    log(f"Python runner: {PYTHON_CMD}")
    log("=" * 80)

    aggregated = []

    for nodes in TEST_POINTS:
        sa_times = []
        bs_times = []

        for run_idx in range(RUNS):
            log(f"\n▶ Run {run_idx + 1}/{RUNS} — {nodes / 1e6:.1f}M nodes")
            result = run_benchmark(nodes, skip_generate=run_idx > 0)
            if result:
                sa_times.append(result["standalone_ms"])
                bs_times.append(result["burst_ms"])
            else:
                log(f"⚠️  Run {run_idx + 1} failed, skipping")
            if run_idx < RUNS - 1:
                time.sleep(3)

        if not sa_times:
            log(f"⚠️  All runs failed for {nodes / 1e6:.1f}M, skipping point")
            continue

        sa_mean = statistics.mean(sa_times)
        bs_mean = statistics.mean(bs_times)
        sa_std = statistics.stdev(sa_times) if len(sa_times) > 1 else 0.0
        bs_std = statistics.stdev(bs_times) if len(bs_times) > 1 else 0.0
        speedup = sa_mean / bs_mean if bs_mean > 0 else 0.0
        winner = "Burst" if speedup > 1.0 else "Standalone"

        log(f"\n📊 Aggregate {nodes / 1e6:.1f}M ({len(sa_times)} runs):")
        log(f"   Standalone: {sa_mean:.1f} ± {sa_std:.1f} ms  (runs: {[f'{v:.0f}' for v in sa_times]})")
        log(f"   Burst span: {bs_mean:.1f} ± {bs_std:.1f} ms  (runs: {[f'{v:.0f}' for v in bs_times]})")
        log(f"   Speedup:    {speedup:.2f}x  →  {winner}")

        aggregated.append({
            "nodes": nodes,
            "standalone_ms": round(sa_mean, 2),
            "standalone_std_ms": round(sa_std, 2),
            "standalone_runs_ms": sa_times,
            "burst_ms": round(bs_mean, 2),
            "burst_std_ms": round(bs_std, 2),
            "burst_runs_ms": bs_times,
            "speedup": round(speedup, 4),
            "winner": winner,
        })
        time.sleep(5)

    log("\n" + "=" * 80)
    log("LP CROSSOVER VALIDATION SUMMARY (MEAN OVER RUNS)")
    log("=" * 80)
    log(f"{'Nodes':>12} {'SA mean':>12} {'SA std':>9} {'BS mean':>12} {'BS std':>9} {'Speedup':>10} {'Winner':>12}")
    log("-" * 80)

    crossover_found = False
    crossover_point = None
    for i, result in enumerate(aggregated):
        log(
            f"{result['nodes'] / 1e6:>10.1f}M "
            f"{result['standalone_ms']:>10.1f}ms "
            f"{result['standalone_std_ms']:>7.1f}ms "
            f"{result['burst_ms']:>10.1f}ms "
            f"{result['burst_std_ms']:>7.1f}ms "
            f"{result['speedup']:>9.2f}x "
            f"{result['winner']:>12}"
        )
        if i > 0 and not crossover_found:
            prev = aggregated[i - 1]
            if prev["speedup"] < 1.0 and result["speedup"] >= 1.0:
                crossover_found = True
                m = (result["speedup"] - prev["speedup"]) / (result["nodes"] - prev["nodes"])
                b = prev["speedup"] - m * prev["nodes"]
                crossover_point = (1.0 - b) / m
                log("-" * 80)
                log(f"📍 CROSSOVER DETECTED between {prev['nodes'] / 1e6:.1f}M and {result['nodes'] / 1e6:.1f}M")
                log(f"📍 Refined estimate: {crossover_point / 1e6:.2f}M nodes")
                log("-" * 80)

    log("=" * 80)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_points": TEST_POINTS,
        "runs_per_point": RUNS,
        "results": aggregated,
        "crossover_estimate": crossover_point,
        "configuration": {
            "partitions": PARTITIONS,
            "max_iter": MAX_ITER,
            "memory_mb": MEMORY,
            "density": DENSITY,
            "python_cmd": PYTHON_CMD,
        },
    }

    with open("crossover_validation_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    log("💾 Results saved: crossover_validation_results.json")
    log("✅ CROSSOVER VALIDATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as exc:
        log(f"\n❌ Error: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
