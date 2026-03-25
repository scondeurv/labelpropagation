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
    100_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    4_000_000,
    4_500_000,
    5_000_000,
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
BENCHMARK_JSON_PREFIX = "BENCHMARK_RESULT_JSON:"


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def crossing_intervals(results: list[dict], x_key: str) -> list[tuple[dict, dict]]:
    intervals = []
    for idx in range(1, len(results)):
        prev = results[idx - 1]
        curr = results[idx]
        prev_speedup = prev.get("speedup")
        curr_speedup = curr.get("speedup")
        if prev_speedup is None or curr_speedup is None:
            continue
        if (prev_speedup - 1.0) * (curr_speedup - 1.0) <= 0 and prev_speedup != curr_speedup:
            intervals.append((prev, curr))
    return intervals


def estimate_crossover(results: list[dict], x_key: str) -> tuple[float | None, list[tuple[dict, dict]]]:
    intervals = crossing_intervals(results, x_key)
    upward = [(prev, curr) for prev, curr in intervals if prev["speedup"] < 1.0 <= curr["speedup"]]
    if len(upward) != 1:
        return None, intervals
    prev, curr = upward[0]
    m = (curr["speedup"] - prev["speedup"]) / (curr[x_key] - prev[x_key])
    b = prev["speedup"] - m * prev[x_key]
    return (1.0 - b) / m, intervals


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

    benchmark_summary = None
    for line in output.splitlines():
        if line.startswith(BENCHMARK_JSON_PREFIX):
            try:
                benchmark_summary = json.loads(line[len(BENCHMARK_JSON_PREFIX):])
            except json.JSONDecodeError as exc:
                log(f"⚠️  Could not decode benchmark JSON: {exc}")
            break

    if benchmark_summary is None:
        log("⚠️  Benchmark did not emit structured JSON summary")
        return None

    standalone_exec = benchmark_summary.get("standalone", {}).get("execution_time_ms")
    standalone_time = benchmark_summary.get("standalone", {}).get("total_time_ms")
    burst_span = benchmark_summary.get("burst", {}).get("processing_time_ms")
    burst_time = benchmark_summary.get("burst", {}).get("total_time_ms")
    burst_host_total = benchmark_summary.get("burst", {}).get("host_total_time_ms")

    if standalone_exec is not None and standalone_time is not None and burst_time is not None:
        speedup_span = (standalone_exec / burst_span) if burst_span not in (None, 0) else None
        speedup_warm = standalone_time / burst_time if burst_time > 0 else 0.0
        winner = "Burst" if speedup_warm > 1.0 else "Standalone"
        log(f"\n📊 Results for {nodes / 1e6:.1f}M nodes:")
        log(f"   Standalone (load + exec): {standalone_time:.2f} ms")
        log(f"   Burst warm total:         {burst_time:.2f} ms")
        if burst_span is not None:
            log(f"   Burst span:               {burst_span:.2f} ms")
        log(f"   Speedup (warm): {speedup_warm:.2f}x  →  {winner} ✅")
        if speedup_span is not None:
            log(f"   Speedup (span): {speedup_span:.2f}x")
        return {
            "nodes": nodes,
            "standalone_ms": standalone_exec,
            "standalone_exec_ms": standalone_exec,
            "standalone_total_ms": standalone_time,
            "burst_ms": burst_span,
            "burst_span_ms": burst_span,
            "burst_total_ms": burst_host_total,
            "burst_warm_ms": burst_time,
            "speedup": speedup_span,
            "speedup_span": speedup_span,
            "speedup_total": benchmark_summary.get("standalone", {}).get("total_time_ms") / burst_host_total if burst_host_total not in (None, 0) else None,
            "speedup_warm": speedup_warm,
            "winner": winner,
            "validation": benchmark_summary.get("validation", {}),
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
        sa_exec_times = []
        bs_span_times = []
        sa_total_times = []
        bs_warm_times = []
        bs_total_times = []
        validation_state = None

        for run_idx in range(RUNS):
            log(f"\n▶ Run {run_idx + 1}/{RUNS} — {nodes / 1e6:.1f}M nodes")
            result = run_benchmark(nodes, skip_generate=run_idx > 0)
            if result:
                sa_exec_times.append(result["standalone_ms"])
                bs_span_times.append(result["burst_ms"])
                if result.get("standalone_total_ms") is not None:
                    sa_total_times.append(result["standalone_total_ms"])
                if result.get("burst_warm_ms") is not None:
                    bs_warm_times.append(result["burst_warm_ms"])
                if result.get("burst_total_ms") is not None:
                    bs_total_times.append(result["burst_total_ms"])
                validation_state = result.get("validation")
            else:
                log(f"⚠️  Run {run_idx + 1} failed, skipping")
            if run_idx < RUNS - 1:
                time.sleep(3)

        if not sa_exec_times or not sa_total_times or not bs_warm_times:
            log(f"⚠️  All runs failed for {nodes / 1e6:.1f}M, skipping point")
            continue

        sa_exec_mean = statistics.mean(sa_exec_times)
        bs_span_mean = statistics.mean(bs_span_times)
        sa_exec_std = statistics.stdev(sa_exec_times) if len(sa_exec_times) > 1 else 0.0
        bs_span_std = statistics.stdev(bs_span_times) if len(bs_span_times) > 1 else 0.0
        sa_total_mean = statistics.mean(sa_total_times)
        bs_warm_mean = statistics.mean(bs_warm_times)
        sa_total_std = statistics.stdev(sa_total_times) if len(sa_total_times) > 1 else 0.0
        bs_warm_std = statistics.stdev(bs_warm_times) if len(bs_warm_times) > 1 else 0.0
        bs_total_mean = statistics.mean(bs_total_times) if bs_total_times else None
        speedup = sa_exec_mean / bs_span_mean if bs_span_mean > 0 else 0.0
        speedup_warm = sa_total_mean / bs_warm_mean if bs_warm_mean > 0 else 0.0
        speedup_total = None
        if sa_total_mean is not None and bs_total_mean not in (None, 0):
            speedup_total = sa_total_mean / bs_total_mean
        winner = "Burst" if speedup_warm > 1.0 else "Standalone"

        log(f"\n📊 Aggregate {nodes / 1e6:.1f}M ({len(sa_total_times)} runs):")
        log(f"   Standalone total: {sa_total_mean:.1f} ± {sa_total_std:.1f} ms  (runs: {[f'{v:.0f}' for v in sa_total_times]})")
        log(f"   Burst warm total: {bs_warm_mean:.1f} ± {bs_warm_std:.1f} ms  (runs: {[f'{v:.0f}' for v in bs_warm_times]})")
        log(f"   Speedup (warm): {speedup_warm:.2f}x  →  {winner}")
        log(f"   Span secondary: SA exec {sa_exec_mean:.1f} ± {sa_exec_std:.1f} ms vs Burst span {bs_span_mean:.1f} ± {bs_span_std:.1f} ms")

        aggregated.append({
            "nodes": nodes,
            "standalone_ms": round(sa_exec_mean, 2),
            "standalone_exec_ms": round(sa_exec_mean, 2),
            "standalone_std_ms": round(sa_exec_std, 2),
            "standalone_runs_ms": sa_exec_times,
            "burst_ms": round(bs_span_mean, 2),
            "burst_span_ms": round(bs_span_mean, 2),
            "burst_std_ms": round(bs_span_std, 2),
            "burst_runs_ms": bs_span_times,
            "speedup": round(speedup, 4),
            "speedup_span": round(speedup, 4),
            "winner": winner,
            "standalone_total_ms": round(sa_total_mean, 2) if sa_total_mean is not None else None,
            "burst_total_ms": round(bs_total_mean, 2) if bs_total_mean is not None else None,
            "speedup_total": round(speedup_total, 4) if speedup_total is not None else None,
            "burst_warm_ms": round(bs_warm_mean, 2),
            "speedup_warm": round(speedup_warm, 4),
            "validation": validation_state or {},
        })
        time.sleep(5)

    log("\n" + "=" * 80)
    log("LP CROSSOVER VALIDATION SUMMARY (MEAN OVER RUNS)")
    log("=" * 80)
    log(f"{'Nodes':>12} {'SA mean':>12} {'SA std':>9} {'BS mean':>12} {'BS std':>9} {'Speedup':>10} {'Winner':>12}")
    log("-" * 80)

    crossover_point, intervals = estimate_crossover(aggregated, "nodes")
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
    if crossover_point is not None:
        prev, curr = [pair for pair in intervals if pair[0]["speedup"] < 1.0 <= pair[1]["speedup"]][0]
        log("-" * 80)
        log(f"📍 CROSSOVER DETECTED between {prev['nodes'] / 1e6:.1f}M and {curr['nodes'] / 1e6:.1f}M")
        log(f"📍 Refined estimate: {crossover_point / 1e6:.2f}M nodes")
        log("-" * 80)
    elif len(intervals) > 1:
        log("⚠️  Multiple speedup sign changes detected; crossover estimate omitted as ambiguous")

    log("=" * 80)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_points": TEST_POINTS,
        "runs_per_point": RUNS,
        "results": aggregated,
        "crossover_estimate": crossover_point,
        "crossing_intervals": [[prev["nodes"], curr["nodes"]] for prev, curr in intervals],
        "crossover_warning": "multiple_sign_changes" if len(intervals) > 1 else None,
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
