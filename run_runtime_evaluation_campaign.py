#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/sergio/src")
HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "runtime_evaluation" / "results"
CHAR_DIR = RESULTS_ROOT / "characterization"
APP_DIR = RESULTS_ROOT / "applications"
BENCHMARK_JSON_PREFIX = "BENCHMARK_RESULT_JSON:"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def repo_python(repo_name: str) -> Path:
    candidates = [
        ROOT / repo_name / ".venv/bin/python",
        ROOT / "labelpropagation/.venv/bin/python",
        ROOT / "bfs/.venv/bin/python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def ensure_dirs() -> None:
    for directory in (CHAR_DIR, APP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def parse_summary(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(BENCHMARK_JSON_PREFIX):
            return json.loads(line[len(BENCHMARK_JSON_PREFIX):])
    raise RuntimeError("command did not emit a structured BENCHMARK_RESULT_JSON payload")


def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict:
    run_env = os.environ.copy()
    run_env.setdefault("MPLCONFIGDIR", "/tmp/mpl-runtime-eval")
    if env:
        run_env.update(env)
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, env=run_env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return parse_summary(completed.stdout)


def divisors(value: int) -> list[int]:
    return [candidate for candidate in (1, 2, 4, 8, 16) if candidate <= value and value % candidate == 0]


def parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def resolve_granularities(raw: str | None, partitions: int) -> list[int]:
    if not raw:
        return divisors(partitions)
    values = []
    for token in parse_csv_tokens(raw):
        value = int(token)
        if value <= 0 or value > partitions or partitions % value != 0:
            raise ValueError(
                f"invalid granularity {value}: it must divide partitions={partitions}"
            )
        values.append(value)
    return values


def save_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote {path}")


def append_rows(path: Path, rows: list[dict]) -> None:
    existing = []
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = payload.get("results", [])
    save_payload(path, {"results": existing + rows})


def best_ptp_configuration(rows: list[dict]) -> tuple[str, int] | None:
    best = None
    for row in rows:
        throughput = row.get("metrics", {}).get("throughput_mb_s")
        if throughput is None:
            continue
        config = row.get("configuration", {})
        candidate = (float(throughput), config.get("backend"), int(config.get("chunk_size", 0)))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return best[1], best[2]


def run_characterization(args: argparse.Namespace) -> None:
    python = repo_python("labelpropagation")
    startup_rows = []
    load_rows = []
    ptp_rows = []
    ptp_burst_size_rows = []
    collective_rows = []
    worker_endpoint = os.environ.get("S3_WORKER_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000")
    host_endpoint = os.environ.get("S3_HOST_ENDPOINT", "http://localhost:9000")
    bucket = os.environ.get("S3_BUCKET", "test-bucket")

    for workers in args.workers:
        load_prefix = f"graphs/load-probe-{args.load_nodes}-{workers}"
        ensure_lp_dataset(args.load_nodes, workers, host_endpoint, bucket, prefix=load_prefix)
        for granularity in divisors(workers):
            startup_cmd = [
                str(python),
                "benchmark_runtime_probe.py",
                "--mode", "startup",
                "--workers", str(workers),
                "--granularity", str(granularity),
                "--memory", str(args.probe_memory_mb),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
            ]
            log(f"startup workers={workers} granularity={granularity}")
            startup_rows.append(run_command(startup_cmd, cwd=HERE))

            load_cmd = [
                str(python),
                "benchmark_runtime_probe.py",
                "--mode", "load",
                "--workers", str(workers),
                "--granularity", str(granularity),
                "--memory", str(args.probe_memory_mb),
                "--bucket", bucket,
                "--key-prefix", load_prefix,
                "--s3-endpoint", worker_endpoint,
            ]
            log(f"load workers={workers} granularity={granularity}")
            load_rows.append(run_command(load_cmd, cwd=HERE))

            for mode in ("broadcast", "all_to_all"):
                collective_cmd = [
                    str(python),
                    "benchmark_runtime_probe.py",
                    "--mode", mode,
                    "--workers", str(workers),
                    "--granularity", str(granularity),
                    "--memory", str(args.probe_memory_mb),
                    "--payload-bytes", str(args.collective_payload_bytes),
                    "--iterations", str(args.collective_iterations),
                    "--backend", args.backend,
                    "--chunk-size", str(args.chunk_size_kb),
                ]
                log(f"{mode} workers={workers} granularity={granularity}")
                collective_rows.append(run_command(collective_cmd, cwd=HERE))

    for backend in args.backends:
        for chunk_size in args.ptp_chunk_sizes_kb:
            ptp_cmd = [
                str(python),
                "benchmark_runtime_probe.py",
                "--mode", "ptp",
                "--workers", "2",
                "--granularity", "1",
                "--memory", str(args.probe_memory_mb),
                "--payload-bytes", str(args.ptp_payload_bytes),
                "--iterations", str(args.ptp_iterations),
                "--backend", backend,
                "--chunk-size", str(chunk_size),
            ]
            log(f"ptp backend={backend} chunk_size={chunk_size}KB")
            ptp_rows.append(run_command(ptp_cmd, cwd=HERE))

    best_ptp = best_ptp_configuration(ptp_rows)
    if best_ptp is not None:
        best_backend, best_chunk_size = best_ptp
        for workers in args.workers:
            if workers < 4 or workers % 2 != 0:
                continue
            ptp_burst_cmd = [
                str(python),
                "benchmark_runtime_probe.py",
                "--mode", "ptp_pairs",
                "--workers", str(workers),
                "--granularity", "1",
                "--memory", str(args.probe_memory_mb),
                "--payload-bytes", str(args.ptp_payload_bytes),
                "--iterations", str(args.ptp_iterations),
                "--backend", best_backend,
                "--chunk-size", str(best_chunk_size),
            ]
            log(
                f"ptp_pairs workers={workers} backend={best_backend} chunk_size={best_chunk_size}KB"
            )
            ptp_burst_size_rows.append(run_command(ptp_burst_cmd, cwd=HERE))

    save_payload(CHAR_DIR / "startup.json", {"results": startup_rows})
    save_payload(CHAR_DIR / "load.json", {"results": load_rows})
    save_payload(CHAR_DIR / "collectives.json", {"results": collective_rows})
    save_payload(CHAR_DIR / "ptp.json", {"results": ptp_rows})
    save_payload(CHAR_DIR / "ptp_burst_size.json", {"results": ptp_burst_size_rows})


def ensure_sssp_dataset(nodes: int, partitions: int, endpoint: str, bucket: str) -> None:
    python = repo_python("sssp")
    command = [
        str(python), "setup_large_sssp_data.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--bucket", bucket,
        "--endpoint", endpoint,
        "--density", "10",
        "--max-weight", "10.0",
        "--source", "0",
    ]
    subprocess.run(command, cwd=ROOT / "sssp", check=True, text=True, capture_output=True)


def ensure_lp_dataset(nodes: int, partitions: int, endpoint: str, bucket: str, prefix: str | None = None) -> None:
    python = repo_python("labelpropagation")
    command = [
        str(python), "setup_large_lp_data.py",
        "--nodes", str(nodes),
        "--partitions", str(partitions),
        "--bucket", bucket,
        "--endpoint", endpoint,
        "--density", "20",
        "--prefix", prefix or f"graphs/large-{nodes}",
        "--output", f"large_{nodes}.txt",
    ]
    subprocess.run(command, cwd=HERE, check=True, text=True, capture_output=True)


def run_applications(args: argparse.Namespace) -> None:
    worker_endpoint = os.environ.get("S3_WORKER_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000")
    host_endpoint = os.environ.get("S3_HOST_ENDPOINT", "http://localhost:9000")
    bucket = os.environ.get("S3_BUCKET", "test-bucket")

    ensure_sssp_dataset(args.application_nodes, args.app_partitions, host_endpoint, bucket)
    ensure_lp_dataset(args.application_nodes, args.app_partitions, host_endpoint, bucket)

    rows: list[dict] = []
    sssp_python = repo_python("sssp")
    lp_python = repo_python("labelpropagation")
    algorithms = set(parse_csv_tokens(args.algorithms))
    granularities = resolve_granularities(args.app_granularities, args.app_partitions)

    for granularity in granularities:
        if "sssp" in algorithms:
            sssp_cmd = [
                str(sssp_python), "benchmark_sssp.py",
                "--nodes", str(args.application_nodes),
                "--partitions", str(args.app_partitions),
                "--granularity", str(granularity),
                "--max-iterations", str(args.sssp_max_iterations),
                "--memory", str(args.app_memory_mb),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--s3-endpoint", worker_endpoint,
                "--validation-endpoint", host_endpoint,
                "--bucket", bucket,
                "--key-prefix", "graphs",
                "--skip-standalone",
            ]
            log(f"application sssp granularity={granularity}")
            rows.append(run_command(sssp_cmd, cwd=ROOT / "sssp"))

        if "labelpropagation" in algorithms or "lp" in algorithms:
            lp_cmd = [
                str(lp_python), "benchmark_lp.py",
                "--nodes", str(args.application_nodes),
                "--partitions", str(args.app_partitions),
                "--granularity", str(granularity),
                "--iter", str(args.lp_iterations),
                "--memory", str(args.app_memory_mb),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--s3-endpoint", worker_endpoint,
                "--validation-endpoint", host_endpoint,
                "--bucket", bucket,
                "--key-prefix", "graphs",
                "--skip-standalone",
            ]
            log(f"application lp granularity={granularity}")
            rows.append(run_command(lp_cmd, cwd=HERE))

    output_path = APP_DIR / "phase_breakdown.json"
    if args.append:
        append_rows(output_path, rows)
    else:
        save_payload(output_path, {"results": rows})


def parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the runtime characterization and application data campaign.")
    parser.add_argument("--phase", choices=("characterization", "applications"), required=True)
    parser.add_argument("--workers", type=parse_int_list, default=parse_int_list("4,8,16"), help="Comma-separated worker counts.")
    parser.add_argument("--probe-memory-mb", type=int, default=1024)
    parser.add_argument("--collective-payload-bytes", type=int, default=1048576)
    parser.add_argument("--collective-iterations", type=int, default=8)
    parser.add_argument("--ptp-payload-bytes", type=int, default=1048576)
    parser.add_argument("--ptp-iterations", type=int, default=16)
    parser.add_argument("--ptp-chunk-sizes-kb", type=parse_int_list, default=parse_int_list("4,64,256,1024"))
    parser.add_argument("--backend", default="redis-list")
    parser.add_argument("--backends", nargs="*", default=["redis-list", "redis-stream", "s3"])
    parser.add_argument("--chunk-size-kb", type=int, default=1024)
    parser.add_argument("--application-nodes", type=int, default=1_000_000)
    parser.add_argument("--load-nodes", type=int, default=100_000)
    parser.add_argument("--app-partitions", type=int, default=8)
    parser.add_argument("--app-memory-mb", type=int, default=4096)
    parser.add_argument("--app-granularities", default=None, help="Comma-separated granularities to run. Defaults to all divisors of app-partitions.")
    parser.add_argument("--algorithms", default="sssp,labelpropagation", help="Comma-separated application algorithms to run.")
    parser.add_argument("--sssp-max-iterations", type=int, default=500)
    parser.add_argument("--lp-iterations", type=int, default=10)
    parser.add_argument("--append", action="store_true", help="Append application rows to an existing phase_breakdown.json.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.phase == "characterization":
        run_characterization(args)
    else:
        run_applications(args)


if __name__ == "__main__":
    main()
