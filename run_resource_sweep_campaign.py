#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from resource_capacity import (
    HostBudget,
    detect_host_capacity,
    divisors,
    feasible_request_rows,
    max_request,
    parse_memory_to_mb,
    burst_cluster_request,
    spark_cluster_request,
)


ROOT = Path("/home/sergio/src")
HERE = Path(__file__).resolve().parent
SWEEP_ROOT = HERE / "resource_sweep"
PLAN_DIR = SWEEP_ROOT / "plans"
CHAR_DIR = SWEEP_ROOT / "characterization"
FEAS_DIR = SWEEP_ROOT / "feasibility"
CONFIG_DIR = SWEEP_ROOT / "config_sweep"
SIZE_DIR = SWEEP_ROOT / "size_sweep"
SPARK_RESULTS_DIR = HERE / "spark_baseline" / "results"
BENCHMARK_PREFIXES = ("BENCHMARK_RESULT_JSON:", "SPARK_BENCHMARK_RESULT_JSON:")


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


def parse_prefixed_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        token = line.strip()
        for prefix in BENCHMARK_PREFIXES:
            if token.startswith(prefix):
                return json.loads(token[len(prefix):])
    raise RuntimeError("command did not emit a structured benchmark JSON payload")


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.setdefault("MPLCONFIGDIR", "/tmp/mpl-resource-sweep")
    if env:
        run_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            env=run_env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"command timed out after {timeout_seconds}s: {' '.join(command)}\n"
            f"STDOUT:\n{exc.stdout or ''}\nSTDERR:\n{exc.stderr or ''}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def probe_command(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote {path}")


def parse_int_csv(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def parse_str_csv(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def burst_worker_count(partitions: int, granularity: int) -> int:
    if granularity <= 0 or partitions % granularity != 0:
        raise ValueError(f"invalid Burst layout: partitions={partitions}, granularity={granularity}")
    return partitions // granularity


def spark_executor_candidates(partitions: int) -> list[int]:
    return {
        4: [4],
        8: [8],
        16: [16],
        32: [16],
    }[partitions]


def burst_layout_supported(algorithm: str, partitions: int, granularity: int) -> bool:
    workers = burst_worker_count(partitions, granularity)
    if workers < 2:
        return False
    if partitions == 32 and granularity == 16:
        return False
    if algorithm == "labelpropagation" and partitions == 32 and granularity == 2:
        return False
    return True


def normalize_granularities(partitions: int, *, reduced: bool = False) -> list[int]:
    all_values = divisors(partitions)
    all_values = [value for value in all_values if burst_worker_count(partitions, value) >= 2]
    if not reduced:
        return all_values
    reduced_candidates = {
        4: [1, 2, 4],
        8: [1, 2, 4, 8],
        16: [1, 2, 4, 8, 16],
        32: [1, 2, 4, 8, 16, 32],
    }.get(partitions, all_values)
    return [
        value
        for value in reduced_candidates
        if partitions % value == 0 and burst_worker_count(partitions, value) >= 2
    ]


def ensure_dirs() -> None:
    for directory in (PLAN_DIR, CHAR_DIR, FEAS_DIR, CONFIG_DIR, SIZE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_budget(args: argparse.Namespace) -> HostBudget:
    return HostBudget(
        host=detect_host_capacity(),
        reserved_cpus=args.host_reserve_cpus,
        reserved_memory_mb=args.host_reserve_mb,
    )


def build_characterization_rows(args: argparse.Namespace, budget: HostBudget) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workers in parse_int_csv(args.probe_workers):
        for granularity in normalize_granularities(workers):
            rows.append(
                {
                    "workers": workers,
                    "granularity": granularity,
                    "memory_mb": args.probe_memory_mb,
                    "kind": "characterization",
                }
            )
    return feasible_request_rows(
        rows,
        request_builder=lambda row: burst_cluster_request(
            workers=int(row["workers"]),
            memory_per_worker_mb=int(row["memory_mb"]),
            system_reserved_cpus=args.burst_system_reserved_cpus,
            system_reserved_mem_mb=args.burst_system_reserved_mem_mb,
        ),
        budget=budget,
    )


def burst_feasibility_memory_matrix(partitions: int, profile: str = "full") -> list[int]:
    matrices = {
        "full": {
            4: [3072, 4096, 6144],
            8: [2048, 3072, 4096],
            16: [2048],
            32: [2048],
        },
        "safe": {
            4: [3072, 4096],
            8: [2048, 3072],
            16: [2048],
            32: [2048],
        },
    }
    return matrices[profile][partitions]


def spark_feasibility_memory_matrix(executors: int) -> list[str]:
    return {
        4: ["4g", "6g"],
        8: ["3g", "4g", "6g"],
        16: ["3g"],
    }[executors]


def build_burst_feasibility_rows(args: argparse.Namespace, budget: HostBudget) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ("sssp", "labelpropagation"):
        for partitions in parse_int_csv(args.app_partitions):
            for memory_mb in burst_feasibility_memory_matrix(
                partitions, profile=args.burst_feasibility_profile
            ):
                for granularity in normalize_granularities(partitions):
                    if not burst_layout_supported(algorithm, partitions, granularity):
                        continue
                    rows.append(
                        {
                            "framework": "burst",
                            "algorithm": algorithm,
                            "nodes": args.feasibility_nodes,
                            "partitions": partitions,
                            "granularity": granularity,
                            "memory_mb": memory_mb,
                            "runs": 1,
                        }
                    )
    return feasible_request_rows(
        rows,
        request_builder=lambda row: burst_cluster_request(
            workers=burst_worker_count(int(row["partitions"]), int(row["granularity"])),
            memory_per_worker_mb=int(row["memory_mb"]),
            system_reserved_cpus=args.burst_system_reserved_cpus,
            system_reserved_mem_mb=args.burst_system_reserved_mem_mb,
        ),
        budget=budget,
    )


def build_spark_feasibility_rows(args: argparse.Namespace, budget: HostBudget) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ("sssp", "labelpropagation"):
        for partitions in parse_int_csv(args.app_partitions):
            for executors in spark_executor_candidates(partitions):
                for executor_memory in spark_feasibility_memory_matrix(executors):
                    rows.append(
                        {
                            "framework": "spark",
                            "algorithm": algorithm,
                            "nodes": args.feasibility_nodes,
                            "partitions": partitions,
                            "executors": executors,
                            "executor_memory": executor_memory,
                            "runs": 1,
                        }
                    )
    return feasible_request_rows(
        rows,
        request_builder=lambda row: spark_cluster_request(
            executors=int(row["executors"]),
            executor_memory=str(row["executor_memory"]),
            master_cpus=args.spark_master_cpus,
            master_memory=args.spark_master_memory,
        ),
        budget=budget,
    )


def build_config_sweep_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in parse_str_csv(args.algorithms):
        reduced = algorithm in {"bfs", "wcc"}
        for partitions in parse_int_csv(args.app_partitions):
            for granularity in normalize_granularities(partitions, reduced=reduced):
                if not burst_layout_supported(algorithm, partitions, granularity):
                    continue
                rows.append(
                    {
                        "framework": "burst",
                        "algorithm": algorithm,
                        "nodes": args.config_nodes,
                        "partitions": partitions,
                        "granularity": granularity,
                        "runs": args.config_runs,
                        "memory_selection": "best_stable_from_feasibility",
                    }
                )
        for partitions in parse_int_csv(args.app_partitions):
            for executors in spark_executor_candidates(partitions):
                rows.append(
                    {
                        "framework": "spark",
                        "algorithm": algorithm,
                        "nodes": args.config_nodes,
                        "partitions": partitions,
                        "executors": executors,
                        "runs": args.config_runs,
                        "memory_selection": "best_stable_from_feasibility",
                    }
                )
    return rows


def build_size_sweep_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in parse_str_csv(args.algorithms):
        for partitions in parse_int_csv(args.app_partitions):
            for nodes in parse_int_csv(args.size_nodes):
                rows.append(
                    {
                        "algorithm": algorithm,
                        "nodes": nodes,
                        "partitions": partitions,
                        "runs": args.size_runs,
                        "selection": "best_config_from_config_sweep",
                    }
                )
    return rows


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    budget = build_budget(args)
    characterization = build_characterization_rows(args, budget)
    burst_feasibility = build_burst_feasibility_rows(args, budget)
    spark_feasibility = build_spark_feasibility_rows(args, budget)
    config_sweep = build_config_sweep_rows(args)
    size_sweep = build_size_sweep_rows(args)
    return {
        "timestamp": datetime.now().isoformat(),
        "host_budget": budget.to_dict(),
        "characterization": {
            "rows": characterization,
            "max_cluster_profile": max_request(characterization),
        },
        "burst_feasibility": {
            "rows": burst_feasibility,
            "max_cluster_profile": max_request(burst_feasibility),
        },
        "spark_feasibility": {
            "rows": spark_feasibility,
            "max_cluster_profile": max_request(spark_feasibility),
        },
        "config_sweep": {
            "rows": config_sweep,
            "note": "La memoria por particion/executor se resolvera a partir de la criba de factibilidad.",
        },
        "size_sweep": {
            "rows": size_sweep,
            "note": "La mejor configuracion se elige por tiempo end-to-end y validacion correcta.",
        },
        "constraints": {
            "single_worker_burst": "Se excluyen configuraciones con workers=1 (granularity=partitions) porque no representan paralelismo distribuido real.",
            "power_of_two_burst_sizes": "Las comparativas principales usan burst sizes potencia de dos porque el reduce actual del middleware exige esa topologia.",
            "burst_runtime_filters": "Se excluyen configuraciones ya diagnosticadas como no defendibles: P=32,g=16 para Burst y LP con P=32,g=2.",
            "phase_separation": "Burst/OpenWhisk y Spark se ejecutan en fases separadas.",
        },
    }


def annotate_burst_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budget = build_budget(args)
    return feasible_request_rows(
        rows,
        request_builder=lambda row: burst_cluster_request(
            workers=burst_worker_count(int(row["partitions"]), int(row["granularity"])),
            memory_per_worker_mb=int(row["memory_mb"]),
            system_reserved_cpus=args.burst_system_reserved_cpus,
            system_reserved_mem_mb=args.burst_system_reserved_mem_mb,
        ),
        budget=budget,
    )


def annotate_spark_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budget = build_budget(args)
    return feasible_request_rows(
        rows,
        request_builder=lambda row: spark_cluster_request(
            executors=int(row["executors"]),
            executor_memory=str(row["executor_memory"]),
            master_cpus=args.spark_master_cpus,
            master_memory=args.spark_master_memory,
        ),
        budget=budget,
    )


def save_plan(args: argparse.Namespace) -> Path:
    ensure_dirs()
    path = PLAN_DIR / "resource_sweep_plan.json"
    write_json(path, build_plan(args))
    return path


def ensure_bfs_dataset(nodes: int, partitions: int, endpoint: str, bucket: str) -> None:
    python = repo_python("bfs")
    run_command(
        [
            str(python),
            "setup_large_bfs_data.py",
            "--nodes", str(nodes),
            "--partitions", str(partitions),
            "--bucket", bucket,
            "--endpoint", endpoint,
            "--density", "10",
            "--source", "0",
        ],
        cwd=ROOT / "bfs",
        timeout_seconds=900,
    )


def ensure_sssp_dataset(nodes: int, partitions: int, endpoint: str, bucket: str) -> None:
    python = repo_python("sssp")
    run_command(
        [
            str(python),
            "setup_large_sssp_data.py",
            "--nodes", str(nodes),
            "--partitions", str(partitions),
            "--bucket", bucket,
            "--endpoint", endpoint,
            "--density", "10",
            "--max-weight", "10.0",
            "--source", "0",
        ],
        cwd=ROOT / "sssp",
        timeout_seconds=900,
    )


def ensure_lp_dataset(nodes: int, partitions: int, endpoint: str, bucket: str) -> None:
    python = repo_python("labelpropagation")
    run_command(
        [
            str(python),
            "setup_large_lp_data.py",
            "--nodes", str(nodes),
            "--partitions", str(partitions),
            "--bucket", bucket,
            "--endpoint", endpoint,
            "--density", "20",
            "--prefix", f"graphs/large-{nodes}",
            "--output", f"large_{nodes}.txt",
        ],
        cwd=HERE,
        timeout_seconds=900,
    )


def ensure_wcc_dataset(nodes: int, partitions: int, endpoint: str, bucket: str) -> None:
    python = repo_python("unionfind")
    run_command(
        [
            str(python),
            "setup_large_uf_data.py",
            "--nodes", str(nodes),
            "--partitions", str(partitions),
            "--bucket", bucket,
            "--endpoint", endpoint,
            "--output", f"wcc_graph_{nodes}.tsv",
            "--format", "binary",
        ],
        cwd=ROOT / "unionfind",
        timeout_seconds=900,
    )


def prepare_dataset(algorithm: str, nodes: int, partitions: int, host_endpoint: str, bucket: str) -> None:
    if algorithm == "bfs":
        ensure_bfs_dataset(nodes, partitions, host_endpoint, bucket)
    elif algorithm == "sssp":
        ensure_sssp_dataset(nodes, partitions, host_endpoint, bucket)
    elif algorithm == "labelpropagation":
        ensure_lp_dataset(nodes, partitions, host_endpoint, bucket)
    elif algorithm == "wcc":
        ensure_wcc_dataset(nodes, partitions, host_endpoint, bucket)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")


def start_burst_cluster(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    profile = max_request(rows)
    if profile is None:
        raise RuntimeError("no feasible burst configuration found")
    if burst_cluster_is_ready(profile):
        log(
            f"reusing existing Burst cluster with enough capacity for {int(profile['cpus'])} CPU / {int(profile['memory_mb'])}MB"
        )
        return
    env = {
        "OW_HOST_RESERVED_CPUS": str(args.host_reserve_cpus),
        "OW_HOST_RESERVED_MEM_MB": str(args.host_reserve_mb),
        "OW_SYSTEM_RESERVED_CPUS": str(args.burst_system_reserved_cpus),
        "OW_SYSTEM_RESERVED_MEM_MB": str(args.burst_system_reserved_mem_mb),
        "OW_CLUSTER_CPUS": str(int(profile["cpus"])),
        "OW_CLUSTER_MEMORY_MB": str(int(profile["memory_mb"])),
        "OW_WORKER_COUNT": str(
            max(
                burst_worker_count(int(row["partitions"]), int(row["granularity"]))
                if "partitions" in row and "granularity" in row
                else int(row.get("workers", row.get("partitions", 1)))
                for row in rows
            )
        ),
        "OW_MEMORY_PER_WORKER_MB": str(
            max(int(row.get("memory_mb", args.probe_memory_mb)) for row in rows)
        ),
    }
    log(
        f"starting Burst cluster with {env['OW_CLUSTER_CPUS']} CPU and {env['OW_CLUSTER_MEMORY_MB']}MB"
    )
    run_command(
        ["bash", str(ROOT / "openwhisk-deploy-kube-burst" / "start-exposed.sh")],
        cwd=ROOT / "openwhisk-deploy-kube-burst",
        env=env,
    )
    wait_for_burst_cluster(profile)


def wait_for_burst_cluster(profile: dict[str, Any], timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if burst_cluster_is_ready(profile):
            log("Burst cluster is ready")
            return
        time.sleep(3)
    raise RuntimeError(
        f"Burst cluster did not become ready within {timeout_seconds}s "
        f"for {int(profile['cpus'])} CPU / {int(profile['memory_mb'])}MB"
    )


def burst_cluster_is_ready(profile: dict[str, Any]) -> bool:
    status = probe_command(["minikube", "status"])
    if status.returncode != 0 or "host: Running" not in status.stdout:
        return False

    capacity = probe_command(
        [
            "kubectl",
            "get",
            "node",
            "minikube",
            "-o",
            "jsonpath={.status.allocatable.cpu} {.status.allocatable.memory}",
        ]
    )
    if capacity.returncode != 0:
        return False
    parts = capacity.stdout.strip().split()
    if len(parts) != 2:
        return False
    try:
        allocatable_cpus = int(parts[0])
        allocatable_memory_mb = int(parts[1].removesuffix("Ki")) // 1024
    except ValueError:
        return False
    if allocatable_cpus < int(profile["cpus"]) or allocatable_memory_mb < int(profile["memory_mb"]):
        return False

    endpoint = probe_command(["curl", "-sk", "https://127.0.0.1:31001"])
    return endpoint.returncode == 0


def start_spark_cluster(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    feasible = [row for row in rows if row.get("feasible", True)]
    if not feasible:
        raise RuntimeError("no feasible spark configuration found")
    max_workers = max(int(row["executors"]) for row in feasible)
    max_memory = max(parse_memory_to_mb(str(row["executor_memory"])) for row in feasible)
    env = {
        "SPARK_WORKER_REPLICAS": str(max_workers),
        "SPARK_MASTER_CPUS": str(args.spark_master_cpus),
        "SPARK_MASTER_MEMORY": args.spark_master_memory,
        "SPARK_WORKER_CORES": "1",
        "SPARK_WORKER_MEMORY": f"{max_memory}m",
        "SPARK_WORKER_CPUS_LIMIT": "1.0",
        "SPARK_WORKER_MEMORY_LIMIT": f"{max_memory}m",
    }
    log(f"starting Spark cluster with {max_workers} workers and {max_memory}MB per worker")
    run_command(
        ["bash", str(HERE / "spark_baseline" / "scripts" / "start-cluster.sh")],
        cwd=HERE / "spark_baseline",
        env=env,
    )


def stop_spark_cluster() -> None:
    run_command(
        ["bash", str(HERE / "spark_baseline" / "scripts" / "stop-cluster.sh")],
        cwd=HERE / "spark_baseline",
    )


def normalize_burst_result(summary: dict[str, Any]) -> dict[str, Any]:
    burst = summary.get("burst", {})
    validation = summary.get("validation", {})
    host_total_ms = burst.get("host_total_time_ms")
    warm_total_ms = burst.get("total_time_ms")
    processing_ms = burst.get("processing_time_ms")
    return {
        "status": "passed" if host_total_ms is not None and validation.get("passed", False) else "failed",
        "primary_metric_ms": host_total_ms if host_total_ms is not None else warm_total_ms,
        "warm_total_ms": warm_total_ms,
        "processing_ms": processing_ms,
        "validation": validation,
        "summary": summary,
    }


def run_burst_benchmark(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    algorithm = str(row["algorithm"])
    nodes = int(row["nodes"])
    partitions = int(row["partitions"])
    granularity = int(row["granularity"])
    memory_mb = int(row["memory_mb"])
    bucket = args.bucket
    host_endpoint = args.host_s3_endpoint
    worker_endpoint = args.worker_s3_endpoint
    prepare_dataset(algorithm, nodes, partitions, host_endpoint, bucket)

    if algorithm == "bfs":
        completed = run_command(
            [
                str(repo_python("bfs")),
                "benchmark_bfs.py",
                "--nodes", str(nodes),
                "--partitions", str(partitions),
                "--granularity", str(granularity),
                "--max-levels", str(args.bfs_max_levels),
                "--memory", str(memory_mb),
                "--ow-host", args.ow_host,
                "--ow-port", str(args.ow_port),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--validate",
                "--s3-endpoint", worker_endpoint,
                "--validation-endpoint", host_endpoint,
                "--bucket", bucket,
                "--key-prefix", args.key_prefix,
            ],
            cwd=ROOT / "bfs",
            timeout_seconds=args.command_timeout_sec,
        )
    elif algorithm == "sssp":
        completed = run_command(
            [
                str(repo_python("sssp")),
                "benchmark_sssp.py",
                "--nodes", str(nodes),
                "--partitions", str(partitions),
                "--granularity", str(granularity),
                "--max-iterations", str(args.sssp_max_iterations),
                "--memory", str(memory_mb),
                "--ow-host", args.ow_host,
                "--ow-port", str(args.ow_port),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--validate",
                "--s3-endpoint", worker_endpoint,
                "--validation-endpoint", host_endpoint,
                "--bucket", bucket,
                "--key-prefix", args.key_prefix,
            ],
            cwd=ROOT / "sssp",
            timeout_seconds=args.command_timeout_sec,
        )
    elif algorithm == "labelpropagation":
        completed = run_command(
            [
                str(repo_python("labelpropagation")),
                "benchmark_lp.py",
                "--nodes", str(nodes),
                "--partitions", str(partitions),
                "--granularity", str(granularity),
                "--iter", str(args.lp_iterations),
                "--memory", str(memory_mb),
                "--ow-host", args.ow_host,
                "--ow-port", str(args.ow_port),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--validate",
                "--s3-endpoint", worker_endpoint,
                "--validation-endpoint", host_endpoint,
                "--bucket", bucket,
                "--key-prefix", args.key_prefix,
            ],
            cwd=HERE,
            timeout_seconds=args.command_timeout_sec,
        )
    elif algorithm == "wcc":
        output_path = FEAS_DIR / f"wcc_{nodes}_{partitions}_{granularity}_{memory_mb}.json"
        completed = run_command(
            [
                str(repo_python("unionfind")),
                "benchmark_uf.py",
                "--ow-host", args.ow_host,
                "--ow-port", str(args.ow_port),
                "--runtime-memory", str(memory_mb),
                "--backend", args.backend,
                "--chunk-size", str(args.chunk_size_kb),
                "--wcc-endpoint", worker_endpoint,
                "--local-endpoint", host_endpoint,
                "--partitions", str(partitions),
                "--bucket", bucket,
                "--granularity", str(granularity),
                "--sizes", str(nodes),
                "--output", str(output_path),
            ],
            cwd=ROOT / "unionfind",
            timeout_seconds=args.command_timeout_sec,
        )
    else:
        raise ValueError(f"unsupported burst algorithm: {algorithm}")

    return normalize_burst_result(parse_prefixed_json(completed.stdout))


def load_spark_result(algorithm: str, nodes: int) -> dict[str, Any]:
    path = SPARK_RESULTS_DIR / f"{algorithm}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results", []):
        if int(row.get("nodes", -1)) == nodes:
            return row
    raise RuntimeError(f"Spark result for {algorithm} nodes={nodes} not found in {path}")


def normalize_spark_result(row: dict[str, Any]) -> dict[str, Any]:
    validation = row.get("validation", {})
    total_ms = row.get("spark_total_ms")
    return {
        "status": "passed" if total_ms is not None and validation.get("passed", False) else "failed",
        "primary_metric_ms": total_ms,
        "validation": validation,
        "summary": row,
    }


def run_spark_benchmark(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    algorithm = str(row["algorithm"])
    nodes = int(row["nodes"])
    partitions = int(row["partitions"])
    executors = int(row["executors"])
    executor_memory = str(row["executor_memory"])
    completed = run_command(
        [
            str(repo_python("labelpropagation")),
            str(HERE / "spark_baseline" / "run_spark_graph_benchmarks.py"),
            "--algorithms", algorithm,
            "--runs", str(int(row["runs"])),
            "--force",
            "--partitions", str(partitions),
            "--executors", str(executors),
            "--executor-memory", executor_memory,
            "--points", str(nodes),
        ],
        cwd=HERE / "spark_baseline",
        timeout_seconds=args.command_timeout_sec,
    )
    if completed.stdout:
        log(f"spark runner finished for {algorithm} p={partitions} e={executors} mem={executor_memory}")
    return normalize_spark_result(load_spark_result(algorithm, nodes))


def persist_phase_rows(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    write_json(path, {"metadata": metadata, "results": rows})


def load_phase_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("results", [])


ROW_KEY_FIELDS = (
    "kind",
    "framework",
    "algorithm",
    "nodes",
    "partitions",
    "granularity",
    "memory_mb",
    "executors",
    "executor_memory",
    "runs",
    "workers",
)


def phase_row_key(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((field, row.get(field)) for field in ROW_KEY_FIELDS if field in row)


def load_existing_phase_results(
    path: Path,
    *,
    resume_existing: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[tuple[str, Any], ...], int]]:
    if not resume_existing or not path.exists():
        return [], {}
    rows = load_phase_rows(path)
    index = {phase_row_key(row): position for position, row in enumerate(rows)}
    return rows, index


def should_retry_existing_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    status = str(row.get("status", "")).lower()
    if status == "passed":
        return False
    if status == "failed":
        return args.retry_failed
    if status == "skipped":
        return args.retry_skipped
    return True


def upsert_phase_result(
    results: list[dict[str, Any]],
    index: dict[tuple[tuple[str, Any], ...], int],
    row: dict[str, Any],
) -> None:
    key = phase_row_key(row)
    position = index.get(key)
    if position is None:
        index[key] = len(results)
        results.append(row)
        return
    results[position] = row


def select_best_memory_map(rows: list[dict[str, Any]], *, framework: str) -> dict[str, dict[int, Any]]:
    selected: dict[str, dict[int, Any]] = {}
    for row in rows:
        if row.get("framework") != framework:
            continue
        if row.get("status") != "passed":
            continue
        algorithm = str(row["algorithm"])
        partitions = int(row["partitions"])
        selected.setdefault(algorithm, {})
        current = selected[algorithm].get(partitions)
        candidate_memory = row["memory_mb"] if framework == "burst" else row["executor_memory"]
        if current is None:
            selected[algorithm][partitions] = candidate_memory
            continue
        current_mb = int(current) if framework == "burst" else parse_memory_to_mb(str(current))
        candidate_mb = int(candidate_memory) if framework == "burst" else parse_memory_to_mb(str(candidate_memory))
        if candidate_mb > current_mb:
            selected[algorithm][partitions] = candidate_memory
    return selected


def select_best_config_rows(rows: list[dict[str, Any]], *, framework: str) -> dict[tuple[str, int], dict[str, Any]]:
    winners: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("framework") != framework or row.get("status") != "passed":
            continue
        key = (str(row["algorithm"]), int(row["partitions"]))
        current = winners.get(key)
        if current is None or float(row["primary_metric_ms"]) < float(current["primary_metric_ms"]):
            winners[key] = row
    return winners


def materialize_burst_config_rows(args: argparse.Namespace, memory_map: dict[str, dict[int, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_map = {4: 4096, 8: 3072, 16: 2048, 32: 2048}
    for template in build_config_sweep_rows(args):
        if template["framework"] != "burst":
            continue
        algorithm = str(template["algorithm"])
        partitions = int(template["partitions"])
        memory_mb = memory_map.get(algorithm, {}).get(partitions)
        if memory_mb is None:
            memory_mb = memory_map.get("sssp", {}).get(partitions, fallback_map[partitions])
        rows.append({**template, "memory_mb": int(memory_mb)})
    return rows


def materialize_spark_config_rows(args: argparse.Namespace, memory_map: dict[str, dict[int, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fallback_map = {4: "4g", 8: "4g", 16: "3g", 32: "3g"}
    for template in build_config_sweep_rows(args):
        if template["framework"] != "spark":
            continue
        algorithm = str(template["algorithm"])
        partitions = int(template["partitions"])
        executor_memory = memory_map.get(algorithm, {}).get(partitions)
        if executor_memory is None:
            executor_memory = memory_map.get("sssp", {}).get(partitions, fallback_map[partitions])
        rows.append({**template, "executor_memory": str(executor_memory)})
    return rows


def apply_limit(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


def run_burst_phase(args: argparse.Namespace, phase_name: str, rows: list[dict[str, Any]], output_path: Path) -> Path:
    selected = apply_limit(rows, args.max_configs)
    results, result_index = load_existing_phase_results(output_path, resume_existing=args.resume_existing)
    if args.prepare_burst_cluster:
        start_burst_cluster(args, selected)
    for row in selected:
        row_key = phase_row_key(row)
        existing = results[result_index[row_key]] if row_key in result_index else None
        if existing is not None and not should_retry_existing_row(existing, args):
            log(
                f"Skipping existing {phase_name} burst {row['algorithm']} "
                f"n={row.get('nodes', '-')} p={row.get('partitions', '-')} "
                f"g={row.get('granularity', '-')} [{existing.get('status', 'unknown')}]"
            )
            continue
        if not row.get("feasible", True):
            upsert_phase_result(results, result_index, {**row, "status": "skipped", "skip_reason": "exceeds host budget"})
            continue
        log(
            f"{phase_name} burst {row['algorithm']} n={row['nodes']} p={row['partitions']} g={row['granularity']}"
        )
        try:
            outcome = run_burst_benchmark(row, args)
            upsert_phase_result(results, result_index, {**row, **outcome})
        except Exception as exc:
            upsert_phase_result(results, result_index, {**row, "status": "failed", "error": str(exc)})
        persist_phase_rows(output_path, {"phase": phase_name, "partial": True}, results)
    persist_phase_rows(output_path, {"phase": phase_name, "partial": False}, results)
    return output_path


def run_spark_phase(args: argparse.Namespace, phase_name: str, rows: list[dict[str, Any]], output_path: Path) -> Path:
    selected = apply_limit(rows, args.max_configs)
    results, result_index = load_existing_phase_results(output_path, resume_existing=args.resume_existing)
    started = False
    try:
        if args.prepare_spark_cluster:
            start_spark_cluster(args, selected)
            started = True
        for row in selected:
            row_key = phase_row_key(row)
            existing = results[result_index[row_key]] if row_key in result_index else None
            if existing is not None and not should_retry_existing_row(existing, args):
                log(
                    f"Skipping existing {phase_name} spark {row['algorithm']} "
                    f"n={row.get('nodes', '-')} p={row.get('partitions', '-')} "
                    f"e={row.get('executors', '-')} [{existing.get('status', 'unknown')}]"
                )
                continue
            if not row.get("feasible", True):
                upsert_phase_result(results, result_index, {**row, "status": "skipped", "skip_reason": "exceeds host budget"})
                continue
            log(
                f"{phase_name} spark {row['algorithm']} n={row['nodes']} p={row['partitions']} e={row['executors']}"
            )
            try:
                outcome = run_spark_benchmark(row, args)
                upsert_phase_result(results, result_index, {**row, **outcome})
            except Exception as exc:
                upsert_phase_result(results, result_index, {**row, "status": "failed", "error": str(exc)})
            persist_phase_rows(output_path, {"phase": phase_name, "partial": True}, results)
        persist_phase_rows(output_path, {"phase": phase_name, "partial": False}, results)
        return output_path
    finally:
        if started and args.stop_spark_cluster:
            stop_spark_cluster()


def run_burst_feasibility(args: argparse.Namespace) -> Path:
    budget = build_budget(args)
    rows = build_burst_feasibility_rows(args, budget)
    return run_burst_phase(args, "burst-feasibility", rows, FEAS_DIR / "burst_feasibility.json")


def run_spark_feasibility(args: argparse.Namespace) -> Path:
    budget = build_budget(args)
    rows = build_spark_feasibility_rows(args, budget)
    return run_spark_phase(args, "spark-feasibility", rows, FEAS_DIR / "spark_feasibility.json")


def run_config_sweep(args: argparse.Namespace) -> list[Path]:
    burst_memory = select_best_memory_map(load_phase_rows(FEAS_DIR / "burst_feasibility.json"), framework="burst")
    spark_memory = select_best_memory_map(load_phase_rows(FEAS_DIR / "spark_feasibility.json"), framework="spark")
    burst_rows = annotate_burst_rows(args, materialize_burst_config_rows(args, burst_memory))
    spark_rows = annotate_spark_rows(args, materialize_spark_config_rows(args, spark_memory))
    outputs: list[Path] = []
    if args.framework_scope in {"both", "burst"}:
        outputs.append(
            run_burst_phase(args, "config-sweep-burst", burst_rows, CONFIG_DIR / "burst_config_sweep.json")
        )
    if args.framework_scope in {"both", "spark"}:
        outputs.append(
            run_spark_phase(args, "config-sweep-spark", spark_rows, CONFIG_DIR / "spark_config_sweep.json")
        )
    return outputs


def run_size_sweep(args: argparse.Namespace) -> list[Path]:
    burst_winners = select_best_config_rows(load_phase_rows(CONFIG_DIR / "burst_config_sweep.json"), framework="burst")
    spark_winners = select_best_config_rows(load_phase_rows(CONFIG_DIR / "spark_config_sweep.json"), framework="spark")
    burst_rows: list[dict[str, Any]] = []
    spark_rows: list[dict[str, Any]] = []
    for template in build_size_sweep_rows(args):
        key = (str(template["algorithm"]), int(template["partitions"]))
        burst_winner = burst_winners.get(key)
        if burst_winner is not None:
            burst_rows.append(
                {
                    "framework": "burst",
                    "algorithm": template["algorithm"],
                    "nodes": template["nodes"],
                    "partitions": template["partitions"],
                    "granularity": burst_winner["granularity"],
                    "memory_mb": burst_winner["memory_mb"],
                    "runs": template["runs"],
                }
            )
        spark_winner = spark_winners.get(key)
        if spark_winner is not None:
            spark_rows.append(
                {
                    "framework": "spark",
                    "algorithm": template["algorithm"],
                    "nodes": template["nodes"],
                    "partitions": template["partitions"],
                    "executors": spark_winner["executors"],
                    "executor_memory": spark_winner["executor_memory"],
                    "runs": template["runs"],
                }
            )
    outputs: list[Path] = []
    if burst_rows and args.framework_scope in {"both", "burst"}:
        outputs.append(
            run_burst_phase(
                args,
                "size-sweep-burst",
                annotate_burst_rows(args, burst_rows),
                SIZE_DIR / "burst_size_sweep.json",
            )
        )
    if spark_rows and args.framework_scope in {"both", "spark"}:
        outputs.append(
            run_spark_phase(
                args,
                "size-sweep-spark",
                annotate_spark_rows(args, spark_rows),
                SIZE_DIR / "spark_size_sweep.json",
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resource-aware graph benchmark sweep driver.")
    parser.add_argument(
        "--phase",
        default="plan",
        choices=["plan", "burst-feasibility", "spark-feasibility", "config-sweep", "size-sweep"],
        help="Sweep phase to execute.",
    )
    parser.add_argument("--algorithms", default="bfs,sssp,labelpropagation,wcc")
    parser.add_argument("--probe-workers", default="4,8,16")
    parser.add_argument("--app-partitions", default="4,8,16,32")
    parser.add_argument("--size-nodes", default="100000,500000,1000000,2000000")
    parser.add_argument("--feasibility-nodes", type=int, default=500000)
    parser.add_argument("--config-nodes", type=int, default=500000)
    parser.add_argument("--config-runs", type=int, default=3)
    parser.add_argument("--size-runs", type=int, default=3)
    parser.add_argument("--max-configs", type=int, default=0, help="Optional cap for debugging a phase.")
    parser.add_argument("--probe-memory-mb", type=int, default=1024)
    parser.add_argument("--host-reserve-mb", type=int, default=8192)
    parser.add_argument("--host-reserve-cpus", type=int, default=2)
    parser.add_argument("--burst-system-reserved-cpus", type=int, default=6)
    parser.add_argument("--burst-system-reserved-mem-mb", type=int, default=8192)
    parser.add_argument("--spark-master-cpus", type=float, default=1.0)
    parser.add_argument("--spark-master-memory", default="1g")
    parser.add_argument("--prepare-burst-cluster", action="store_true")
    parser.add_argument("--prepare-spark-cluster", action="store_true")
    parser.add_argument("--stop-spark-cluster", action="store_true")
    parser.add_argument("--ow-host", default=os.environ.get("OW_HOST", "localhost"))
    parser.add_argument("--ow-port", type=int, default=int(os.environ.get("OW_PORT", "31001")))
    parser.add_argument("--worker-s3-endpoint", default=os.environ.get("S3_WORKER_ENDPOINT", "http://minio-service.default.svc.cluster.local:9000"))
    parser.add_argument("--host-s3-endpoint", default=os.environ.get("S3_HOST_ENDPOINT", "http://localhost:9000"))
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "test-bucket"))
    parser.add_argument("--key-prefix", default="graphs")
    parser.add_argument("--backend", default="redis-list")
    parser.add_argument("--chunk-size-kb", type=int, default=1024)
    parser.add_argument("--bfs-max-levels", type=int, default=500)
    parser.add_argument("--sssp-max-iterations", type=int, default=500)
    parser.add_argument("--lp-iterations", type=int, default=10)
    parser.add_argument("--command-timeout-sec", type=int, default=600)
    parser.add_argument(
        "--resume-existing",
        dest="resume_existing",
        action="store_true",
        default=True,
        help="Reuse existing phase JSON outputs and skip rows that are already complete.",
    )
    parser.add_argument(
        "--no-resume-existing",
        dest="resume_existing",
        action="store_false",
        help="Ignore any existing phase JSON output and rebuild the phase from scratch.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, re-run rows already recorded as failed.",
    )
    parser.add_argument(
        "--retry-skipped",
        action="store_true",
        help="When resuming, re-run rows already recorded as skipped.",
    )
    parser.add_argument(
        "--burst-feasibility-profile",
        choices=["full", "safe"],
        default="full",
        help="Memory matrix for Burst feasibility sweeps.",
    )
    parser.add_argument(
        "--framework-scope",
        choices=["both", "burst", "spark"],
        default="both",
        help="For config/size sweeps, run only the requested framework.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.phase == "plan":
        save_plan(args)
        return
    if args.phase == "burst-feasibility":
        run_burst_feasibility(args)
        return
    if args.phase == "spark-feasibility":
        run_spark_feasibility(args)
        return
    if args.phase == "config-sweep":
        run_config_sweep(args)
        return
    if args.phase == "size-sweep":
        run_size_sweep(args)
        return
    raise SystemExit(f"unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()
