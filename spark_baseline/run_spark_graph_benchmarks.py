#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/home/sergio/src")
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
RESULT_PREFIX = "SPARK_BENCHMARK_RESULT_JSON:"
UNVISITED = 2**32 - 1
@dataclass(frozen=True)
class SparkAlgorithm:
    slug: str
    x_key: str
    points: list[int]
    generator_repo: str | None
    generator_script: str | None
    dataset_name_fn: Any
    generator_args_fn: Any
    submit_script: str | None
    submit_args_fn: Any
    configuration: dict[str, Any]
    supported: bool = True
    unsupported_reason: str | None = None


def connected_components_algorithm(slug: str) -> SparkAlgorithm:
    return SparkAlgorithm(
        slug=slug,
        x_key="nodes",
        points=[100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000],
        generator_repo="unionfind",
        generator_script="setup_large_uf_data.py",
        dataset_name_fn=lambda size: f"wcc_graph_{size}.tsv",
        generator_args_fn=lambda size, dataset: [
            "--nodes", str(size),
            "--edges-per-node", "5",
            "--components", "10",
            "--partitions", "4",
            "--output", str(dataset),
            "--key", f"wcc-graphs/wcc-{size}",
            "--no-s3",
        ],
        submit_script="submit-wcc.sh",
        submit_args_fn=lambda container_input, output_path, _size: [
            container_input,
            output_path,
            "4",
        ],
        configuration={
            "partitions": 4,
            "edges_per_node": 5,
            "components": 10,
            "spark_env": {
                "SPARK_TOTAL_EXECUTOR_CORES": "4",
                "SPARK_EXECUTOR_CORES": "1",
                "SPARK_EXECUTOR_MEMORY": "4g",
                "SPARK_DEFAULT_PARALLELISM": "4",
                "SPARK_SHUFFLE_PARTITIONS": "4",
            },
        },
    )


def checkpoint_path(algorithm: SparkAlgorithm) -> Path:
    return RESULTS_DIR / f"{algorithm.slug}.partial.json"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def repo_python(repo_name: str) -> Path:
    candidates = [
        ROOT / repo_name / ".venv/bin/python",
        ROOT / "labelpropagation/.venv/bin/python",
        ROOT / "bfs/.venv/bin/python",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            probe = subprocess.run(
                [str(candidate), "--version"],
                text=True,
                capture_output=True,
                timeout=10,
            )
            if probe.returncode == 0:
                return candidate
        except Exception:
            continue
    return Path(sys.executable)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(RESULTS_DIR, 0o777)


def remove_output_dir(host_output_dir: Path, container_output_dir: str) -> None:
    if not host_output_dir.exists():
        return
    try:
        shutil.rmtree(host_output_dir)
        return
    except PermissionError:
        pass

    cleanup = subprocess.run(
        ["docker", "exec", "spark-master", "rm", "-rf", container_output_dir],
        text=True,
        capture_output=True,
    )
    if cleanup.returncode != 0:
        raise RuntimeError(
            f"failed to clean Spark output directory {container_output_dir}\n"
            f"STDOUT:\n{cleanup.stdout}\nSTDERR:\n{cleanup.stderr}"
        )
    if host_output_dir.exists():
        shutil.rmtree(host_output_dir)


def parse_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise ValueError("spark benchmark output did not contain a structured JSON result")


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def run_lp_standalone(graph_file: Path, num_nodes: int, max_iter: int) -> dict[str, Any]:
    binary_path = ROOT / "labelpropagation/lpst/target/release/label-propagation"
    if not binary_path.exists():
        raise RuntimeError(f"standalone LP binary not found at {binary_path}")
    completed = run_command(
        [str(binary_path), str(graph_file), str(num_nodes), str(max_iter)],
        cwd=binary_path.parent.parent.parent,
    )
    return json.loads(completed.stdout.strip())


def run_bfs_standalone(graph_file: Path, num_nodes: int, source_node: int, max_levels: int) -> dict[str, Any]:
    binary_path = ROOT / "bfs/bfs-standalone/target/release/bfs-standalone"
    if not binary_path.exists():
        raise RuntimeError(f"standalone BFS binary not found at {binary_path}")
    completed = run_command(
        [str(binary_path), str(graph_file), str(num_nodes), str(source_node), str(max_levels)],
        cwd=binary_path.parent.parent.parent,
    )
    return json.loads(completed.stdout.strip())


def run_sssp_standalone(graph_file: Path, num_nodes: int, source_node: int) -> dict[str, Any]:
    binary_path = ROOT / "sssp/sssp-standalone/target/release/sssp-standalone"
    if not binary_path.exists():
        raise RuntimeError(f"standalone SSSP binary not found at {binary_path}")
    completed = run_command(
        [str(binary_path), str(graph_file), str(num_nodes), str(source_node)],
        cwd=binary_path.parent.parent.parent,
    )
    return json.loads(completed.stdout.strip())


def run_wcc_standalone(graph_file: Path, num_nodes: int) -> dict[str, Any]:
    candidates = [
        ROOT / "unionfind/ufst/target/release/union-find",
        ROOT / "unionfind/ufst/target/release/ufst",
    ]
    binary_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if binary_path is None:
        raise RuntimeError(f"standalone WCC binary not found in {candidates}")
    completed = run_command(
        [str(binary_path), str(graph_file), str(num_nodes)],
        cwd=binary_path.parent.parent.parent,
    )
    return json.loads(completed.stdout.strip())


def load_spark_labels(output_dir: Path) -> dict[int, int]:
    labels: dict[int, int] = {}
    if not output_dir.exists():
        raise RuntimeError(f"spark output directory not found: {output_dir}")
    for part_file in sorted(output_dir.glob("part-*")):
        with part_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                node_text, label_text = line.split("\t", 1)
                if label_text == "UNKNOWN":
                    continue
                labels[int(node_text)] = int(label_text)
    return labels


def load_seed_labels(graph_file: Path) -> dict[int, int]:
    seeds: dict[int, int] = {}
    with graph_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                seeds[int(parts[0])] = int(parts[2])
    return seeds


def load_int_vector(output_dir: Path, unknown_text: str | None = None, unknown_value: int | None = None) -> dict[int, int]:
    values: dict[int, int] = {}
    if not output_dir.exists():
        raise RuntimeError(f"spark output directory not found: {output_dir}")
    for part_file in sorted(output_dir.glob("part-*")):
        with part_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                node_text, value_text = line.split("\t", 1)
                if unknown_text is not None and value_text == unknown_text:
                    if unknown_value is None:
                        continue
                    values[int(node_text)] = unknown_value
                else:
                    values[int(node_text)] = int(value_text)
    return values


def load_float_vector(output_dir: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    if not output_dir.exists():
        raise RuntimeError(f"spark output directory not found: {output_dir}")
    for part_file in sorted(output_dir.glob("part-*")):
        with part_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                node_text, value_text = line.split("\t", 1)
                if value_text == "Infinity":
                    values[int(node_text)] = math.inf
                else:
                    values[int(node_text)] = float(value_text)
    return values


def normalize_partition_labels(labels: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    normalized: list[int] = []
    next_id = 0
    for label in labels:
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
        normalized.append(mapping[label])
    return normalized


def canonical_component_hash_from_labels(labels: list[int]) -> str:
    hash_value = 0xCBF29CE484222325
    for component in normalize_partition_labels(labels):
        hash_value ^= component
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def validate_bfs_spark_output(
    graph_file: Path,
    num_nodes: int,
    source_node: int,
    max_levels: int,
    spark_output_dir: Path,
) -> dict[str, Any]:
    standalone_output = run_bfs_standalone(graph_file, num_nodes, source_node, max_levels)
    standalone_levels = standalone_output.get("levels")
    if not isinstance(standalone_levels, list):
        raise RuntimeError("standalone BFS output does not contain levels")
    spark_levels_map = load_int_vector(spark_output_dir, unknown_text="UNVISITED", unknown_value=UNVISITED)
    spark_levels = [spark_levels_map.get(node, UNVISITED) for node in range(num_nodes)]
    mismatches = []
    for node, (spark_level, standalone_level) in enumerate(zip(spark_levels, standalone_levels)):
        if int(spark_level) != int(standalone_level):
            mismatches.append({"node": node, "spark": int(spark_level), "standalone": int(standalone_level)})
            if len(mismatches) >= 20:
                break
    return {
        "performed": True,
        "passed": len(mismatches) == 0,
        "num_nodes": num_nodes,
        "mismatches": len(mismatches),
        "sample_mismatches": mismatches,
    }


def validate_sssp_spark_output(
    graph_file: Path,
    num_nodes: int,
    source_node: int,
    spark_output_dir: Path,
) -> dict[str, Any]:
    standalone_output = run_sssp_standalone(graph_file, num_nodes, source_node)
    standalone_distances = standalone_output.get("distances")
    if not isinstance(standalone_distances, list):
        raise RuntimeError("standalone SSSP output does not contain distances")
    spark_distance_map = load_float_vector(spark_output_dir)
    mismatches = []
    for node in range(num_nodes):
        spark_distance = spark_distance_map.get(node, math.inf)
        standalone_distance_raw = standalone_distances[node]
        standalone_distance = math.inf if standalone_distance_raw is None else float(standalone_distance_raw)
        if spark_distance == standalone_distance:
            continue
        if isinstance(standalone_distance, (int, float)) and isinstance(spark_distance, (int, float)):
            if math.isinf(float(standalone_distance)) and math.isinf(float(spark_distance)):
                continue
            tolerance = max(abs(float(standalone_distance)) * 1e-4, 1e-3)
            if abs(float(standalone_distance) - float(spark_distance)) <= tolerance:
                continue
        mismatches.append(
            {
                "node": node,
                "spark": spark_distance,
                "standalone": standalone_distance,
            }
        )
        if len(mismatches) >= 20:
            break
    return {
        "performed": True,
        "passed": len(mismatches) == 0,
        "num_nodes": num_nodes,
        "mismatches": len(mismatches),
        "sample_mismatches": mismatches,
    }


def validate_wcc_spark_output(
    graph_file: Path,
    num_nodes: int,
    spark_output_dir: Path,
) -> dict[str, Any]:
    standalone_output = run_wcc_standalone(graph_file, num_nodes)
    standalone_parent = standalone_output.get("parent")
    if not isinstance(standalone_parent, list):
        raise RuntimeError("standalone Union-Find output does not contain parent")
    spark_component_map = load_int_vector(spark_output_dir)
    spark_components = [spark_component_map[node] for node in range(num_nodes)]
    standalone_hash = canonical_component_hash_from_labels([int(parent) for parent in standalone_parent])
    spark_hash = canonical_component_hash_from_labels([int(component) for component in spark_components])
    standalone_norm = normalize_partition_labels([int(parent) for parent in standalone_parent])
    spark_norm = normalize_partition_labels([int(component) for component in spark_components])
    mismatches = []
    for node, (spark_component, standalone_component) in enumerate(zip(spark_norm, standalone_norm)):
        if spark_component != standalone_component:
            mismatches.append(
                {"node": node, "spark": spark_component, "standalone": standalone_component}
            )
            if len(mismatches) >= 20:
                break
    return {
        "performed": True,
        "passed": standalone_hash == spark_hash and len(mismatches) == 0,
        "num_nodes": num_nodes,
        "standalone_component_hash": standalone_hash,
        "spark_component_hash": spark_hash,
        "mismatches": len(mismatches),
        "sample_mismatches": mismatches,
    }


def validate_lp_spark_output(
    graph_file: Path,
    num_nodes: int,
    max_iter: int,
    spark_output_dir: Path,
) -> dict[str, Any]:
    standalone_output = run_lp_standalone(graph_file, num_nodes, max_iter)
    standalone_labels = standalone_output.get("labels")
    if not isinstance(standalone_labels, list):
        raise RuntimeError("standalone LP output does not contain a label vector")

    spark_labels = load_spark_labels(spark_output_dir)
    seeds = load_seed_labels(graph_file)

    mismatches = 0
    sample_mismatches: list[dict[str, int]] = []
    seed_violations = 0
    sample_seed_violations: list[dict[str, int]] = []

    for node in range(num_nodes):
        standalone_label = int(standalone_labels[node])
        spark_label = spark_labels.get(node)
        if spark_label is None:
            mismatches += 1
            if len(sample_mismatches) < 20:
                sample_mismatches.append(
                    {"node": node, "spark": -1, "standalone": standalone_label}
                )
            continue

        if spark_label != standalone_label:
            mismatches += 1
            if len(sample_mismatches) < 20:
                sample_mismatches.append(
                    {"node": node, "spark": spark_label, "standalone": standalone_label}
                )

        if node in seeds and spark_label != seeds[node]:
            seed_violations += 1
            if len(sample_seed_violations) < 20:
                sample_seed_violations.append(
                    {"node": node, "expected": seeds[node], "spark": spark_label}
                )

    accuracy = ((num_nodes - mismatches) / num_nodes) * 100 if num_nodes else 0.0
    return {
        "performed": True,
        "passed": mismatches == 0 and seed_violations == 0,
        "num_nodes": num_nodes,
        "accuracy": round(accuracy, 6),
        "mismatches": mismatches,
        "seed_violations": seed_violations,
        "sample_mismatches": sample_mismatches,
        "sample_seed_violations": sample_seed_violations,
    }


def ensure_dataset(algorithm: SparkAlgorithm, size: int, force: bool) -> Path | None:
    if not algorithm.generator_repo or not algorithm.generator_script:
        return None

    dataset_path = DATA_DIR / algorithm.dataset_name_fn(size)
    if dataset_path.exists() and not force:
        return dataset_path

    python = repo_python(algorithm.generator_repo)
    script = ROOT / algorithm.generator_repo / algorithm.generator_script
    command = [str(python), str(script), *algorithm.generator_args_fn(size, dataset_path)]
    log(f"Generating dataset for {algorithm.slug} at {dataset_path.name}")
    run_command(command, cwd=script.parent)
    if not dataset_path.exists():
        raise RuntimeError(f"dataset generation did not produce {dataset_path}")
    return dataset_path


def benchmark_point(
    algorithm: SparkAlgorithm,
    size: int,
    runs: int,
) -> dict[str, Any]:
    dataset_path = DATA_DIR / algorithm.dataset_name_fn(size)
    container_input = f"/opt/tfm-spark/data/{dataset_path.name}"
    load_runs: list[float] = []
    exec_runs: list[float] = []
    total_runs: list[float] = []
    last_payload: dict[str, Any] | None = None
    validation_summary: dict[str, Any] | None = None

    for run_idx in range(runs):
        output_path = f"/opt/tfm-spark/results/{algorithm.slug}_{size}_run{run_idx + 1}"
        host_output_dir = RESULTS_DIR / f"{algorithm.slug}_{size}_run{run_idx + 1}"
        remove_output_dir(host_output_dir, output_path)
        command = ["bash", str(HERE / "scripts" / algorithm.submit_script), *algorithm.submit_args_fn(container_input, output_path, size)]
        log(f"Running Spark {algorithm.slug} size={size:,} run={run_idx + 1}/{runs}")
        env = os.environ.copy()
        for key, value in algorithm.configuration.get("spark_env", {}).items():
            env[key] = str(value)
        if algorithm.slug in {"bfs", "sssp", "labelpropagation", "wcc"} and run_idx == 0:
            env["SPARK_PERSIST_OUTPUT"] = "true"
        else:
            env["SPARK_PERSIST_OUTPUT"] = "false"
        completed = subprocess.run(
            command,
            cwd=HERE,
            text=True,
            capture_output=True,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "command failed with exit code "
                f"{completed.returncode}: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        payload = parse_result(completed.stdout)
        load_runs.append(float(payload["load_time_ms"]))
        exec_runs.append(float(payload["execution_time_ms"]))
        total_runs.append(float(payload["total_time_ms"]))
        last_payload = payload

        if run_idx == 0:
            if algorithm.slug == "labelpropagation":
                validation_summary = validate_lp_spark_output(
                    dataset_path,
                    size,
                    int(algorithm.configuration["max_iter"]),
                    host_output_dir,
                )
            elif algorithm.slug == "bfs":
                validation_summary = validate_bfs_spark_output(
                    dataset_path,
                    size,
                    int(algorithm.configuration["source_node"]),
                    int(algorithm.configuration["max_levels"]),
                    host_output_dir,
                )
            elif algorithm.slug == "sssp":
                validation_summary = validate_sssp_spark_output(
                    dataset_path,
                    size,
                    int(algorithm.configuration["source_node"]),
                    host_output_dir,
                )
            elif algorithm.slug == "wcc":
                validation_summary = validate_wcc_spark_output(
                    dataset_path,
                    size,
                    host_output_dir,
                )
            if validation_summary is not None and not validation_summary.get("passed", False):
                raise RuntimeError(
                    f"Spark validation failed for {algorithm.slug} at size {size}: {validation_summary}"
                )

    load_mean, load_std = mean_std(load_runs)
    exec_mean, exec_std = mean_std(exec_runs)
    total_mean, total_std = mean_std(total_runs)
    validated_load = float(load_runs[0]) if load_runs else None
    validated_exec = float(exec_runs[0]) if exec_runs else None
    validated_total = float(total_runs[0]) if total_runs else None
    unvalidated_load_mean, unvalidated_load_std = mean_std(load_runs[1:]) if len(load_runs) > 1 else (0.0, 0.0)
    unvalidated_exec_mean, unvalidated_exec_std = mean_std(exec_runs[1:]) if len(exec_runs) > 1 else (0.0, 0.0)
    unvalidated_total_mean, unvalidated_total_std = mean_std(total_runs[1:]) if len(total_runs) > 1 else (0.0, 0.0)
    row = {
        algorithm.x_key: size,
        "spark_load_ms": round(load_mean, 2),
        "spark_load_std_ms": round(load_std, 2),
        "spark_exec_ms": round(exec_mean, 2),
        "spark_exec_std_ms": round(exec_std, 2),
        "spark_total_ms": round(total_mean, 2),
        "spark_total_std_ms": round(total_std, 2),
        "spark_load_runs_ms": [round(value, 2) for value in load_runs],
        "spark_exec_runs_ms": [round(value, 2) for value in exec_runs],
        "spark_total_runs_ms": [round(value, 2) for value in total_runs],
        "spark_load_validated_run_ms": round(validated_load, 2) if validated_load is not None else None,
        "spark_exec_validated_run_ms": round(validated_exec, 2) if validated_exec is not None else None,
        "spark_total_validated_run_ms": round(validated_total, 2) if validated_total is not None else None,
        "spark_load_avg_unvalidated_ms": round(unvalidated_load_mean, 2) if len(load_runs) > 1 else None,
        "spark_load_std_unvalidated_ms": round(unvalidated_load_std, 2) if len(load_runs) > 1 else None,
        "spark_exec_avg_unvalidated_ms": round(unvalidated_exec_mean, 2) if len(exec_runs) > 1 else None,
        "spark_exec_std_unvalidated_ms": round(unvalidated_exec_std, 2) if len(exec_runs) > 1 else None,
        "spark_total_avg_unvalidated_ms": round(unvalidated_total_mean, 2) if len(total_runs) > 1 else None,
        "spark_total_std_unvalidated_ms": round(unvalidated_total_std, 2) if len(total_runs) > 1 else None,
    }
    if last_payload:
        for key in ("visited_nodes", "max_level", "reachable_nodes", "max_distance", "iterations", "converged", "labeled_nodes", "distinct_labels", "num_components", "component_hash"):
            if key in last_payload:
                row[key] = last_payload[key]
    if validation_summary is not None:
        row["validation"] = validation_summary
    return row


def result_path(algorithm: SparkAlgorithm) -> Path:
    return RESULTS_DIR / f"{algorithm.slug}.json"


def write_unsupported_result(algorithm: SparkAlgorithm) -> Path:
    payload = {
        "algorithm": algorithm.slug,
        "framework": "spark",
        "supported": False,
        "reason": algorithm.unsupported_reason,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = result_path(algorithm)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_algorithm(algorithm: SparkAlgorithm, runs: int, force: bool, force_data: bool) -> Path:
    out_path = result_path(algorithm)
    partial_path = checkpoint_path(algorithm)
    if out_path.exists() and not force:
        log(f"Skipping Spark {algorithm.slug}: result already present at {out_path}")
        return out_path

    if not algorithm.supported:
        log(f"Recording unsupported Spark baseline for {algorithm.slug}")
        return write_unsupported_result(algorithm)

    rows = []
    completed_points: set[int] = set()
    if partial_path.exists() and not force:
        partial_payload = json.loads(partial_path.read_text(encoding="utf-8"))
        rows = partial_payload.get("results", [])
        completed_points = {int(row[algorithm.x_key]) for row in rows if algorithm.x_key in row}
        if completed_points:
            log(f"Resuming Spark {algorithm.slug} from checkpoint: {sorted(completed_points)}")

    for size in algorithm.points:
        if size in completed_points:
            log(f"Skipping Spark {algorithm.slug} size={size:,} (checkpointed)")
            continue
        ensure_dataset(algorithm, size, force=force_data)
        rows.append(benchmark_point(algorithm, size, runs))
        partial_payload = {
            "algorithm": algorithm.slug,
            "framework": "spark",
            "supported": True,
            "timestamp": datetime.now().isoformat(),
            "runs_per_point": runs,
            "test_points": algorithm.points,
            "results": rows,
            "configuration": algorithm.configuration,
            "partial": True,
        }
        if "experimental_note" in algorithm.configuration:
            partial_payload["experimental_note"] = algorithm.configuration["experimental_note"]
        partial_path.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")

    payload = {
        "algorithm": algorithm.slug,
        "framework": "spark",
        "supported": True,
        "timestamp": datetime.now().isoformat(),
        "runs_per_point": runs,
        "test_points": algorithm.points,
        "results": rows,
        "configuration": algorithm.configuration,
    }
    if "experimental_note" in algorithm.configuration:
        payload["experimental_note"] = algorithm.configuration["experimental_note"]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if partial_path.exists():
        partial_path.unlink()
    log(f"Wrote Spark results for {algorithm.slug}: {out_path}")
    return out_path


def build_algorithms() -> dict[str, SparkAlgorithm]:
    return {
        "bfs": SparkAlgorithm(
            slug="bfs",
            x_key="nodes",
            points=[100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000],
            generator_repo="bfs",
            generator_script="setup_large_bfs_data.py",
            dataset_name_fn=lambda size: f"large_bfs_{size}.txt",
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", "4",
                "--density", "10",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_script="submit-bfs.sh",
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "0",
                "500",
                "4",
            ],
            configuration={
                "partitions": 4,
                "source_node": 0,
                "max_levels": 500,
                "density": 10,
                "spark_env": {
                    "SPARK_TOTAL_EXECUTOR_CORES": "4",
                    "SPARK_EXECUTOR_CORES": "1",
                    "SPARK_EXECUTOR_MEMORY": "4g",
                    "SPARK_DEFAULT_PARALLELISM": "4",
                    "SPARK_SHUFFLE_PARTITIONS": "4",
                },
            },
        ),
        "sssp": SparkAlgorithm(
            slug="sssp",
            x_key="nodes",
            points=[100_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000],
            generator_repo="sssp",
            generator_script="setup_large_sssp_data.py",
            dataset_name_fn=lambda size: f"large_sssp_{size}.txt",
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", "4",
                "--density", "10",
                "--max-weight", "10.0",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_script="submit-sssp.sh",
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "0",
                "4",
                "500",
            ],
            configuration={
                "partitions": 4,
                "source_node": 0,
                "max_iter": 500,
                "density": 10,
                "max_weight": 10.0,
                "spark_env": {
                    "SPARK_TOTAL_EXECUTOR_CORES": "4",
                    "SPARK_EXECUTOR_CORES": "1",
                    "SPARK_EXECUTOR_MEMORY": "4g",
                    "SPARK_DEFAULT_PARALLELISM": "4",
                    "SPARK_SHUFFLE_PARTITIONS": "4",
                },
            },
        ),
        "labelpropagation": SparkAlgorithm(
            slug="labelpropagation",
            x_key="nodes",
            points=[100_000, 500_000, 1_000_000, 2_000_000],
            generator_repo="labelpropagation",
            generator_script="setup_large_lp_data.py",
            dataset_name_fn=lambda size: f"large_lp_{size}.txt",
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", "4",
                "--density", "20",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_script="submit-lp.sh",
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "10",
                "4",
            ],
            configuration={
                "partitions": 4,
                "max_iter": 10,
                "density": 20,
                "experimental_note": (
                    "La baseline Spark para Label Propagation se truncó en 2M nodos "
                    "porque a partir de 3M el coste por repetición superó el presupuesto "
                    "experimental razonable de la campaña."
                ),
                "spark_env": {
                    "SPARK_TOTAL_EXECUTOR_CORES": "4",
                    "SPARK_EXECUTOR_CORES": "1",
                    "SPARK_EXECUTOR_MEMORY": "4g",
                    "SPARK_DEFAULT_PARALLELISM": "4",
                    "SPARK_SHUFFLE_PARTITIONS": "4",
                },
            },
        ),
        "wcc": connected_components_algorithm("wcc"),
        "louvain": SparkAlgorithm(
            slug="louvain",
            x_key="nodes",
            points=[],
            generator_repo=None,
            generator_script=None,
            dataset_name_fn=lambda size: "",
            generator_args_fn=lambda size, dataset: [],
            submit_script=None,
            submit_args_fn=lambda container_input, output_path, size: [],
            configuration={},
            supported=False,
            unsupported_reason=(
                "No se ha integrado una implementacion Louvain en Spark que sea semantica y "
                "metodologicamente equivalente a la usada en standalone/burst; el ciclo Spark "
                "la registra como no disponible para evitar una comparacion injusta."
            ),
        ),
    }


def parse_points_arg(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def with_resource_overrides(
    algorithm: SparkAlgorithm,
    *,
    partitions: int,
    executors: int,
    executor_memory: str,
    points: list[int] | None,
) -> SparkAlgorithm:
    if not algorithm.supported:
        return algorithm

    config = json.loads(json.dumps(algorithm.configuration))
    config["partitions"] = partitions
    spark_env = config.setdefault("spark_env", {})
    spark_env["SPARK_TOTAL_EXECUTOR_CORES"] = str(executors)
    spark_env["SPARK_EXECUTOR_CORES"] = "1"
    spark_env["SPARK_EXECUTOR_MEMORY"] = executor_memory
    spark_env["SPARK_DEFAULT_PARALLELISM"] = str(partitions)
    spark_env["SPARK_SHUFFLE_PARTITIONS"] = str(partitions)

    if algorithm.slug == "bfs":
        return replace(
            algorithm,
            points=points or algorithm.points,
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", str(partitions),
                "--density", "10",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "0",
                "500",
                str(partitions),
            ],
            configuration=config,
        )

    if algorithm.slug == "sssp":
        return replace(
            algorithm,
            points=points or algorithm.points,
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", str(partitions),
                "--density", "10",
                "--max-weight", "10.0",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "0",
                str(partitions),
                "500",
            ],
            configuration=config,
        )

    if algorithm.slug == "labelpropagation":
        return replace(
            algorithm,
            points=points or algorithm.points,
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--partitions", str(partitions),
                "--density", "20",
                "--output", str(dataset),
                "--no-s3",
            ],
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                "10",
                str(partitions),
            ],
            configuration=config,
        )

    if algorithm.slug == "wcc":
        return replace(
            algorithm,
            points=points or algorithm.points,
            generator_args_fn=lambda size, dataset: [
                "--nodes", str(size),
                "--edges-per-node", "5",
                "--components", "10",
                "--partitions", str(partitions),
                "--output", str(dataset),
                "--key", f"wcc-graphs/wcc-{size}",
                "--no-s3",
            ],
            submit_args_fn=lambda container_input, output_path, _size: [
                container_input,
                output_path,
                str(partitions),
            ],
            configuration=config,
        )

    return replace(algorithm, points=points or algorithm.points, configuration=config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spark baselines for the graph algorithms used in the TFM.")
    parser.add_argument(
        "--algorithms",
        default="bfs,sssp,labelpropagation,wcc,louvain",
        help="Comma-separated Spark baseline steps to run.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per point.")
    parser.add_argument("--force", action="store_true", help="Re-run even if the JSON already exists.")
    parser.add_argument("--force-data", action="store_true", help="Re-generate local datasets even if they exist.")
    parser.add_argument("--partitions", type=int, default=4, help="Dataset partitions and Spark parallelism.")
    parser.add_argument("--executors", type=int, default=4, help="Spark executors with 1 core each.")
    parser.add_argument("--executor-memory", default="4g", help="Memory per Spark executor.")
    parser.add_argument("--points", default="", help="Comma-separated override for graph sizes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    algorithms = build_algorithms()
    override_points = parse_points_arg(args.points)
    requested = [token.strip() for token in args.algorithms.split(",") if token.strip()]
    deduped: list[str] = []
    for name in requested:
        if name not in deduped:
            deduped.append(name)
    requested = deduped
    unknown = [name for name in requested if name not in algorithms]
    if unknown:
        raise SystemExit(f"Unknown Spark algorithms: {', '.join(unknown)}")

    for name in requested:
        algorithm = with_resource_overrides(
            algorithms[name],
            partitions=args.partitions,
            executors=args.executors,
            executor_memory=args.executor_memory,
            points=override_points,
        )
        run_algorithm(algorithm, runs=args.runs, force=args.force, force_data=args.force_data)


if __name__ == "__main__":
    main()
