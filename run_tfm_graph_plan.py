#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from spark_baseline.run_spark_graph_benchmarks import build_algorithms, ensure_dirs, run_algorithm


ROOT = Path("/home/sergio/src")
PLAN_ALGORITHMS = ("bfs", "sssp", "labelpropagation", "unionfind")
PLAN_POINTS = [100_000, 500_000, 1_000_000, 2_000_000]
SMOKE_POINTS = [100_000]
BURST_SCRIPTS = {
    "bfs": ("bfs", "validate_crossover_bfs.py"),
    "sssp": ("sssp", "validate_crossover_sssp.py"),
    "labelpropagation": ("labelpropagation", "validate_crossover.py"),
    "unionfind": ("unionfind", "validate_crossover.py"),
}
RESET_GLOBS = {
    "bfs": (
        "large_bfs_*.txt",
        "crossover_bfs_*.json",
        "*_analysis.png",
        "runlogs/*",
    ),
    "sssp": (
        "large_sssp_*.txt",
        "crossover_sssp_*.json",
        "*_analysis.png",
        "runlogs/*",
        "exploratory_logs_20260320/*",
    ),
    "labelpropagation": (
        "large_*.txt",
        "crossover_*.json",
        "validation_report.json",
        "validation_run.log",
        "validation_*.log",
        "benchmark_lp_*.log",
        "benchmark_reports/docs/*",
        "benchmark_reports/figures/*",
        "benchmark_reports/raw/*",
        "crossover_data.json",
    ),
    "unionfind": (
        "uf_graph_*.tsv",
        "uf_crossover_*.json",
    ),
}
SPARK_RESET_DIRS = (
    ROOT / "labelpropagation/spark_baseline/data",
    ROOT / "labelpropagation/spark_baseline/results",
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def repo_python(repo_name: str, fallback: Path) -> Path:
    candidates = [
        ROOT / repo_name / ".venv/bin/python",
        ROOT / "labelpropagation/.venv/bin/python",
        ROOT / "bfs/.venv/bin/python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact graph campaign phases from PLAN_REINICIO_GRAFOS_TFM.md.")
    parser.add_argument(
        "--phase",
        choices=("reset", "verify-reset", "smoke-burst", "smoke-spark", "burst", "spark"),
        required=True,
        help="Campaign phase to execute.",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(PLAN_ALGORITHMS),
        help="Comma-separated subset of bfs,sssp,labelpropagation,unionfind.",
    )
    parser.add_argument(
        "--validation-python",
        default=str(repo_python("bfs", Path(sys.executable))),
        help="Python interpreter used for burst validators and generators.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run Spark outputs even if the final JSON exists.")
    parser.add_argument("--force-data", action="store_true", help="Re-generate Spark local datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Show reset actions without deleting files.")
    return parser.parse_args()


def selected_algorithms(raw: str) -> list[str]:
    requested = [token.strip() for token in raw.split(",") if token.strip()]
    unknown = [name for name in requested if name not in PLAN_ALGORITHMS]
    if unknown:
        raise SystemExit(f"Unknown algorithms: {', '.join(unknown)}")
    return requested


def run_burst_phase(algorithms: list[str], validation_python: Path, *, smoke: bool) -> None:
    points = SMOKE_POINTS if smoke else PLAN_POINTS
    runs = 1 if smoke else 3
    env = os.environ.copy()
    env["VALIDATION_PYTHON"] = str(validation_python)
    env["TFM_TEST_POINTS"] = ",".join(str(point) for point in points)
    env["TFM_RUNS"] = str(runs)

    log(f"Burst phase points={points} runs={runs}")
    for algorithm in algorithms:
        repo_name, script = BURST_SCRIPTS[algorithm]
        workdir = ROOT / repo_name
        command = [str(repo_python(repo_name, validation_python)), script]
        log(f"Running burst {algorithm}: {' '.join(command)}")
        completed = subprocess.run(command, cwd=workdir, env=env, text=True)
        if completed.returncode != 0:
            raise SystemExit(f"Burst phase failed for {algorithm} with exit code {completed.returncode}")


def run_spark_phase(algorithms: list[str], *, smoke: bool, force: bool, force_data: bool) -> None:
    points = SMOKE_POINTS if smoke else PLAN_POINTS
    runs = 1 if smoke else 3
    ensure_dirs()
    spark_algorithms = build_algorithms()

    log(f"Spark phase points={points} runs={runs} force={force} force_data={force_data}")
    for name in algorithms:
        algorithm = replace(spark_algorithms[name], points=points)
        log(f"Running spark {name} with points={algorithm.points}")
        run_algorithm(algorithm, runs=runs, force=force, force_data=force_data)


def remove_path(path: Path, *, dry_run: bool, removed: list[Path], blocked: list[Path]) -> None:
    if not path.exists():
        return
    if dry_run:
        removed.append(path)
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    except PermissionError:
        blocked.append(path)


def reset_repo_artifacts(repo_name: str, *, dry_run: bool, removed: list[Path], blocked: list[Path]) -> None:
    repo_root = ROOT / repo_name
    for pattern in RESET_GLOBS[repo_name]:
        for path in sorted(repo_root.glob(pattern)):
            remove_path(path, dry_run=dry_run, removed=removed, blocked=blocked)


def reset_directory_contents(directory: Path, *, dry_run: bool, removed: list[Path], blocked: list[Path]) -> None:
    if not directory.exists():
        return
    for path in sorted(directory.iterdir()):
        if path.name == ".gitkeep":
            continue
        remove_path(path, dry_run=dry_run, removed=removed, blocked=blocked)


def run_reset(*, dry_run: bool) -> None:
    removed: list[Path] = []
    blocked: list[Path] = []
    for repo_name in PLAN_ALGORITHMS:
        reset_repo_artifacts(repo_name, dry_run=dry_run, removed=removed, blocked=blocked)
    for directory in SPARK_RESET_DIRS:
        reset_directory_contents(directory, dry_run=dry_run, removed=removed, blocked=blocked)

    mode = "would remove" if dry_run else "removed"
    if not removed and not blocked:
        log(f"Reset phase found nothing to clean ({mode}: 0 paths).")
        return
    if removed:
        log(f"Reset phase {mode} {len(removed)} paths:")
        for path in removed:
            log(f"  - {path}")
    if blocked:
        log(f"Reset phase could not remove {len(blocked)} paths due to permissions:")
        for path in blocked:
            log(f"  - {path}")


def collect_reset_leftovers() -> list[Path]:
    leftovers: list[Path] = []
    for repo_name in PLAN_ALGORITHMS:
        repo_root = ROOT / repo_name
        for pattern in RESET_GLOBS[repo_name]:
            leftovers.extend(sorted(repo_root.glob(pattern)))
    for directory in SPARK_RESET_DIRS:
        if not directory.exists():
            continue
        leftovers.extend(sorted(path for path in directory.iterdir() if path.name != ".gitkeep"))
    return leftovers


def verify_reset() -> None:
    leftovers = collect_reset_leftovers()
    spark_ok = all(
        not directory.exists() or sorted(path.name for path in directory.iterdir()) in ([], [".gitkeep"])
        for directory in SPARK_RESET_DIRS
    )
    docker_running = False
    docker_proc = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        text=True,
        capture_output=True,
    )
    if docker_proc.returncode == 0:
        docker_running = any(line.strip().startswith("spark-") or line.strip().startswith("spark_") for line in docker_proc.stdout.splitlines())

    if leftovers:
        log("Reset verification found remaining artifacts:")
        for path in leftovers:
            log(f"  - {path}")
        raise SystemExit(1)
    if not spark_ok:
        raise SystemExit("Spark reset verification failed: data/results are not empty.")
    if docker_running:
        raise SystemExit("Reset verification failed: Spark containers are still running.")
    log("Reset verification passed: no generated campaign artifacts remain and Spark is stopped.")


def main() -> None:
    args = parse_args()
    algorithms = selected_algorithms(args.algorithms)
    validation_python = Path(args.validation_python)

    if args.phase == "reset":
        run_reset(dry_run=args.dry_run)
        return
    if args.phase == "verify-reset":
        verify_reset()
        return
    if args.phase == "smoke-burst":
        run_burst_phase(algorithms, validation_python, smoke=True)
        return
    if args.phase == "smoke-spark":
        run_spark_phase(algorithms, smoke=True, force=args.force, force_data=args.force_data)
        return
    if args.phase == "burst":
        run_burst_phase(algorithms, validation_python, smoke=False)
        return
    if args.phase == "spark":
        run_spark_phase(algorithms, smoke=False, force=args.force, force_data=args.force_data)
        return

    raise SystemExit(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    main()
