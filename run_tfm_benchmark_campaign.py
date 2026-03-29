#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/sergio/src")
DEFAULT_VALIDATION_PYTHON = Path("/home/sergio/src/bfs/.venv/bin/python")
DEFAULT_GRAPH_STEPS = (
    "bfs,spark-bfs,"
    "sssp,spark-sssp,"
    "labelpropagation,spark-labelpropagation,"
    "unionfind,spark-unionfind,"
    "reports"
)
DEFAULT_REPORT_SLUGS = "bfs,sssp,labelpropagation,unionfind"


@dataclass(frozen=True)
class CampaignStep:
    name: str
    workdir: Path
    command: list[str]
    result_file: Path | None = None


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def repo_python(repo_name: str, validation_python: Path) -> Path:
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
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0:
                return candidate
        except Exception:
            continue
    return validation_python


def build_steps(validation_python: Path) -> dict[str, CampaignStep]:
    return {
        "bfs": CampaignStep(
            name="bfs",
            workdir=ROOT / "bfs",
            command=[str(repo_python("bfs", validation_python)), "validate_crossover_bfs.py"],
            result_file=ROOT / "bfs/crossover_bfs_results.json",
        ),
        "sssp": CampaignStep(
            name="sssp",
            workdir=ROOT / "sssp",
            command=[str(repo_python("sssp", validation_python)), "validate_crossover_sssp.py"],
            result_file=ROOT / "sssp/crossover_sssp_results.json",
        ),
        "labelpropagation": CampaignStep(
            name="labelpropagation",
            workdir=ROOT / "labelpropagation",
            command=[str(repo_python("labelpropagation", validation_python)), "validate_crossover.py"],
            result_file=ROOT / "labelpropagation/crossover_validation_results.json",
        ),
        "louvain": CampaignStep(
            name="louvain",
            workdir=ROOT / "louvain",
            command=[str(repo_python("louvain", validation_python)), "validate_crossover_louvain.py"],
            result_file=ROOT / "louvain/crossover_louvain_results.json",
        ),
        "gradientboosting": CampaignStep(
            name="gradientboosting",
            workdir=ROOT / "gradientboosting",
            command=[str(repo_python("gradientboosting", validation_python)), "validate_crossover_gb.py"],
            result_file=ROOT / "gradientboosting/crossover_gb_results.json",
        ),
        "collaborativefiltering": CampaignStep(
            name="collaborativefiltering",
            workdir=ROOT / "collaborativefiltering",
            command=[str(repo_python("collaborativefiltering", validation_python)), "validate_crossover_cf.py"],
            result_file=ROOT / "collaborativefiltering/crossover_cf_results.json",
        ),
        "unionfind": CampaignStep(
            name="unionfind",
            workdir=ROOT / "unionfind",
            command=[str(repo_python("unionfind", validation_python)), "validate_crossover.py"],
            result_file=ROOT / "unionfind/uf_crossover_validation_results.json",
        ),
        "spark-bfs": CampaignStep(
            name="spark-bfs",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "spark_baseline/run_spark_graph_benchmarks.py",
                "--algorithms",
                "bfs",
            ],
            result_file=ROOT / "labelpropagation/spark_baseline/results/bfs.json",
        ),
        "spark-sssp": CampaignStep(
            name="spark-sssp",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "spark_baseline/run_spark_graph_benchmarks.py",
                "--algorithms",
                "sssp",
            ],
            result_file=ROOT / "labelpropagation/spark_baseline/results/sssp.json",
        ),
        "spark-labelpropagation": CampaignStep(
            name="spark-labelpropagation",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "spark_baseline/run_spark_graph_benchmarks.py",
                "--algorithms",
                "labelpropagation",
            ],
            result_file=ROOT / "labelpropagation/spark_baseline/results/labelpropagation.json",
        ),
        "spark-unionfind": CampaignStep(
            name="spark-unionfind",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "spark_baseline/run_spark_graph_benchmarks.py",
                "--algorithms",
                "unionfind",
            ],
            result_file=ROOT / "labelpropagation/spark_baseline/results/unionfind.json",
        ),
        "spark-louvain": CampaignStep(
            name="spark-louvain",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "spark_baseline/run_spark_graph_benchmarks.py",
                "--algorithms",
                "louvain",
            ],
            result_file=ROOT / "labelpropagation/spark_baseline/results/louvain.json",
        ),
        "reports": CampaignStep(
            name="reports",
            workdir=ROOT / "labelpropagation",
            command=[
                str(repo_python("labelpropagation", validation_python)),
                "generate_crossover_reports.py",
                "--slugs",
                DEFAULT_REPORT_SLUGS,
            ],
            result_file=ROOT / "labelpropagation/benchmark_reports/raw/index.json",
        ),
    }


def run_step(step: CampaignStep, validation_python: Path, force: bool) -> None:
    if step.result_file and step.result_file.exists() and not force:
        log(f"⏭️  Skipping {step.name}: result already present at {step.result_file}")
        return

    env = os.environ.copy()
    env["VALIDATION_PYTHON"] = str(validation_python)
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-benchmark-campaign")
    command = list(step.command)
    if force and step.name.startswith("spark-") and "--force" not in command:
        command.append("--force")

    log(f"▶ Running {step.name}")
    log(f"   cwd: {step.workdir}")
    log(f"   cmd: {' '.join(command)}")

    completed = subprocess.run(
        command,
        cwd=step.workdir,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{step.name} failed with exit code {completed.returncode}")

    if step.result_file:
        if not step.result_file.exists():
            raise SystemExit(f"{step.name} completed but did not produce {step.result_file}")
        log(f"✅ {step.name} wrote {step.result_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the serious benchmark campaign for the TFM and regenerate reports."
    )
    parser.add_argument(
        "--steps",
        default=DEFAULT_GRAPH_STEPS,
        help=(
            "Comma-separated steps to run. "
            "Available: bfs,sssp,labelpropagation,louvain,gradientboosting,"
            "collaborativefiltering,unionfind,spark-bfs,spark-sssp,"
            "spark-labelpropagation,spark-unionfind,spark-louvain,reports"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run a step even if its result file already exists.",
    )
    parser.add_argument(
        "--validation-python",
        default=str(DEFAULT_VALIDATION_PYTHON if DEFAULT_VALIDATION_PYTHON.exists() else sys.executable),
        help="Python interpreter used for validation helpers and dataset generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_python = Path(args.validation_python)
    steps_by_name = build_steps(validation_python)

    requested_steps = [token.strip() for token in args.steps.split(",") if token.strip()]
    unknown = [step for step in requested_steps if step not in steps_by_name]
    if unknown:
        raise SystemExit(f"Unknown steps: {', '.join(unknown)}")

    log("TFM benchmark campaign starting")
    log(f"Requested steps: {', '.join(requested_steps)}")
    log(f"Validation python: {validation_python}")

    for step_name in requested_steps:
        run_step(steps_by_name[step_name], validation_python, force=args.force)

    log("✅ TFM benchmark campaign finished")


if __name__ == "__main__":
    main()
