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


@dataclass(frozen=True)
class CampaignStep:
    name: str
    workdir: Path
    command: list[str]
    result_file: Path | None = None


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def repo_python(repo_name: str, validation_python: Path) -> Path:
    candidate = ROOT / repo_name / ".venv/bin/python"
    if candidate.exists():
        return candidate
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
        "reports": CampaignStep(
            name="reports",
            workdir=ROOT / "labelpropagation",
            command=[str(repo_python("labelpropagation", validation_python)), "generate_crossover_reports.py"],
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

    log(f"▶ Running {step.name}")
    log(f"   cwd: {step.workdir}")
    log(f"   cmd: {' '.join(step.command)}")

    completed = subprocess.run(
        step.command,
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
        default="labelpropagation,gradientboosting,unionfind,reports",
        help=(
            "Comma-separated steps to run. "
            "Available: bfs,sssp,labelpropagation,gradientboosting,"
            "collaborativefiltering,unionfind,reports"
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
