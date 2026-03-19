#!/usr/bin/env python3
"""Run only the burst implementation for Label Propagation."""
import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "benchmark_lp.py", "--skip-standalone", *sys.argv[1:]]))