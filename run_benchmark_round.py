#!/usr/bin/env python3
"""Run one full benchmark round comparing standalone vs burst for Label Propagation."""
import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "validate_crossover.py", *sys.argv[1:]]))