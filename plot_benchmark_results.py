#!/usr/bin/env python3
"""Generate standard Label Propagation benchmark plots from benchmark round results."""
import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "plot_new_results.py", *sys.argv[1:]]))