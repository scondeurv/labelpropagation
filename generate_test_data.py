#!/usr/bin/env python3
"""Generate local benchmark test data (without S3 upload)."""
import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "setup_large_lp_data.py", "--no-s3", *sys.argv[1:]]))