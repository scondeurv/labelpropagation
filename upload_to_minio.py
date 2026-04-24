#!/usr/bin/env python3
import os
from pathlib import Path
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / "data_utils")
    runpy.run_path(str(root / "data_utils" / "upload_to_minio.py"), run_name="__main__")


if __name__ == "__main__":
    main()
