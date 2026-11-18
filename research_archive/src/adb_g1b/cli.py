"""adb-g1b console entry point (R1-04).

Usage:
    adb-g1b --experiment full --out-dir results/g1b
    adb-g1b --experiment pilot
    adb-g1b --help
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="adb-g1b",
        description="Official classical baselines benchmark (G1-B).")
    parser.add_argument("--experiment", choices=["pilot", "full"], default="full",
                        help="pilot = small budget sanity grid; full = 20-rep grid")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default results/g1b)")
    args = parser.parse_args(argv)
    from adb_g1b.bench import run
    out = run(args.experiment, out_dir=Path(args.out_dir) if args.out_dir else None)
    print("OUTPUT_DIR =", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())