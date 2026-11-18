"""Reproducible CLI for the G1-A exact benchmark.

Commands (all deterministic, CPU only):
  adb-g1a run-all [--out-dir DIR]   compute all five scenarios and write
                                    machine-readable CSV/JSON + comparison
  adb-g1a table                     print the official 5x5 comparison table
  adb-g1a metadata                  print environment, deps, source hashes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

from . import __version__
from .model import SCENARIOS, compute_all_scenarios
from .reference import (METRIC_SPECS, OFFICIAL_REFERENCE_TABLE_5,
                        REFERENCE_SOURCE_ID, compare_all)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Some integration checks (official source provenance) depend on the original
# ADB Challenge Brief PDF and prior planning documents, which are deliberately
# NOT redistributed. A reviewer must supply them at the documented paths; if
# they are absent the hash record reports "MISSING" explicitly and never fakes
# a SOURCE_VERIFIED status.
EXTERNAL_OFFICIAL_SOURCE_REQUIRED = "YES"

_SOURCE_FILES = [
    "resources/official/ADB Quantum Innovation Challenge Brief.pdf",
    "resources/planning/ADB_G0_Requirement_Benchmark_Plan_2026-11-04.md",
    "resources/planning/ADB_Quantum_2026_Phase1_Form_Requirements_2026-11-04.md",
    "resources/planning/ADB_Quantum_2026_Dev_Window_Starter_2026-11-04.md",
]


def source_hashes(root: Path | None = None) -> dict[str, str]:
    """SHA256 of the four official/planning sources below a project root.

    A missing file is reported as the literal marker ``"MISSING"``; the record
    never substitutes a fabricated hash, so an absent official source is
    explicitly visible instead of silently marked verified.
    """
    base = Path(root) if root is not None else PROJECT_ROOT
    out = {}
    for rel in _SOURCE_FILES:
        p = base / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"
    return out


def _fmt_usd(v: float) -> str:
    return f"{v:,.2f}"


def _fmt_pct(v: float) -> str:
    return f"{v*100:.3f}"


def build_result_rows(computed) -> list[dict]:
    return compare_all(computed)


def environment_metadata() -> dict:
    from importlib.metadata import version as _pkg_version
    try:
        cpu_model = Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0].strip() if Path("/proc/cpuinfo").exists() else "n/a"
    except (IndexError, OSError):
        cpu_model = "n/a"
    import numpy
    import scipy
    import sys

    def _v(name):
        try:
            return _pkg_version(name)
        except Exception:
            return "n/a"

    return {
        "project": "adb-quantum-joint-loss-2026",
        "gate": "G1_A_OFFICIAL_EXACT_BENCHMARK_REPRODUCTION",
        "date": "2026-11-05",
        "package_version": __version__,
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu_model": cpu_model,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pytest": _v("pytest"),
        "external_official_source_required": EXTERNAL_OFFICIAL_SOURCE_REQUIRED,
        "source_hashes": source_hashes(),
    }


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _content_sha256(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_results(computed, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_result_rows(computed)

    # --- long 25-row machine-readable CSV ---
    long_fields = ["scenario", "metric", "computed_full_precision",
                   "published_reference", "displayed_unit", "absolute_error",
                   "official_rounding_tolerance", "within_tolerance", "source"]
    with (out_dir / "exact_ref_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=long_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in long_fields})

    # --- human-oriented comparison table (text) ---
    lines = ["%36s %20s %20s %14s %12s" % ("scenario", "computed", "published",
                                           "abs_error", "within")]
    for r in rows:
        c, p = r["computed_full_precision"], r["published_reference"]
        disp = r["displayed_unit"]
        if disp == "USD million":
            cs, ps = f"{c/1e6:.6f}", f"{p/1e6:.6f}"
        elif disp == "percentage point":
            cs, ps = f"{c*100:.9f}", f"{p*100:.9f}"
        else:
            cs, ps = f"{c:.6f}", f"{p:.6f}"
        lines.append("%36s %20s %20s %14.6g %12s" % (
            f"{r['scenario']}/{r['metric']}", cs, ps,
            r["absolute_error"], "OK" if r["within_tolerance"] else "OUT"))
    (out_dir / "exact_ref_comparison.txt").write_text("\n".join(lines) + "\n")

    # --- JSON: full precision + metadata + official row ---
    payload = {
        "gate": "G1_A_OFFICIAL_EXACT_BENCHMARK_REPRODUCTION",
        "computed_full_precision": computed,
        "computed_sha256": _content_sha256(computed),
        "official_reference_table_5": OFFICIAL_REFERENCE_TABLE_5,
        "comparison": rows,
        "tolerance_specs": {k: v for k, v in METRIC_SPECS.items()},
        "reference_source": REFERENCE_SOURCE_ID,
        "environment": environment_metadata(),
    }
    (out_dir / "exact_ref_results.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    return payload


def print_table(computed) -> None:
    hdr = ["scenario", "E[L] USDm", "joint %", "exceed %", "stoploss USD",
           "both-exh %"]
    widths = [14, 12, 9, 9, 13, 10]
    print("   ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    ref = OFFICIAL_REFERENCE_TABLE_5
    for sc in SCENARIOS:
        c = computed[sc.key]
        r = ref[sc.key]
        row = [
            sc.key,
            f'{c["term_expected_principal_loss_usd"]/1e6:.4f}',
            _fmt_pct(c["term_joint_attachment_probability"]),
            _fmt_pct(c["term_portfolio_loss_exceedance_probability"]),
            _fmt_usd(c["term_portfolio_stop_loss_usd"]),
            f'{c["term_both_bonds_exhausted_probability"]*100:.4f}',
        ]
        print("   ".join(x.ljust(w) for x, w in zip(row, widths)))
    print("\npublished (official Table 5)")
    for sc in SCENARIOS:
        r = ref[sc.key]
        row = [
            sc.key,
            f"{r['term_expected_principal_loss_usd']/1e6:.2f}",
            f"{r['term_joint_attachment_probability']*100:.3f}",
            f"{r['term_portfolio_loss_exceedance_probability']*100:.3f}",
            _fmt_usd(r["term_portfolio_stop_loss_usd"]),
            f"{r['term_both_bonds_exhausted_probability']*100:.4f}",
        ]
        print("   ".join(x.ljust(w) for x, w in zip(row, widths)))


def run_all(args) -> int:
    start = time.perf_counter()
    computed = compute_all_scenarios()
    out_dir = Path(args.out_dir or "results/g1a_exact")
    payload = write_results(computed, out_dir)
    wall = time.perf_counter() - start
    rss = rss_kb()
    # runtime measurements are per-run and therefore kept out of the
    # deterministic scientific payload
    (out_dir / "runtime_measurement.json").write_text(
        json.dumps({
            "wall_time_seconds": wall,
            "peak_rss_kb": rss,
            "computed_sha256": payload["computed_sha256"],
        }, indent=2) + "\n")
    print_table(computed)
    print(f"\nRESULTS_DIR = {out_dir}")
    print(f"WALL_TIME_S = {wall:.3f}")
    print(f"PEAK_RSS_KB = {rss}")
    ok = all(r["within_tolerance"] for r in payload["comparison"])
    print(f"WITHIN_TOLERANCE = {sum(r['within_tolerance'] for r in payload['comparison'])}/25")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="adb-g1a")
    sub = parser.add_subparsers(dest="command")
    p_run = sub.add_parser("run-all", help="compute all scenarios and write results")
    p_run.add_argument("--out-dir", default=None)
    sub.add_parser("table", help="print official 5x5 comparison")
    sub.add_parser("metadata", help="print environment metadata")
    args = parser.parse_args(argv)
    if args.command == "table":
        print_table(compute_all_scenarios())
        return 0
    if args.command == "metadata":
        for k, v in environment_metadata().items():
            print(f"{k} = {v}")
        return 0
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())