from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from adb_g1a.model import compute_all_scenarios
from adb_g1a.reference import compare_all

from .reference import load_reference

USD_METRICS = {
    "term_expected_principal_loss_usd",
    "term_portfolio_stop_loss_usd",
}
PROBABILITY_METRICS = {
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_both_bonds_exhausted_probability",
}


def verify_v1_exact(candidate_root: Path | None = None) -> dict:
    start = time.perf_counter()
    root = candidate_root or Path(__file__).resolve().parents[2]
    computed = compute_all_scenarios()
    rounded_rows = compare_all(computed)
    rounded_pass = sum(1 for row in rounded_rows if row["within_tolerance"])
    reference = load_reference(root)
    exact_rows = []
    exact_pass = 0
    for scenario in reference["public_scenarios"]:
        if scenario["track"] != "adb_drb":
            continue
        values = computed[scenario["scenario"]]
        for metric in scenario["metrics"]:
            name = metric["name"]
            expected = float(metric["exact_reference"])
            actual = float(values[name])
            diff = abs(actual - expected)
            tolerance = 1e-6 if name in USD_METRICS else 1e-12
            passed = diff <= tolerance
            exact_pass += int(passed)
            exact_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "metric": name,
                    "computed": actual,
                    "exact_reference": expected,
                    "abs_error": diff,
                    "diagnostic_tolerance": tolerance,
                    "within_diagnostic_tolerance": passed,
                }
            )
    status = "PASS" if rounded_pass == 25 and exact_pass == 25 else "FAIL"
    return {
        "status": status,
        "model": "frozen local V1 exact-DP implementation; not an official source-interface execution",
        "rounding_reference_pass": rounded_pass,
        "rounding_reference_total": 25,
        "exact_reference_pass": exact_pass,
        "exact_reference_total": 25,
        "max_exact_usd_abs_error": max(
            row["abs_error"] for row in exact_rows if row["metric"] in USD_METRICS
        ),
        "max_exact_probability_abs_error": max(
            row["abs_error"] for row in exact_rows if row["metric"] in PROBABILITY_METRICS
        ),
        "exact_rows": exact_rows,
        "resource": {
            "wall_s": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_status": "RSS_MONITOR_ONLY",
        },
    }
