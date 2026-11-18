from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from adb_v2.environment import environment
from adb_v2.p1b_evidence import verify_p1b
from adb_v2.official_source import official_source_audit
from adb_v2.reference import load_reference, validate_reference_ledger
from adb_v2.v1_reference import verify_v1_exact


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the V2-Enhanced candidate against explicit official source.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    results = CANDIDATE_ROOT / "results"
    reference = load_reference(CANDIDATE_ROOT)
    ledger = validate_reference_ledger(reference)
    source_audit = official_source_audit(args.source_root, args.archive, reference)
    if source_audit["status"] == "PASS":
        sys.path.insert(0, str(Path(args.source_root).resolve() / "src"))
        from qjlb.cli import load_model

        rows = {row["scenario"]: row for row in reference["public_scenarios"]}
        checks = []
        for name in sorted(rows):
            model, _ = load_model(Path(args.source_root) / "scenarios" / f"{name}.json")
            checks.append(
                {
                    "scenario": name,
                    "problem_id": model.fingerprint,
                    "expected_problem_id": rows[name]["problem_id"],
                    "pass": model.fingerprint == rows[name]["problem_id"],
                }
            )
        source_audit.update(
            problem_ids=checks,
            problem_ids_pass=sum(row["pass"] for row in checks),
            problem_ids_total=len(checks),
            interface_validation="PASS",
            official_interface_validation="PASS",
        )
        if not all(row["pass"] for row in checks):
            source_audit["status"] = "FAIL"
    p1b = verify_p1b(CANDIDATE_ROOT)
    v1 = verify_v1_exact(CANDIDATE_ROOT)
    seven = None
    seven_path = results / "seven_case_summary.json"
    if seven_path.is_file():
        seven = json.loads(seven_path.read_text())
    env = environment(
        CANDIDATE_ROOT,
        "ADMITTED_AND_EXECUTED" if seven and seven["status"] == "PASS_CANDIDATE" else "ADMITTED_NOT_EXECUTED",
    )

    write_json(results / "official_reference_ledger.json", ledger)
    write_json(results / "official_source_audit.json", source_audit)
    write_json(results / "p1b_integration.json", p1b)
    write_json(results / "v1_exact_reference.json", v1)
    write_json(results / "environment.json", env)

    executed_pass = (
        ledger["status"] == "PASS"
        and source_audit["status"] == "PASS"
        and source_audit.get("problem_ids_pass") == 7
        and p1b["status"] == "PASS"
        and v1["status"] == "PASS"
        and seven is not None
        and seven["status"] == "PASS_CANDIDATE"
    )
    summary = {
        "status": "PASS_CANDIDATE" if executed_pass else "PARTIAL",
        "executed_checks": "PASS" if executed_pass else "FAIL",
        "official_reference_ledger": ledger["status"],
        "official_source_audit": source_audit["status"],
        "official_path_contract": "PASS" if source_audit["canonical_path_count"] == 27 else "FAIL",
        "official_problem_ids": f"{source_audit.get('problem_ids_pass', 0)}/{source_audit.get('problem_ids_total', 7)}",
        "official_interface_validation": source_audit.get("interface_validation"), 
        "official_seven_case_interface": "PASS" if seven and seven["status"] == "PASS_CANDIDATE" else "NOT_VALIDATED",
        "p1b_integration": p1b["status"],
        "p1b_scientific_semantics": "UNCHANGED" if p1b["semantics_unchanged"] else "FAIL",
        "v1_exact_reference": v1["status"],
        "p1b_gatekeeper": p1b["p1b_gatekeeper"],
        "resource": {
            "wall_s": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_status": "RSS_MONITOR_ONLY",
            "budget_s": 900,
            "recommended_rss_limit_kib": 1572864,
        },
        "claims": {
            "formal_benchmark": "NOT_RUN",
            "new_quantum_algorithm": "NO",
            "full_three_year_quantum_poc": "NOT_RUN",
            "quantum_advantage": "NOT_CLAIMED",
            "official_source_interface_executed": "YES" if seven and seven["status"] == "PASS_CANDIDATE" else "NO",
            "git_write_operations": "NO",
            "publication": "NO",
            "submission": "NO",
        },
    }
    write_json(results / "reproducibility_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if executed_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
