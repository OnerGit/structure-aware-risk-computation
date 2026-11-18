from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from adb_v2.reference import load_reference
from adb_v2.official_source import official_source_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the explicit official source root.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    reference = (
        json.loads(Path(args.reference).read_text())
        if args.reference
        else load_reference(CANDIDATE_ROOT)
    )
    audit = official_source_audit(args.source_root, args.archive, reference)
    if audit["status"] != "PASS":
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 3

    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root / "src"))
    from qjlb.cli import load_model

    reference_rows = {
        row["scenario"]: row for row in reference["public_scenarios"]
    }
    checks = []
    for name in sorted(reference_rows):
        config = source_root / "scenarios" / f"{name}.json"
        model, _ = load_model(config)
        expected = reference_rows[name]["problem_id"]
        actual = model.fingerprint
        checks.append(
            {
                "scenario": name,
                "problem_id": actual,
                "expected_problem_id": expected,
                "pass": actual == expected,
            }
        )
    problems_pass = all(row["pass"] for row in checks)
    audit.update(
        problem_ids=checks,
        problem_ids_pass=sum(row["pass"] for row in checks),
        problem_ids_total=len(checks),
        official_path_contract="PASS" if audit["status"] == "PASS" else "FAIL",
        official_interface_validation="PASS",
        interface_validation="PASS",
        status="PASS" if audit["status"] == "PASS" and problems_pass else "FAIL",
    )
    if not problems_pass:
        audit["status"] = "FAIL"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
