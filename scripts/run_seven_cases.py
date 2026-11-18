from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from adb_v2.independent_reference import independent_drb, independent_elt
from adb_v2.official_source import DRB_CASES, ELT_CASES, official_source_audit
from adb_v2.reference import load_reference

SEVEN_CASES = (
    "adb-base",
    "adb-common-shock",
    "adb-independent",
    "adb-rain-cover",
    "adb-rain-stress",
    "kaz-kgz-base",
    "tjk-uzb-base",
)
CASE_TIMEOUT_SECONDS = 60.0
TOTAL_BUDGET_SECONDS = 900.0
SAMPLES = 8192


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sanitize(text: str, source_root: Path, candidate_root: Path, work: Path) -> str:
    replacements = [
        (str(source_root.resolve()), "<OFFICIAL_SOURCE_ROOT>"),
        (str(candidate_root.resolve()), "<CANDIDATE_ROOT>"),
        (str(work.resolve()), "<CASE_WORKDIR>"),
        (str(Path.home()), "<HOME>"),
    ]
    for value, label in replacements:
        text = text.replace(value, label)
    return text


def observe_time_trace(path: Path) -> dict:
    text = path.read_text()
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    status = re.search(r"Exit status: (\d+)", text)
    return {
        "external_wall_clock": elapsed.group(1) if elapsed else None,
        "max_rss_kib": int(rss.group(1)) if rss else None,
        "exit_status": int(status.group(1)) if status else None,
    }


def copy_case_inputs(case: str, source_root: Path, work: Path) -> None:
    config_source = source_root / "scenarios" / f"{case}.json"
    config = json.loads(config_source.read_text())
    (work / "cases").mkdir(parents=True, exist_ok=True)
    (work / "data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_source, work / "cases" / "case.json")
    if config["track"] == "adb_drb":
        names = ["adb_terms.json"]
    else:
        names = ["audit.json", "seismic.npz", "flood.npz"]
    for name in names:
        shutil.copyfile(source_root / "data" / name, work / "data" / name)


def compare_values(candidate: dict, reference: dict, tolerances: dict) -> list[dict]:
    rows = []
    for name in reference:
        left = float(candidate[name])
        right = float(reference[name])
        tolerance = float(tolerances[name])
        error = abs(left - right)
        rows.append(
            {
                "metric": name,
                "candidate": left,
                "reference": right,
                "absolute_error": error,
                "tolerance": tolerance,
                "within_tolerance": error <= tolerance,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public seven cases via the official CLI contract.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default=str(CANDIDATE_ROOT / "results" / "seven_cases"))
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"output already exists; choose a fresh path: {output}")
    output.mkdir(parents=True)

    reference = load_reference(CANDIDATE_ROOT)
    source_audit = official_source_audit(source_root, archive, reference)
    write_json(output.parent / "source_admission_runner.json", source_audit)
    if source_audit["status"] != "PASS":
        print(json.dumps(source_audit, indent=2, sort_keys=True))
        return 3

    # The official runtime is imported only after the explicit source audit.
    sys.path.insert(0, str(source_root / "src"))
    from qjlb.cli import load_model, methods_for
    from qjlb.evaluation import strict_submission, tolerances
    from qjlb.validation import check_submission, read_json

    reference_rows = {
        row["scenario"]: row for row in reference["public_scenarios"]
    }
    start = time.perf_counter()
    case_results = []
    for index, case in enumerate(SEVEN_CASES):
        case_dir = output / case
        work = Path(tempfile.mkdtemp(prefix=f"s1-{case}-"))
        completed = None
        try:
            copy_case_inputs(case, source_root, work)
            model, preparation_seconds = load_model(work / "cases/case.json")
            expected_problem = reference_rows[case]["problem_id"]
            if model.fingerprint != expected_problem:
                raise ValueError("official problem_id differs from frozen ledger")
            if list(model.metric_names) != [
                metric["name"] for metric in reference_rows[case]["metrics"]
            ]:
                raise ValueError("official metric names differ from frozen ledger")
            if case in DRB_CASES:
                independent = independent_drb(model.config, case)
            elif case in ELT_CASES:
                independent = independent_elt(model.config, model.dataset_directory)
            else:
                raise ValueError("case is not one of the authorized seven")
            official_reference = methods_for(model)["exact"](model, 0, 0)
            official_tolerance = tolerances(model, official_reference)

            request = {
                "schema_version": 1,
                "run_id": secrets.token_hex(16),
                "problem_id": model.fingerprint,
                "track": model.config["track"],
                "config_file": "cases/case.json",
                "seed": 1000 + index,
                "samples": SAMPLES,
                "targets": list(model.metric_names),
            }
            request_path = work / "request.json"
            write_json(request_path, request)
            result_path = work / "result.json"
            command = [
                "/usr/bin/time",
                "-o",
                "time.txt",
                "-v",
                sys.executable,
                str(CANDIDATE_ROOT / "team_solver.py"),
                "--request",
                "request.json",
                "--output",
                "result.json",
            ]
            env = {
                k: os.environ[k]
                for k in ("PATH", "LANG", "SYSTEMROOT")
                if k in os.environ
            }
            env.update(
                HOME=str(work),
                TMPDIR=str(work),
                PYTHONNOUSERSITE="1",
                QJLB_SOURCE_ROOT=str(source_root / "src"),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            process_start = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=work,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=CASE_TIMEOUT_SECONDS,
            )
            measured_wall = time.perf_counter() - process_start
            time_trace = observe_time_trace(work / "time.txt") if (work / "time.txt").exists() else None
            submission = read_json(result_path)
            check = check_submission(submission, model.fingerprint, model.metric_names)
            strict = strict_submission(submission, request, model, "classical_cpu", measured_wall)
            if strict != "code_review_required":
                raise ValueError(f"unexpected strict verification state: {strict}")

            team_values = submission["estimates"]
            team_vs_official = compare_values(
                team_values,
                {name: official_reference["estimates"][name]["estimate"] for name in model.metric_names},
                official_tolerance,
            )
            independent_vs_official = compare_values(
                independent["values"],
                {name: official_reference["estimates"][name]["estimate"] for name in model.metric_names},
                official_tolerance,
            )
            team_vs_independent = compare_values(
                team_values,
                independent["values"],
                official_tolerance,
            )
            five_metric_pass = all(
                row["within_tolerance"]
                for group in (team_vs_official, independent_vs_official, team_vs_independent)
                for row in group
            )
            resource = submission["resources"]
            if (
                type(resource["shots"]) is not int
                or resource["shots"] != 0
                or type(resource["qubits"]) is not int
                or resource["qubits"] != 0
                or type(resource["model_evaluations"]) is not int
                or resource["model_evaluations"] < 0
                or not math.isfinite(float(resource["elapsed_seconds"]))
                or float(resource["elapsed_seconds"]) > measured_wall * 1.1 + 0.05
            ):
                raise ValueError("resource record is not truthful")
            if submission["execution"] != "classical_cpu":
                raise ValueError("execution label is not classical_cpu")

            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(case_dir / "request.json", request)
            write_json(case_dir / "result.json", submission)
            write_json(case_dir / "config.json", model.config)
            write_json(case_dir / "official_reference.json", official_reference)
            write_json(case_dir / "independent_reference.json", independent)
            write_json(
                case_dir / "comparison.json",
                {
                    "case": case,
                    "track": model.config["track"],
                    "problem_id": model.fingerprint,
                    "five_metric_pass": five_metric_pass,
                    "team_vs_official": team_vs_official,
                    "independent_vs_official": independent_vs_official,
                    "team_vs_independent": team_vs_independent,
                    "official_tolerance": official_tolerance,
                },
            )
            write_json(
                case_dir / "validation.json",
                {
                    "check_submission": "PASS",
                    "strict_submission": strict,
                    "execution": submission["execution"],
                    "method": submission["method"],
                    "resources": resource,
                },
            )
            (case_dir / "stdout.txt").write_text(
                sanitize(completed.stdout, source_root, CANDIDATE_ROOT, work)
            )
            (case_dir / "stderr.txt").write_text(
                sanitize(completed.stderr, source_root, CANDIDATE_ROOT, work)
            )
            if (work / "time.txt").exists():
                (case_dir / "time_v.txt").write_text(
                    sanitize((work / "time.txt").read_text(), source_root, CANDIDATE_ROOT, work)
                )
            write_json(
                case_dir / "resource.json",
                {
                    "measured_wall_seconds": measured_wall,
                    "reported_elapsed_seconds": resource["elapsed_seconds"],
                    "reported_preparation_seconds": resource["preparation_seconds"],
                    "max_rss_kib": time_trace["max_rss_kib"] if time_trace else None,
                    "rss_status": "RSS_MONITOR_ONLY",
                    "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
                    "external_wall_clock": time_trace["external_wall_clock"] if time_trace else None,
                    "exit_status": completed.returncode,
                },
            )
            case_results.append(
                {
                    "case": case,
                    "track": model.config["track"],
                    "problem_id": model.fingerprint,
                    "problem_id_expected": expected_problem,
                    "problem_id_pass": model.fingerprint == expected_problem,
                    "method": submission["method"],
                    "execution": submission["execution"],
                    "five_metric_pass": five_metric_pass,
                    "strict_protocol_pass": True,
                    "resources": resource,
                    "measured_wall_seconds": measured_wall,
                    "max_rss_kib": time_trace["max_rss_kib"] if time_trace else None,
                    "status": "PASS" if five_metric_pass else "FAIL",
                }
            )
        except Exception as error:
            case_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                case_dir / "failure.json",
                {
                    "case": case,
                    "status": "FAIL",
                    "error": sanitize(str(error), source_root, CANDIDATE_ROOT, work),
                },
            )
            (case_dir / "stderr.txt").write_text(
                sanitize(
                    completed.stderr if completed is not None else "",
                    source_root,
                    CANDIDATE_ROOT,
                    work,
                )
            )
            case_results.append(
                {
                    "case": case,
                    "status": "FAIL",
                    "error": sanitize(str(error), source_root, CANDIDATE_ROOT, work),
                }
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

    total_wall = time.perf_counter() - start
    drb = [row for row in case_results if row.get("track") == "adb_drb" and row.get("status") == "PASS"]
    elt = [row for row in case_results if row.get("track") == "worldbank_elt" and row.get("status") == "PASS"]
    peak_rss = max((row.get("max_rss_kib") or 0 for row in case_results), default=0)
    overall = "PASS_CANDIDATE" if all(row.get("status") == "PASS" for row in case_results) else "PARTIAL"
    summary = {
        "status": overall,
        "source_admission": "PASS",
        "official_path_contract": "PASS",
        "problem_ids": f"{sum(bool(row.get('problem_id_pass')) for row in case_results)}/7",
        "solver_execution": f"{len(drb)}/5_DRB + {len(elt)}/2_ELT",
        "public_seven_case_solver": "PASS" if overall == "PASS_CANDIDATE" else "PARTIAL",
        "five_metric_correctness": "PASS" if overall == "PASS_CANDIDATE" else "PARTIAL",
        "official_protocol": "PASS" if all(row.get("strict_protocol_pass") for row in case_results if row.get("status") == "PASS") else "FAIL",
        "resource": {
            "total_wall_seconds": total_wall,
            "total_budget_seconds": TOTAL_BUDGET_SECONDS,
            "peak_case_rss_kib": peak_rss,
            "target_peak_rss_kib": 1572864,
            "rss_status": "RSS_MONITOR_ONLY",
        },
        "method": "official_exact_classical_dp",
        "execution": "classical_cpu",
        "stages": case_results,
    }
    write_json(output.parent / "seven_case_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if overall == "PASS_CANDIDATE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
