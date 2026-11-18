from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT / "src"))

from adb_v2.official_source import CANONICAL_PATHS, official_source_audit
from adb_v2.reference import load_reference


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sanitize(text: str, source_root: Path, work: Path) -> str:
    replacements = [
        (str(source_root.resolve()), "<OFFICIAL_SOURCE_ROOT>"),
        (str(CANDIDATE_ROOT.resolve()), "<CANDIDATE_ROOT>"),
        (str(work.resolve()), "<CASE_WORKDIR>"),
        (str(Path.home()), "<HOME>"),
    ]
    for value, label in replacements:
        text = text.replace(value, label)
    return text


def rejected(name: str, error) -> dict:
    return {
        "control": name,
        "status": "REJECTED",
        "detail": str(error),
    }


def accepted(name: str, detail: str) -> dict:
    return {"control": name, "status": "ACCEPTED", "detail": detail}


def expect_rejection(name: str, function) -> dict:
    try:
        function()
    except Exception as error:
        return rejected(name, f"{type(error).__name__}: {error}")
    return accepted(name, "validator did not reject the control")


def fake_root_missing_flood(source_root: Path, directory: Path) -> Path:
    from adb_v2.official_source import OFFICIAL_ROOT_NAME

    root = directory / OFFICIAL_ROOT_NAME
    for rel in CANONICAL_PATHS:
        if rel == "data/flood.npz":
            continue
        source = source_root / rel
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Exercise mandatory rejection paths.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--results", default=str(CANDIDATE_ROOT / "results" / "seven_cases"))
    parser.add_argument("--output", default=str(CANDIDATE_ROOT / "results" / "negative_controls.json"))
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    archive = Path(args.archive).resolve()
    results = Path(args.results).resolve()
    output = Path(args.output).resolve()
    reference = load_reference(CANDIDATE_ROOT)
    sys.path.insert(0, str(source_root / "src"))
    from qjlb.cli import load_model
    from qjlb.evaluation import strict_submission
    from qjlb.validation import read_json

    drb_case = results / "adb-base"
    elt_case = results / "kaz-kgz-base"
    drb_request = json.loads((drb_case / "request.json").read_text())
    drb_submission = json.loads((drb_case / "result.json").read_text())
    elt_request = json.loads((elt_case / "request.json").read_text())
    elt_submission = json.loads((elt_case / "result.json").read_text())
    drb_model, _ = load_model(source_root / "scenarios" / "adb-base.json")
    elt_model, _ = load_model(source_root / "scenarios" / "kaz-kgz-base.json")
    controls = []

    def baseline():
        review = strict_submission(drb_submission, drb_request, drb_model, "classical_cpu", 10.0)
        if review != "code_review_required":
            raise ValueError("valid classical baseline did not pass strict validation")
        second = strict_submission(elt_submission, elt_request, elt_model, "classical_cpu", 10.0)
        if second != "code_review_required":
            raise ValueError("valid ELT baseline did not pass strict validation")

    try:
        baseline()
        controls.append({"control": "valid_baseline", "status": "PASS"})
    except Exception as error:
        controls.append({"control": "valid_baseline", "status": "FAIL", "detail": str(error)})

    def with_change(source: dict, mutation):
        changed = copy.deepcopy(source)
        mutation(changed)
        return changed

    def wrong_problem():
        submission = with_change(drb_submission, lambda s: s.__setitem__("problem_id", "0" * 64))
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def wrong_run():
        submission = with_change(drb_submission, lambda s: s.__setitem__("run_id", "stale-run"))
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def wrong_seed():
        submission = with_change(drb_submission, lambda s: s.__setitem__("seed", 999))
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def missing_model_evaluations():
        submission = with_change(
            drb_submission, lambda s: s["resources"].pop("model_evaluations")
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def negative_resource():
        submission = with_change(
            drb_submission, lambda s: s["resources"].__setitem__("elapsed_seconds", -1.0)
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def noninteger_resource():
        submission = with_change(
            drb_submission, lambda s: s["resources"].__setitem__("shots", 1.5)
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def nan_estimate():
        name = drb_model.metric_names[0]
        submission = with_change(
            drb_submission, lambda s: s["estimates"].__setitem__(name, float("nan"))
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def inf_estimate():
        name = drb_model.metric_names[2]
        submission = with_change(
            drb_submission, lambda s: s["estimates"].__setitem__(name, float("inf"))
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def out_of_range():
        name = drb_model.metric_names[0]
        submission = with_change(
            drb_submission, lambda s: s["estimates"].__setitem__(name, -1.0)
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def incomplete_five():
        submission = with_change(
            drb_submission,
            lambda s: s["estimates"].pop(drb_model.metric_names[4]),
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def metric_mix():
        names = list(elt_model.metric_names)
        submission = with_change(
            drb_submission,
            lambda s: s.__setitem__(
                "estimates", {name: float(value) for name, value in zip(names, s["estimates"].values())}
            ),
        )
        strict_submission(submission, drb_request, drb_model, "classical_cpu", 10.0)

    def invalid_execution():
        submission = with_change(
            drb_submission, lambda s: s.__setitem__("execution", "quantum_simulator")
        )
        strict_submission(submission, drb_request, drb_model, "quantum_simulator", 10.0)

    for name, function in (
        ("wrong_problem_id", wrong_problem),
        ("wrong_run_id", wrong_run),
        ("wrong_seed", wrong_seed),
        ("missing_model_evaluations", missing_model_evaluations),
        ("negative_resource", negative_resource),
        ("noninteger_resource", noninteger_resource),
        ("nan_estimate", nan_estimate),
        ("inf_estimate", inf_estimate),
        ("out_of_range_estimate", out_of_range),
        ("incomplete_five_metrics", incomplete_five),
        ("drb_elt_metric_mix", metric_mix),
        ("invalid_execution", invalid_execution),
    ):
        controls.append(expect_rejection(name, function))

    def duplicate_key_json():
        with tempfile.TemporaryDirectory(prefix="s1-negative-json-") as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"method":"a","method":"b"}\n')
            read_json(duplicate)

    controls.append(expect_rejection("duplicate_json_key", duplicate_key_json))

    try:
        audit = official_source_audit(None, archive, reference)
        controls.append(
            rejected("missing_source_root", audit["status"])
            if audit["status"].startswith("FAIL")
            else accepted("missing_source_root", audit["status"])
        )
    except Exception as error:
        controls.append(rejected("missing_source_root", f"{type(error).__name__}: {error}"))

    temp = Path(tempfile.mkdtemp(prefix="s1-negative-missing-"))
    try:
        fake = fake_root_missing_flood(source_root, temp)
        audit = official_source_audit(fake, archive, reference)
        controls.append(
            rejected("missing_npz", json.dumps(audit["errors"]))
            if audit["status"] == "FAIL"
            else accepted("missing_npz", audit["status"])
        )
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    stale = Path(tempfile.mkdtemp(prefix="s1-negative-stale-"))
    try:
        (stale / "cases").mkdir()
        (stale / "data").mkdir()
        shutil.copyfile(source_root / "scenarios" / "kaz-kgz-base.json", stale / "cases" / "case.json")
        for name in ("audit.json", "seismic.npz", "flood.npz"):
            shutil.copyfile(source_root / "data" / name, stale / "data" / name)
        with (stale / "data" / "seismic.npz").open("ab") as stream:
            stream.write(b"stale")
        request = {
            "schema_version": 1,
            "run_id": "negative-stale-data",
            "problem_id": elt_request["problem_id"],
            "track": "worldbank_elt",
            "config_file": "cases/case.json",
            "seed": 0,
            "samples": 1,
            "targets": elt_request["targets"],
        }
        write_json(stale / "request.json", request)
        env = {
            k: os.environ[k]
            for k in ("PATH", "LANG", "SYSTEMROOT")
            if k in os.environ
        }
        env.update(
            HOME=str(stale),
            TMPDIR=str(stale),
            PYTHONNOUSERSITE="1",
            QJLB_SOURCE_ROOT=str(source_root / "src"),
        )
        process = subprocess.run(
            [
                sys.executable,
                str(CANDIDATE_ROOT / "team_solver.py"),
                "--request",
                "request.json",
                "--output",
                "result.json",
            ],
            cwd=stale,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        if process.returncode != 0 and not (stale / "result.json").exists():
            controls.append(
                rejected(
                    "stale_data_hash",
                    sanitize(process.stderr.strip().splitlines()[-1], source_root, stale),
                )
            )
        else:
            controls.append(accepted("stale_data_hash", f"exit {process.returncode}"))
    finally:
        shutil.rmtree(stale, ignore_errors=True)

    all_rejected = all(row["status"] == "REJECTED" for row in controls if row["control"] != "valid_baseline")
    baseline_pass = next(row for row in controls if row["control"] == "valid_baseline")["status"] == "PASS"
    summary = {
        "status": "PASS" if all_rejected and baseline_pass else "FAIL",
        "valid_baseline": baseline_pass,
        "controls_total": len(controls) - 1,
        "controls_rejected": sum(row["status"] == "REJECTED" for row in controls),
        "all_controls_rejected": all_rejected,
        "tolerance_relaxation": "NO",
        "controls": controls,
    }
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
