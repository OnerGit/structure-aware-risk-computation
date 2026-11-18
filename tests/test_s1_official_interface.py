from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))

from team_solver import validate_request

RESULTS = CANDIDATE_ROOT / "results"
CASES = (
    "adb-base",
    "adb-common-shock",
    "adb-independent",
    "adb-rain-cover",
    "adb-rain-stress",
    "kaz-kgz-base",
    "tjk-uzb-base",
)
DRB_METRICS = {
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
}
ELT_METRICS = {
    "annual_economic_loss_usd",
    "annual_layer_payment_usd",
    "joint_layer_exceedance_rate_per_year",
    "portfolio_layer_exceedance_rate_per_year",
    "annual_layer_stop_loss_usd",
}


def load(path: Path):
    return json.loads(path.read_text())


def test_official_source_admission_is_seven_of_seven():
    admission = load(RESULTS / "official_source_admission.json")
    assert admission["status"] == "PASS"
    assert admission["archive"]["status"] == "PASS"
    assert admission["problem_ids_pass"] == 7
    assert admission["problem_ids_total"] == 7
    assert admission["canonical_path_count"] == 27
    assert admission["recorded_elt_path"] == "src/qjlb/model.py"
    assert admission["root_manifest"]["status"] == "PARTIAL_36_OF_37_GITIGNORE_ABSENT"
    assert admission["root_manifest"]["missing"] == [".gitignore"]
    assert admission["root_manifest"]["matched"] == 36


def test_seven_case_summary_is_candidate_pass():
    summary = load(RESULTS / "seven_case_summary.json")
    assert summary["status"] == "PASS_CANDIDATE"
    assert summary["source_admission"] == "PASS"
    assert summary["problem_ids"] == "7/7"
    assert summary["solver_execution"] == "5/5_DRB + 2/2_ELT"
    assert summary["five_metric_correctness"] == "PASS"
    assert summary["official_protocol"] == "PASS"
    assert summary["execution"] == "classical_cpu"
    assert summary["method"] == "official_exact_classical_dp"
    assert summary["resource"]["total_wall_seconds"] <= 900
    assert summary["resource"]["peak_case_rss_kib"] <= 1572864


@pytest.mark.parametrize("case", CASES)
def test_each_case_has_five_metric_evidence(case):
    directory = RESULTS / "seven_cases" / case
    request = load(directory / "request.json")
    result = load(directory / "result.json")
    comparison = load(directory / "comparison.json")
    validation = load(directory / "validation.json")
    expected = DRB_METRICS if request["track"] == "adb_drb" else ELT_METRICS
    assert set(result["estimates"]) == expected
    assert len(result["estimates"]) == 5
    assert result["execution"] == "classical_cpu"
    assert result["method"] == "official_exact_classical_dp"
    assert result["resources"]["shots"] == 0
    assert result["resources"]["qubits"] == 0
    assert result["resources"]["model_evaluations"] >= 1
    assert validation["strict_submission"] == "code_review_required"
    assert comparison["five_metric_pass"] is True
    for group in ("team_vs_official", "independent_vs_official", "team_vs_independent"):
        assert len(comparison[group]) == 5
        assert all(row["within_tolerance"] for row in comparison[group])


def test_negative_controls_are_all_rejected():
    controls = load(RESULTS / "negative_controls.json")
    assert controls["status"] == "PASS"
    assert controls["valid_baseline"] is True
    assert controls["controls_total"] == 16
    assert controls["controls_rejected"] == 16
    assert controls["all_controls_rejected"] is True
    assert controls["tolerance_relaxation"] == "NO"
    for row in controls["controls"]:
        assert row["control"] == "valid_baseline" or row["status"] == "REJECTED"


def test_request_validation_rejects_missing_and_escaping_paths(tmp_path):
    config = tmp_path / "case.json"
    config.write_text("{}\n")
    valid = {
        "schema_version": 1,
        "run_id": "r",
        "problem_id": "a" * 64,
        "track": "adb_drb",
        "config_file": "case.json",
        "seed": 0,
        "samples": 1,
        "targets": list(DRB_METRICS),
    }
    validate_request(valid, tmp_path)
    broken = dict(valid)
    broken.pop("run_id")
    with pytest.raises(ValueError):
        validate_request(broken, tmp_path)
    escaping = dict(valid)
    escaping["config_file"] = "../case.json"
    with pytest.raises(ValueError):
        validate_request(escaping, tmp_path)
    noninteger = dict(valid)
    noninteger["samples"] = 1.5
    with pytest.raises(ValueError):
        validate_request(noninteger, tmp_path)
