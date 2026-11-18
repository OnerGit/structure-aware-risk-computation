from pathlib import Path

from adb_v2.official_source import official_source_audit
from adb_v2.reference import load_reference, validate_reference_ledger

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]


def test_seven_case_reference_ledger_is_valid():
    document = load_reference(CANDIDATE_ROOT)
    result = validate_reference_ledger(document)
    assert result["status"] == "PASS"
    assert result["scenario_count"] == 7
    assert result["track_counts"] == {"adb_drb": 5, "worldbank_elt": 2}
    assert result["metric_count"] == 35
    assert result["unique_problem_ids"] == 7
    assert result["unique_config_hashes"] == 7


def test_official_source_is_not_fabricated():
    document = load_reference(CANDIDATE_ROOT)
    result = official_source_audit(None, None, document)
    assert result["status"] == "FAIL_CLOSED_SOURCE_ROOT_REQUIRED"
    assert result["interface_validation"] == "NOT_PERFORMED"
    assert result["canonical_path_count"] == 27
    assert "team_solver.py" in result["minimum_missing"]
    assert "data/adb_terms.json" in result["minimum_missing"]
    assert "src/qjlb/model.py" in result["minimum_missing"]
    assert "src/qjlb/elt.py" not in result["minimum_missing"]
