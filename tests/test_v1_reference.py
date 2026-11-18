from pathlib import Path

from adb_v2.v1_reference import verify_v1_exact

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]


def test_v1_exact_dp_matches_frozen_drb_reference():
    result = verify_v1_exact(CANDIDATE_ROOT)
    assert result["status"] == "PASS"
    assert result["rounding_reference_pass"] == 25
    assert result["rounding_reference_total"] == 25
    assert result["exact_reference_pass"] == 25
    assert result["max_exact_usd_abs_error"] <= 1e-6
    assert result["max_exact_probability_abs_error"] <= 1e-12
