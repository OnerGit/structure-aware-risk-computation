from pathlib import Path

from adb_v2.p1b_evidence import verify_p1b

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]


def test_p1b_integration_preserves_verified_semantics():
    result = verify_p1b(CANDIDATE_ROOT)
    assert result["status"] == "PASS"
    assert result["p1b_gatekeeper"] == "PASS_WITH_SCOPE_LIMITS"
    assert result["semantics_unchanged"] is True
    assert result["component_errors"] == []
    assert result["netlist_errors"] == []
    assert result["components"]["P"]["expanded_primitives"] == 66
    assert result["components"]["A"]["expanded_primitives"] == 237
    assert result["components"]["Q"]["expanded_primitives"] == 658
    assert result["components"]["Q"]["depth_expanded"] == 345
    assert result["components"]["Q"]["depth_ccx_macro"] == 141
    assert all(status == "PASS" for status in result["checks"].values())
