"""Official benchmark/verification layer (Brief Table 5 and eq. tolerances).

This module holds the publisher's rounded reference values. It is deliberately
isolated: the computation core in ``adb_g1a.model`` must not import or consult
it. It exists so the comparison tooling and tests can reconcile the exact
computed outputs with the official rounded table without touching the model.
"""

from __future__ import annotations

# Source of the reference values: ADB Quantum Innovation Challenge Brief,
# "Table 5: Exact reference values per scenario (three-year term)",
# as supplied in the project handoff and locked in
# resources/planning/ADB_G0_Requirement_Benchmark_Plan_2026-11-04.md (sec 2.3).
# SHA256 of the official PDF: cf731f2a... (see provenance manifest).
REFERENCE_SOURCE_ID = ("ADB_Quantum_Innovation_Challenge_Brief_Table5_"
                       "2026-11-05_project_handoff")

# (display unit, nominal print precision, tolerance in probability units)
# Internal comparison units: USD for money, probability 0..1 for chances.
METRIC_SPECS = {
    "term_expected_principal_loss_usd": dict(
        unit="USD million",
        display_precision=0.01,          # printed to 2 decimals
        tolerance=0.005e6,               # +/- 0.005 USD m
    ),
    "term_joint_attachment_probability": dict(
        unit="percentage point",
        display_precision=0.001,         # printed to 3 decimals
        tolerance=0.0005e-2,             # +/- 0.0005 percentage points
    ),
    "term_portfolio_loss_exceedance_probability": dict(
        unit="percentage point",
        display_precision=0.001,
        tolerance=0.0005e-2,
    ),
    "term_portfolio_stop_loss_usd": dict(
        unit="USD",
        display_precision=1,             # whole dollars
        tolerance=0.5,                   # +/- 0.5 USD
    ),
    "term_both_bonds_exhausted_probability": dict(
        unit="percentage point",
        display_precision=0.0001,        # printed to 4 decimals
        tolerance=0.00005e-2,            # +/- 0.00005 percentage points
    ),
}

# Official published reference values (Term Table 5).
# Probability fields are stored as probabilities (0..1) before display.
OFFICIAL_REFERENCE_TABLE_5 = {
    "adb-base": {
        "term_expected_principal_loss_usd": 10.06e6,
        "term_joint_attachment_probability": 0.04414,
        "term_portfolio_loss_exceedance_probability": 0.01205,
        "term_portfolio_stop_loss_usd": 282_355.0,
        "term_both_bonds_exhausted_probability": 0.000032,
    },
    "adb-independent": {
        "term_expected_principal_loss_usd": 10.06e6,
        "term_joint_attachment_probability": 0.02716,
        "term_portfolio_loss_exceedance_probability": 0.00719,
        "term_portfolio_stop_loss_usd": 171_266.0,
        "term_both_bonds_exhausted_probability": 0.000004,
    },
    "adb-common-shock": {
        "term_expected_principal_loss_usd": 10.06e6,
        "term_joint_attachment_probability": 0.05053,
        "term_portfolio_loss_exceedance_probability": 0.01607,
        "term_portfolio_stop_loss_usd": 385_043.0,
        "term_both_bonds_exhausted_probability": 0.000044,
    },
    "adb-rain-stress": {
        "term_expected_principal_loss_usd": 11.17e6,
        "term_joint_attachment_probability": 0.07046,
        "term_portfolio_loss_exceedance_probability": 0.01322,
        "term_portfolio_stop_loss_usd": 306_286.0,
        "term_both_bonds_exhausted_probability": 0.000069,
    },
    "adb-rain-cover": {
        "term_expected_principal_loss_usd": 9.97e6,
        "term_joint_attachment_probability": 0.04414,
        "term_portfolio_loss_exceedance_probability": 0.00700,
        "term_portfolio_stop_loss_usd": 170_927.0,
        "term_both_bonds_exhausted_probability": 0.000032,
    },
}


def display_value(metric: str, internal: float) -> float:
    """Convert an internal value to its display unit."""
    spec = METRIC_SPECS[metric]
    if spec["unit"] == "USD million":
        return internal / 1e6
    if spec["unit"] == "percentage point":
        return internal * 100.0
    return internal


def within_tolerance(metric: str, computed: float, published: float) -> bool:
    return abs(computed - published) <= METRIC_SPECS[metric]["tolerance"]


def compare_scenario(scenario: str, computed: dict[str, float]) -> list[dict]:
    """Full diagnostic rows for one scenario (raw units + display info)."""
    rows = []
    ref = OFFICIAL_REFERENCE_TABLE_5[scenario]
    for metric, spec in METRIC_SPECS.items():
        c = computed[metric]
        p = ref[metric]
        rows.append({
            "scenario": scenario,
            "metric": metric,
            "computed_full_precision": c,
            "published_reference": p,
            "displayed_unit": spec["unit"],
            "absolute_error": c - p,
            "official_rounding_tolerance": spec["tolerance"],
            "within_tolerance": within_tolerance(metric, c, p),
            "source": REFERENCE_SOURCE_ID,
        })
    return rows


def compare_all(computed: dict[str, dict[str, float]]) -> list[dict]:
    rows: list[dict] = []
    for scenario in list(OFFICIAL_REFERENCE_TABLE_5):
        rows.extend(compare_scenario(scenario, computed[scenario]))
    return rows