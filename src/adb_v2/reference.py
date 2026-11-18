from __future__ import annotations

import hashlib
import json
from pathlib import Path

REFERENCE_RELATIVE = Path("scenarios/official_seven_cases_reference.json")
P1B_REFERENCE_RELATIVE = Path("scenarios/p1b_annual_reference_amended.json")
P1B_REFERENCE_SHA256 = "06f34f6a0a4d6d8c5d1fa4fb871203fef31733fa71bb1dea5e19195250015bd7"

TRACK_METRICS = {
    "adb_drb": (
        "term_expected_principal_loss_usd",
        "term_joint_attachment_probability",
        "term_portfolio_loss_exceedance_probability",
        "term_portfolio_stop_loss_usd",
        "term_both_bonds_exhausted_probability",
    ),
    "worldbank_elt": (
        "annual_economic_loss_usd",
        "annual_layer_payment_usd",
        "joint_layer_exceedance_rate_per_year",
        "portfolio_layer_exceedance_rate_per_year",
        "annual_layer_stop_loss_usd",
    ),
}

METRIC_UNITS = {
    "term_expected_principal_loss_usd": "USD_over_term",
    "term_joint_attachment_probability": "probability_over_term",
    "term_portfolio_loss_exceedance_probability": "probability_over_term",
    "term_portfolio_stop_loss_usd": "USD_over_term",
    "term_both_bonds_exhausted_probability": "probability_over_term",
    "annual_economic_loss_usd": "USD_per_year",
    "annual_layer_payment_usd": "USD_per_year",
    "joint_layer_exceedance_rate_per_year": "event_rate_per_year",
    "portfolio_layer_exceedance_rate_per_year": "event_rate_per_year",
    "annual_layer_stop_loss_usd": "USD_per_year",
}

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(candidate_root: Path | None = None) -> dict:
    root = candidate_root or Path(__file__).resolve().parents[2]
    return json.loads((root / REFERENCE_RELATIVE).read_text())


def validate_reference_ledger(document: dict) -> dict:
    errors = []
    scenarios = document.get("public_scenarios", [])
    if document.get("public_scenario_count") != 7 or len(scenarios) != 7:
        errors.append("public scenario count is not 7")
    drb = [row for row in scenarios if row.get("track") == "adb_drb"]
    elt = [row for row in scenarios if row.get("track") == "worldbank_elt"]
    if len(drb) != 5:
        errors.append(f"DRB case count is {len(drb)}, expected 5")
    if len(elt) != 2:
        errors.append(f"ELT case count is {len(elt)}, expected 2")

    seen_ids = set()
    seen_configs = set()
    for row in scenarios:
        scenario = row.get("scenario")
        track = row.get("track")
        if track not in TRACK_METRICS:
            errors.append(f"{scenario}: unsupported track {track!r}")
            continue
        problem_id = row.get("problem_id", "")
        config_sha = row.get("config_sha256", "")
        if not isinstance(problem_id, str) or len(problem_id) != 64:
            errors.append(f"{scenario}: invalid problem_id")
        if problem_id in seen_ids:
            errors.append(f"{scenario}: duplicate problem_id")
        seen_ids.add(problem_id)
        if not isinstance(config_sha, str) or len(config_sha) != 64:
            errors.append(f"{scenario}: invalid config_sha256")
        if config_sha in seen_configs:
            errors.append(f"{scenario}: duplicate config_sha256")
        seen_configs.add(config_sha)
        metrics = row.get("metrics", [])
        expected_names = TRACK_METRICS[track]
        if row.get("metric_count") != 5 or len(metrics) != 5:
            errors.append(f"{scenario}: metric count is not 5")
        got_names = tuple(metric.get("name") for metric in metrics)
        if got_names != expected_names:
            errors.append(f"{scenario}: metric names/order mismatch")
        for metric in metrics:
            name = metric.get("name")
            unit = metric.get("unit")
            if METRIC_UNITS.get(name) != unit:
                errors.append(f"{scenario}/{name}: unit {unit!r} mismatch")
            value = metric.get("exact_reference")
            if not isinstance(value, (int, float)) or value != value:
                errors.append(f"{scenario}/{name}: non-finite exact reference")

    p1b_doc = None
    p1b_hash = None
    p1b_errors = []
    root = Path(__file__).resolve().parents[2]
    p1b_path = root / P1B_REFERENCE_RELATIVE
    if p1b_path.is_file():
        p1b_hash = sha256_file(p1b_path)
        p1b_doc = json.loads(p1b_path.read_text())
        if p1b_hash != P1B_REFERENCE_SHA256:
            p1b_errors.append("P1-B amended reference SHA256 mismatch")
        if len(p1b_doc.get("annual_probabilities_100", [])) != 100:
            p1b_errors.append("P1-B annual vector is not length 100")
        probabilities = p1b_doc.get("annual_probabilities_100", [])
        if abs(sum(probabilities) - 1.0) > 1e-12:
            p1b_errors.append("P1-B annual probability mass is not 1")
        if abs(p1b_doc.get("selected_event", {}).get("probability", 0.0) - probabilities[99]) > 1e-15:
            p1b_errors.append("P1-B selected event does not equal annual index 99")
        if p1b_doc.get("rain_order") != [[0, 0], [0, 1], [1, 0], [1, 1]]:
            p1b_errors.append("P1-B rain order is not corrected C-order")
    else:
        p1b_errors.append("P1-B amended reference not found")

    status = "PASS" if not errors and not p1b_errors else "FAIL"
    return {
        "status": status,
        "track_counts": {"adb_drb": len(drb), "worldbank_elt": len(elt)},
        "scenario_count": len(scenarios),
        "metric_count": sum(len(row.get("metrics", [])) for row in scenarios),
        "unique_problem_ids": len(seen_ids),
        "unique_config_hashes": len(seen_configs),
        "errors": errors,
        "p1b_reference": {
            "sha256": p1b_hash,
            "target_index": p1b_doc.get("selected_event", {}).get("annual_state_index") if p1b_doc else None,
            "target_probability": p1b_doc.get("selected_event", {}).get("probability") if p1b_doc else None,
            "errors": p1b_errors,
        },
    }
