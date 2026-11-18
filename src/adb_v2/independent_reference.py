from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from adb_g1a.model import SCENARIOS, compute_scenario

DRB_METRICS = (
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
)
ELT_METRICS = (
    "annual_economic_loss_usd",
    "annual_layer_payment_usd",
    "joint_layer_exceedance_rate_per_year",
    "portfolio_layer_exceedance_rate_per_year",
    "annual_layer_stop_loss_usd",
)
DRB_SCENARIOS = {scenario.key: scenario for scenario in SCENARIOS}


def _assert_drb_config(config: dict, scenario: str) -> None:
    expected = DRB_SCENARIOS.get(scenario)
    if expected is None:
        raise ValueError(f"no frozen V1 scenario for {scenario}")
    if config.get("track") != "adb_drb" or config.get("term_years") != 3:
        raise ValueError("official DRB config does not match the three-year model")
    if config.get("countries") != ["KGZ", "TJK"]:
        raise ValueError("official DRB countries mismatch")
    if config.get("earthquake_sublimit_usd") != [expected.eq_cover, expected.eq_cover]:
        raise ValueError("earthquake sublimit mismatch")
    if config.get("precipitation_sublimit_usd") != [
        expected.rain_cover,
        expected.rain_cover,
    ]:
        raise ValueError("precipitation sublimit mismatch")
    if config.get("earthquake_common_rate_multiplier") != expected.m:
        raise ValueError("earthquake common multiplier mismatch")
    if config.get("precipitation_common_rate_share") != expected.rain_common_share:
        raise ValueError("precipitation common share mismatch")
    if config.get("precipitation_rate_multiplier") != expected.rain_rate_multiplier:
        raise ValueError("precipitation multiplier mismatch")
    if config.get("precipitation_annual_trigger_probability") != [0.02, 0.02]:
        raise ValueError("precipitation trigger probability mismatch")
    if config.get("portfolio_threshold_usd") != 80_000_000.0:
        raise ValueError("portfolio threshold mismatch")


def independent_drb(config: dict, scenario: str) -> dict:
    _assert_drb_config(config, scenario)
    values = compute_scenario(DRB_SCENARIOS[scenario])
    return {
        "track": "adb_drb",
        "method": "frozen_v1_exact_dp",
        "values": {name: float(values[name]) for name in DRB_METRICS},
        "unit_note": "DRB values are USD or probability over the three-year term",
    }


def independent_elt(config: dict, data_directory: Path) -> dict:
    audit = json.loads((data_directory / "audit.json").read_text())
    countries = tuple(config["countries"])
    limit = np.asarray(config["limit_usd"], dtype=float)
    attachment = np.asarray(config["attachment_usd"], dtype=float)
    exhaustion = np.asarray(config["exhaustion_usd"], dtype=float)
    threshold = float(config["portfolio_threshold_usd"])
    if limit.shape != (len(countries),) or attachment.shape != (2, len(countries)):
        raise ValueError("invalid ELT contract shape")
    if exhaustion.shape != attachment.shape:
        raise ValueError("invalid ELT exhaustion shape")
    totals = np.zeros(5, dtype=float)
    hashes = {}
    for h, peril in enumerate(tuple(config["peril_order"])):
        path = data_directory / f"{peril}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != audit["perils"][peril]["sha256"]:
            raise ValueError(f"{peril}: data hash mismatch")
        hashes[peril] = digest
        with np.load(path, allow_pickle=False) as data:
            order = data["countries"].tolist()
            columns = [order.index(country) for country in countries]
            rates = data["annual_rate"].copy()
            loss = data["loss_usd"][:, columns].copy()
        payout = limit * np.clip(
            (loss - attachment[h]) / (exhaustion[h] - attachment[h]), 0.0, 1.0
        )
        total = payout.sum(axis=1)
        per_event = np.column_stack(
            [
                loss.sum(axis=1),
                total,
                np.all(payout > 0, axis=1),
                total > threshold,
                np.maximum(total - threshold, 0.0),
            ]
        )
        totals += rates @ per_event
    if not np.all(np.isfinite(totals)) or np.any(totals < 0):
        raise ValueError("independent ELT result is not finite and nonnegative")
    return {
        "track": "worldbank_elt",
        "method": "independent_npz_row_sum",
        "values": {name: float(totals[i]) for i, name in enumerate(ELT_METRICS)},
        "data_sha256": hashes,
        "unit_note": "ELT values are annual USD/year or event-rate/year",
    }


def absolute_differences(candidate: dict, reference: dict) -> dict:
    names = tuple(reference)
    if tuple(candidate) != names:
        raise ValueError("metric names differ")
    values = {
        name: {
            "candidate": float(candidate[name]),
            "reference": float(reference[name]),
            "absolute_error": abs(float(candidate[name]) - float(reference[name])),
        }
        for name in names
    }
    return values
