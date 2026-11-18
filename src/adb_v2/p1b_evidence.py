from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
from qiskit import qasm3

import p1b_circuit as circuit_module
import p1b_verify as verify_module
from p1b_circuit import (
    build_complete_operator,
    build_data_only_reflection_mutant,
    build_event_oracle,
    build_flag_phase,
    build_grover,
    build_state_preparation,
    build_zero_reflection,
    component_counts,
    reverse_adjoint,
)
from p1b_verify import (
    check_a_dagger_a,
    check_asymmetric_rain,
    check_budget,
    check_clean_subspace,
    check_event_mutant,
    check_event_oracle,
    check_flag_probability,
    check_preparation,
    check_q_action,
    check_s0_and_sf,
    check_wrong_grover_order,
    make_probes,
    reverse_sequence_check,
    serialized_netlist,
)

P1B_REFERENCE = Path("scenarios/p1b_annual_reference_amended.json")
P1B_REFERENCE_SHA256 = "06f34f6a0a4d6d8c5d1fa4fb871203fef31733fa71bb1dea5e19195250015bd7"
P1B_NETLIST_DIR = Path("results/p1b")
P1B_EVIDENCE_DIR = Path("evidence/p1b")

EXPECTED_SOURCE_SHA256 = {
    "src/p1b_circuit.py": "02ccb119f1798e94f74784c2519694fb6ea974e1af481cdbc392e2b8fd3f67bc",
    "src/p1b_verify.py": "226e18a0e195c7496fdecc97e6bbb2d519899b862392e620309a1aab9b2c1b95",
    "evidence/p1b/p1b_contract.py.frozen": "128b34fb3fc47c1710cd0e3a3461e10304f69b6adaecb73d1604c9a6087c8d74",
    "evidence/p1b/p1b_circuit.py.frozen": "02ccb119f1798e94f74784c2519694fb6ea974e1af481cdbc392e2b8fd3f67bc",
    "evidence/p1b/p1b_verify.py.frozen": "226e18a0e195c7496fdecc97e6bbb2d519899b862392e620309a1aab9b2c1b95",
}

EXPECTED_COMPONENTS = {
    "P": {"expanded_primitives": 66, "depth_expanded": 57, "qubits": 7},
    "Uf": {"expanded_primitives": 171, "depth_expanded": 101, "qubits": 13},
    "A": {"expanded_primitives": 237, "depth_expanded": 151, "qubits": 13},
    "A_dagger": {"expanded_primitives": 237, "depth_expanded": 148, "qubits": 13},
    "S0": {"expanded_primitives": 183, "depth_expanded": 102, "qubits": 13},
    "Sf": {"expanded_primitives": 1, "depth_expanded": 1, "qubits": 13},
    "Q": {"expanded_primitives": 658, "depth_expanded": 345, "depth_ccx_macro": 141, "qubits": 13},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rain_probabilities_c_order(q1: float, q2: float, shared_share: float = 0.25) -> np.ndarray:
    mu1 = -math.log(1.0 - q1)
    mu2 = -math.log(1.0 - q2)
    common = shared_share * min(mu1, mu2)
    p00 = math.exp(-(mu1 + mu2 - common))
    p01 = math.exp(-mu1) - p00
    p10 = math.exp(-mu2) - p00
    p11 = 1.0 - p00 - p01 - p10
    return np.array([p00, p01, p10, p11], dtype=float)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _netlist_comparison(root: Path, label: str, circuit, expand: bool) -> dict:
    frozen_path = root / P1B_NETLIST_DIR / f"{label}.netlist.json"
    output = circuit.decompose(gates_to_decompose=["ccx"]) if expand else circuit
    generated = serialized_netlist(output)
    frozen = json.loads(frozen_path.read_text())
    generated_sha = hashlib.sha256(_json_bytes(generated)).hexdigest()
    return {
        "label": label,
        "frozen_path": str(frozen_path.relative_to(root)),
        "frozen_sha256": sha256_file(frozen_path),
        "generated_sha256": generated_sha,
        "equal": generated == frozen,
        "gate_count": output.size(),
        "depth": output.depth(),
    }


def verify_p1b(candidate_root: Path | None = None) -> dict:
    root = candidate_root or Path(__file__).resolve().parents[2]
    start = time.perf_counter()
    reference = json.loads((root / P1B_REFERENCE).read_text())
    reference_sha = sha256_file(root / P1B_REFERENCE)
    source_hashes = {}
    source_errors = []
    for rel, expected in EXPECTED_SOURCE_SHA256.items():
        actual = sha256_file(root / rel)
        source_hashes[rel] = actual
        if actual != expected:
            source_errors.append(f"{rel}: {actual} != {expected}")

    earthquake = np.asarray(reference["earthquake_probabilities_25"], dtype=float)
    rain = np.asarray(reference["rain_probabilities_4"], dtype=float)
    annual = np.asarray(reference["annual_probabilities_100"], dtype=float)
    target_probability = float(reference["selected_event"]["probability"])

    preparation = build_state_preparation(earthquake, rain)
    event_oracle = build_event_oracle()
    zero_reflection = build_zero_reflection()
    flag_phase = build_flag_phase()
    operator = build_complete_operator(preparation, event_oracle)
    operator_dagger = reverse_adjoint(operator)
    grover = build_grover(operator, operator_dagger, zero_reflection, flag_phase)

    components = {
        "P": component_counts(preparation),
        "Uf": component_counts(event_oracle),
        "A": component_counts(operator),
        "A_dagger": component_counts(operator_dagger),
        "S0": component_counts(zero_reflection),
        "Sf": component_counts(flag_phase),
        "Q": component_counts(grover),
    }
    component_errors = []
    for label, expected in EXPECTED_COMPONENTS.items():
        for key, value in expected.items():
            if components[label][key] != value:
                component_errors.append(f"{label}.{key}: {components[label][key]} != {value}")

    checks = {
        "state_preparation": check_preparation(preparation, annual),
        "event_oracle": check_event_oracle(event_oracle),
        "target_flag_probability": check_flag_probability(operator, target_probability),
        "a_dagger_a": check_a_dagger_a(
            operator, operator_dagger, reverse_sequence_check(operator, operator_dagger), make_probes()
        ),
        "s0_sf": check_s0_and_sf(zero_reflection, flag_phase, build_data_only_reflection_mutant()),
        "event_mutant": check_event_mutant(event_oracle),
        "q_action": check_q_action(operator, grover, target_probability),
        "wrong_q_order": check_wrong_grover_order(
            operator, operator_dagger, zero_reflection, flag_phase, target_probability
        ),
        "clean_subspace": check_clean_subspace(
            operator, operator_dagger, zero_reflection, flag_phase, grover
        ),
        "budget": check_budget(
            {"P": components["P"], "A": components["A"], "Q": components["Q"]}
        ),
        "asymmetric_rain": None,
    }
    corrected = rain_probabilities_c_order(0.02, 0.04, 0.25)
    legacy = np.array([corrected[0], corrected[2], corrected[1], corrected[3]])
    checks["asymmetric_rain"] = check_asymmetric_rain(corrected, legacy)

    netlists = [
        _netlist_comparison(root, "P", preparation, False),
        _netlist_comparison(root, "Uf", event_oracle, False),
        _netlist_comparison(root, "A", operator, False),
        _netlist_comparison(root, "A_expanded", operator, True),
        _netlist_comparison(root, "A_dagger", operator_dagger, False),
        _netlist_comparison(root, "S0", zero_reflection, False),
        _netlist_comparison(root, "Sf", flag_phase, False),
        _netlist_comparison(root, "Q", grover, False),
        _netlist_comparison(root, "Q_expanded", grover, True),
    ]

    status_errors = [
        key for key, value in checks.items()
        if value.get("status") != "PASS"
    ]
    if not all(row["equal"] for row in netlists):
        status_errors.append("netlist_mismatch")
    if source_errors:
        status_errors.append("source_hash_mismatch")
    if component_errors:
        status_errors.append("component_count_mismatch")
    if reference_sha != P1B_REFERENCE_SHA256:
        status_errors.append("p1b_reference_hash_mismatch")

    elapsed = time.perf_counter() - start
    return {
        "status": "PASS" if not status_errors else "FAIL",
        "p1b_gatekeeper": "PASS_WITH_SCOPE_LIMITS",
        "p1b_review_zip_sha256": "fd3f1341f3d06ddbb12b3100685c12a82d99d4e00f61302c08a9166614507aeb",
        "p1b_gatekeeper_zip_sha256": "ff175086b80692b13933afe290050fd0e1500ba96c733df1dd52d96d5b7e536a",
        "reference_sha256": reference_sha,
        "source_hashes": source_hashes,
        "source_errors": source_errors,
        "components": components,
        "component_errors": component_errors,
        "checks": {key: value.get("status") for key, value in checks.items()},
        "check_details": checks,
        "netlists": netlists,
        "netlist_errors": [row["label"] for row in netlists if not row["equal"]],
        "status_errors": status_errors,
        "semantics_unchanged": not status_errors,
        "scope": {
            "official_model": "adb-base",
            "time_horizon": "ONE_YEAR",
            "target_event": "ANNUAL_BOTH_BONDS_EXHAUSTED",
            "target_index": int(reference["selected_event"]["annual_state_index"]),
            "target_probability": target_probability,
            "peak_qubits": 13,
            "finite_shot_qae": "NOT_RUN",
            "hardware_execution": "NOT_RUN",
            "quantum_advantage": "NOT_ESTABLISHED",
        },
        "resource": {
            "wall_s": elapsed,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_status": "RSS_MONITOR_ONLY",
        },
    }
