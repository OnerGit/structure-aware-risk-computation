from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from p1b_circuit import (
    ACTIVE_QUBITS,
    DATA_QUBITS,
    FLAG_QUBIT,
    SCRATCH_QUBITS,
    TARGET_DATA_INDEX,
    build_data_only_reflection_mutant,
    build_flag_mutant,
    build_rain_preparation,
    build_flag_phase,
    component_counts,
)

STATE_DIMENSION = 8192
ACTIVE_DIMENSION = 256
TOL = 1e-10


def basis_index(data: int, flag: int = 0, scratch: int = 0) -> int:
    return data + (flag << 7) + (scratch << 8)


def basis_circuit(data: int, flag: int = 0, scratch: int = 0) -> QuantumCircuit:
    circuit = QuantumCircuit(13)
    for qubit in DATA_QUBITS:
        if (data >> qubit) & 1:
            circuit.x(qubit)
    if flag:
        circuit.x(FLAG_QUBIT)
    for position, qubit in enumerate(SCRATCH_QUBITS):
        if (scratch >> position) & 1:
            circuit.x(qubit)
    return circuit


def active_vector(active: np.ndarray) -> np.ndarray:
    vector = np.zeros(STATE_DIMENSION, dtype=complex)
    vector[:ACTIVE_DIMENSION] = np.asarray(active, dtype=complex)
    return vector


def scratch_mass(state: np.ndarray) -> float:
    indices = np.arange(STATE_DIMENSION)
    mask = np.zeros(STATE_DIMENSION, dtype=bool)
    for qubit in SCRATCH_QUBITS:
        mask |= ((indices >> qubit) & 1) == 1
    return float(np.sum(np.abs(state[mask]) ** 2))


def flag_one_mass(state: np.ndarray) -> float:
    indices = np.arange(STATE_DIMENSION)
    return float(np.sum(np.abs(state[((indices >> FLAG_QUBIT) & 1) == 1]) ** 2))


def check_preparation(preparation: QuantumCircuit, annual_reference: np.ndarray) -> dict:
    state = Statevector(preparation).data[:128]
    probabilities = np.abs(state) ** 2
    expected = np.zeros(128, dtype=float)
    expected[:100] = annual_reference
    valid_error = float(np.max(np.abs(probabilities[:100] - annual_reference)))
    padding_error = float(np.max(np.abs(probabilities[100:])))
    amplitude_error = float(np.max(np.abs(state[:100] - np.sqrt(annual_reference))))
    norm_error = abs(float(np.sum(probabilities)) - 1.0)
    return {
        "valid_probability_max_abs_error": valid_error,
        "padding_probability_max_abs_error": padding_error,
        "amplitude_max_abs_error": amplitude_error,
        "norm_error": norm_error,
        "valid_states": 100,
        "padding_states": 28,
        "target_amplitude": [float(state[TARGET_DATA_INDEX].real), float(state[TARGET_DATA_INDEX].imag)],
        "target_probability": float(probabilities[TARGET_DATA_INDEX]),
        "probabilities_128": probabilities.tolist(),
        "status": "PASS"
        if max(valid_error, padding_error, amplitude_error, norm_error) <= TOL
        else "FAIL",
    }


def check_event_oracle(oracle: QuantumCircuit) -> dict:
    records = []
    max_leakage = 0.0
    max_value_error = 0.0
    for data in range(128):
        for flag in (0, 1):
            expected = basis_index(data, flag ^ int(data == TARGET_DATA_INDEX))
            initial = basis_circuit(data, flag)
            state = Statevector(initial).evolve(oracle).data
            leakage = 1.0 - float(abs(state[expected]) ** 2)
            value_error = abs(state[expected] - 1.0)
            max_leakage = max(max_leakage, leakage)
            max_value_error = max(max_value_error, float(value_error))
            records.append(
                {
                    "data": data,
                    "flag_in": flag,
                    "event": data == TARGET_DATA_INDEX,
                    "expected_index": expected,
                    "amplitude_real": float(state[expected].real),
                    "amplitude_imag": float(state[expected].imag),
                    "power_error": float(leakage),
                }
            )
    all_expected = all(
        row["amplitude_real"] == 1.0 and row["amplitude_imag"] == 0.0 for row in records
    )
    return {
        "cases": 256,
        "cases_passed": sum(
            1
            for row in records
            if abs(row["amplitude_real"] - 1.0) <= TOL
            and abs(row["amplitude_imag"]) <= TOL
        ),
        "max_power_leakage": max_leakage,
        "max_amplitude_error": max_value_error,
        "all_expected_basis_states": all_expected,
        "event_cases": [row for row in records if row["event"]],
        "records": records,
        "status": "PASS" if max_leakage <= TOL and max_value_error <= TOL else "FAIL",
    }


def check_flag_probability(operator: QuantumCircuit, target_probability: float) -> dict:
    state = Statevector(operator).data
    observed = flag_one_mass(state)
    return {
        "flag_one_probability": observed,
        "expected_target_probability": target_probability,
        "abs_error": abs(observed - target_probability),
        "scratch_mass": scratch_mass(state),
        "status": "PASS"
        if abs(observed - target_probability) <= TOL and scratch_mass(state) <= TOL
        else "FAIL",
    }


def reverse_sequence_check(operator: QuantumCircuit, dagger: QuantumCircuit) -> dict:
    if len(operator.data) != len(dagger.data):
        return {"status": "FAIL", "reason": "length mismatch"}
    for index, instruction in enumerate(operator.data):
        counterpart = dagger.data[len(operator.data) - 1 - index]
        original_qubits = [operator.find_bit(qubit).index for qubit in instruction.qubits]
        counterpart_qubits = [dagger.find_bit(qubit).index for qubit in counterpart.qubits]
        if original_qubits != counterpart_qubits:
            return {"status": "FAIL", "reason": f"qubit mismatch at {index}"}
        if instruction.operation.inverse() != counterpart.operation:
            return {"status": "FAIL", "reason": f"gate inverse mismatch at {index}"}
    return {
        "status": "PASS" if operator.inverse() == dagger else "FAIL",
        "gate_count": len(operator.data),
        "reason": "strict reversed adjoint sequence",
    }


def make_probes() -> list[dict]:
    probes = []
    for data, flag in ((0, 0), (0, 1), (99, 0), (99, 1), (100, 0), (127, 0), (127, 1)):
        probes.append(
            {
                "label": f"basis_data_{data}_flag_{flag}",
                "vector": active_vector(np.eye(ACTIVE_DIMENSION, dtype=complex)[data + 128 * flag]),
            }
        )
    rng = np.random.default_rng(20261116)
    for index in range(3):
        vector = rng.normal(size=ACTIVE_DIMENSION) + 1j * rng.normal(size=ACTIVE_DIMENSION)
        vector /= np.linalg.norm(vector)
        probes.append(
            {
                "label": f"random_active_superposition_{index}",
                "vector": active_vector(vector),
            }
        )
    return probes


def check_a_dagger_a(
    operator: QuantumCircuit,
    dagger: QuantumCircuit,
    reverse_check: dict,
    probes: list[dict],
) -> dict:
    round_trip = QuantumCircuit(13, name="A_dagger_A")
    round_trip.compose(operator, inplace=True)
    round_trip.compose(dagger, inplace=True)
    records = []
    for probe in probes:
        initial = Statevector(probe["vector"])
        after_a = initial.evolve(operator)
        after_round_trip = Statevector(after_a).evolve(dagger)
        error = float(np.max(np.abs(after_round_trip.data - initial.data)))
        records.append(
            {
                "label": probe["label"],
                "round_trip_max_abs_error": error,
                "scratch_mass_after_A": scratch_mass(after_a.data),
                "scratch_mass_after_A_dagger": scratch_mass(after_round_trip.data),
                "norm_after_A": float(np.linalg.norm(after_a.data)),
                "status": "PASS" if error <= TOL else "FAIL",
            }
        )
    round_trip_zero = Statevector(round_trip).data
    zero_amplitude_error = abs(round_trip_zero[0] - 1.0)
    zero_leakage = max(0.0, 1.0 - float(abs(round_trip_zero[0]) ** 2))
    return {
        "reverse_adjoint": reverse_check,
        "probe_count": len(probes),
        "probe_records": records,
        "all_probes_passed": all(row["status"] == "PASS" for row in records),
        "zero_round_trip_amplitude_error": float(zero_amplitude_error),
        "zero_round_trip_leakage": zero_leakage,
        "status": "PASS"
        if reverse_check["status"] == "PASS"
        and all(row["status"] == "PASS" for row in records)
        and zero_amplitude_error <= TOL
        and zero_leakage <= TOL
        else "FAIL",
    }


def check_s0_and_sf(
    zero_reflection: QuantumCircuit,
    flag_phase: QuantumCircuit,
    data_only_mutant: QuantumCircuit,
) -> dict:
    phase_records = []
    for label, bits in (
        ("zero_active", 0),
        ("data_q0", 1),
        ("data_q6", 64),
        ("flag", 128),
        ("one_data_plus_flag", 129),
        ("all_eight", 255),
    ):
        data = bits & 127
        flag = 1 if bits & 128 else 0
        index = basis_index(data, flag)
        initial = basis_circuit(data, flag)
        state = Statevector(initial).evolve(zero_reflection).data
        expected = -1.0 if bits == 0 else 1.0
        phase_records.append(
            {
                "label": label,
                "active_bits": bits,
                "amplitude_real": float(state[index].real),
                "expected_phase": expected,
                "scratch_mass": scratch_mass(state),
                "status": "PASS"
                if abs(state[index] - expected) <= TOL and scratch_mass(state) <= TOL
                else "FAIL",
            }
        )
    flag_zero = Statevector(basis_circuit(0, 0)).evolve(flag_phase).data
    flag_one = Statevector(basis_circuit(0, 1)).evolve(flag_phase).data
    flag_records = [
        {
            "label": "flag_zero",
            "amplitude": float(flag_zero[basis_index(0, 0)].real),
            "expected_phase": 1.0,
        },
        {
            "label": "flag_one",
            "amplitude": float(flag_one[basis_index(0, 1)].real),
            "expected_phase": -1.0,
        },
    ]
    mutant_initial = basis_circuit(0, 1)
    corrected = Statevector(mutant_initial).evolve(zero_reflection).data
    mutant = Statevector(mutant_initial).evolve(data_only_mutant).data
    corrected_phase = float(corrected[basis_index(0, 1)].real)
    mutant_phase = float(mutant[basis_index(0, 1)].real)
    mutation_distinguished = abs(corrected_phase - mutant_phase) > 0.5
    flag_phase_ok = abs(flag_zero[basis_index(0, 0)] - 1.0) <= TOL and abs(
        flag_one[basis_index(0, 1)] + 1.0
    ) <= TOL
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in phase_records)
        and flag_phase_ok
        and mutation_distinguished
        else "FAIL"
    )
    return {
        "s0_phase_records": phase_records,
        "sf_phase_records": flag_records,
        "s0_flag_omission_negative_control": {
            "input": "data=0, flag=1",
            "corrected_s0_phase": corrected_phase,
            "data_only_mutant_phase": mutant_phase,
            "distinguished": mutation_distinguished,
        },
        "status": status,
    }


def check_event_mutant(oracle: QuantumCircuit) -> dict:
    mutant = build_flag_mutant()
    target_initial = basis_circuit(TARGET_DATA_INDEX, 0)
    corrected = Statevector(target_initial).evolve(oracle).data
    wrong = Statevector(target_initial).evolve(mutant).data
    corrected_flag = flag_one_mass(corrected)
    mutant_flag = flag_one_mass(wrong)
    return {
        "input_data": TARGET_DATA_INDEX,
        "corrected_flag_one_probability": corrected_flag,
        "missing_negative_control_flag_one_probability": mutant_flag,
        "distinguished": corrected_flag - mutant_flag > 0.5,
        "status": "PASS" if corrected_flag - mutant_flag > 0.5 else "FAIL",
    }


def check_q_action(operator: QuantumCircuit, grover: QuantumCircuit, target_probability: float) -> dict:
    prepared = Statevector(operator).data
    after_q = Statevector(Statevector(prepared)).evolve(grover).data
    theta = math.asin(math.sqrt(target_probability))
    alpha_zero = np.zeros(STATE_DIMENSION, dtype=complex)
    for index in range(STATE_DIMENSION):
        if ((index >> FLAG_QUBIT) & 1) == 0:
            alpha_zero[index] = prepared[index]
    alpha_zero /= np.linalg.norm(alpha_zero)
    alpha_one = np.zeros(STATE_DIMENSION, dtype=complex)
    alpha_one[basis_index(TARGET_DATA_INDEX, 1)] = 1.0
    expected = -math.cos(3.0 * theta) * alpha_zero - math.sin(3.0 * theta) * alpha_one
    amplitude_error = float(np.max(np.abs(after_q - expected)))
    target_probability = float(np.sum(np.abs(after_q[basis_index(TARGET_DATA_INDEX, 1)]) ** 2))
    expected_target_probability = math.sin(3.0 * theta) ** 2
    return {
        "theta": theta,
        "global_phase_relation": "-standard_Grover",
        "amplitude_relation_max_abs_error": amplitude_error,
        "target_probability_after_Q": target_probability,
        "expected_target_probability_after_Q": expected_target_probability,
        "target_probability_abs_error": abs(target_probability - expected_target_probability),
        "scratch_mass_after_Q": scratch_mass(after_q),
        "status": "PASS"
        if amplitude_error <= TOL
        and abs(target_probability - expected_target_probability) <= TOL
        and scratch_mass(after_q) <= TOL
        else "FAIL",
    }


def check_wrong_grover_order(
    operator: QuantumCircuit,
    dagger: QuantumCircuit,
    zero_reflection: QuantumCircuit,
    flag_phase: QuantumCircuit,
    target_probability: float,
) -> dict:
    wrong = QuantumCircuit(13, name="Q_wrong_order_mutant")
    wrong.compose(flag_phase, inplace=True)
    wrong.compose(operator, inplace=True)
    wrong.compose(zero_reflection, inplace=True)
    wrong.compose(dagger, inplace=True)
    prepared = Statevector(operator).data
    after_wrong = Statevector(Statevector(prepared)).evolve(wrong).data
    theta = math.asin(math.sqrt(target_probability))
    alpha_zero = np.zeros(STATE_DIMENSION, dtype=complex)
    for index in range(STATE_DIMENSION):
        if ((index >> FLAG_QUBIT) & 1) == 0:
            alpha_zero[index] = prepared[index]
    alpha_zero /= np.linalg.norm(alpha_zero)
    alpha_one = np.zeros(STATE_DIMENSION, dtype=complex)
    alpha_one[basis_index(TARGET_DATA_INDEX, 1)] = 1.0
    expected = -math.cos(3.0 * theta) * alpha_zero - math.sin(3.0 * theta) * alpha_one
    error = float(np.max(np.abs(after_wrong - expected)))
    return {
        "wrong_order_amplitude_error": error,
        "distinguished": error > TOL,
        "status": "PASS" if error > TOL else "FAIL",
    }


def check_budget(counts: dict) -> dict:
    values = {
        "peak_logical_qubits": counts["Q"]["qubits"],
        "P_primitives": counts["P"]["expanded_primitives"],
        "A_primitives": counts["A"]["expanded_primitives"],
        "Q_primitives": counts["Q"]["expanded_primitives"],
        "Q3A_primitives": counts["A"]["expanded_primitives"] + 3 * counts["Q"]["expanded_primitives"],
    }
    caps = {
        "peak_logical_qubits": 13,
        "P_primitives": 100,
        "A_primitives": 300,
        "Q_primitives": 800,
        "Q3A_primitives": 2500,
    }
    return {
        "values": values,
        "caps": caps,
        "within_caps": all(values[key] <= caps[key] for key in caps),
        "status": "PASS" if all(values[key] <= caps[key] for key in caps) else "FAIL",
    }


def serialized_netlist(circuit: QuantumCircuit) -> list[dict]:
    rows = []
    for instruction in circuit.data:
        operation = instruction.operation
        params = []
        for parameter in operation.params:
            try:
                params.append(float(parameter).hex())
            except (TypeError, ValueError):
                params.append(str(parameter))
        rows.append(
            {
                "gate": operation.name,
                "qubits": [circuit.find_bit(qubit).index for qubit in instruction.qubits],
                "params_float_hex": params,
            }
        )
    return rows


def check_asymmetric_rain(rain_probabilities: np.ndarray, legacy_probabilities: np.ndarray) -> dict:
    corrected = Statevector(
        build_rain_preparation(rain_probabilities)
    ).data
    wrong = Statevector(
        build_rain_preparation(legacy_probabilities)
    ).data
    corrected_probabilities = np.abs(corrected) ** 2
    wrong_probabilities = np.abs(wrong) ** 2
    return {
        "corrected_input": np.asarray(rain_probabilities, dtype=float).tolist(),
        "legacy_input_under_same_index": np.asarray(legacy_probabilities, dtype=float).tolist(),
        "corrected_state_probabilities": corrected_probabilities.tolist(),
        "legacy_state_probabilities": wrong_probabilities.tolist(),
        "index_1_corrected_probability": float(corrected_probabilities[1]),
        "index_2_corrected_probability": float(corrected_probabilities[2]),
        "index_1_legacy_probability": float(wrong_probabilities[1]),
        "index_2_legacy_probability": float(wrong_probabilities[2]),
        "index_1_expected": float(rain_probabilities[1]),
        "index_2_expected": float(rain_probabilities[2]),
        "index_1_error": abs(float(corrected_probabilities[1]) - float(rain_probabilities[1])),
        "index_2_error": abs(float(corrected_probabilities[2]) - float(rain_probabilities[2])),
        "legacy_swaps_index_1_and_2": bool(abs(wrong_probabilities[1] - rain_probabilities[1]) > 1e-6),
        "status": "PASS"
        if abs(float(corrected_probabilities[1]) - float(rain_probabilities[1])) <= TOL
        and abs(float(corrected_probabilities[2]) - float(rain_probabilities[2])) <= TOL
        and abs(wrong_probabilities[1] - rain_probabilities[1]) > 1e-6
        else "FAIL",
    }


def check_clean_subspace(
    operator: QuantumCircuit,
    operator_dagger: QuantumCircuit,
    zero_reflection: QuantumCircuit,
    flag_phase: QuantumCircuit,
    grover: QuantumCircuit,
) -> dict:
    rng = np.random.default_rng(20261117)
    random_active = rng.normal(size=ACTIVE_DIMENSION) + 1j * rng.normal(size=ACTIVE_DIMENSION)
    random_active /= np.linalg.norm(random_active)
    initial = Statevector(active_vector(random_active))
    after_a = initial.evolve(operator)
    after_s0 = Statevector(after_a).evolve(zero_reflection)
    after_dagger = Statevector(after_a).evolve(operator_dagger)
    after_sf = Statevector(after_a).evolve(flag_phase)
    after_q = Statevector(after_a).evolve(grover)
    records = []
    for label, state in (
        ("A_random_clean", after_a),
        ("S0_random_clean", after_s0),
        ("A_dagger_random_post_A", after_dagger),
        ("Sf_random_post_A", after_sf),
        ("Q_random_post_A", after_q),
    ):
        records.append(
            {
                "label": label,
                "scratch_mass": scratch_mass(state.data),
                "norm": float(np.linalg.norm(state.data)),
                "status": "PASS"
                if scratch_mass(state.data) <= TOL and abs(float(np.linalg.norm(state.data)) - 1.0) <= TOL
                else "FAIL",
            }
        )
    return {
        "records": records,
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
    }
