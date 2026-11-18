from __future__ import annotations

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate

DATA_QUBITS = list(range(7))
FLAG_QUBIT = 7
SCRATCH_QUBITS = list(range(8, 13))
ACTIVE_QUBITS = list(range(8))
TARGET_DATA_INDEX = 99


def ucry_sequence(angles: list[float]) -> QuantumCircuit:
    count = len(angles)
    if count < 1 or count & (count - 1):
        raise ValueError("UCRY angle count must be a power of two")
    controls = int(math.log2(count))
    circuit = QuantumCircuit(controls + 1)
    circuit.append(UCRYGate([float(angle) for angle in angles]), range(controls + 1))
    decomposed = circuit.decompose(reps=2)
    unexpected = set(Counter(op.operation.name for op in decomposed.data)) - {"ry", "cx"}
    if unexpected:
        raise ValueError(f"unexpected UCRY decomposition: {sorted(unexpected)}")
    return decomposed


def ucry_angles(probabilities: np.ndarray, target_sequence: list[int]) -> list[list[float]]:
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(target_sequence)
    if probabilities.size != 2**n:
        raise ValueError("probability vector size does not match target sequence")
    layers = []
    for position, target in enumerate(target_sequence):
        controls = list(reversed(target_sequence[:position]))
        angles = []
        for prefix in range(2 ** len(controls)):
            p0 = 0.0
            p1 = 0.0
            for index, probability in enumerate(probabilities):
                if position > 0 and (index >> (n - position)) != prefix:
                    continue
                if (index >> (n - 1 - position)) & 1:
                    p1 += float(probability)
                else:
                    p0 += float(probability)
            if p0 + p1 <= 0.0:
                angles.append(0.0)
            else:
                angles.append(2.0 * math.atan2(math.sqrt(p1), math.sqrt(p0)))
        layers.append(angles)
    return layers


def add_preparation(
    circuit: QuantumCircuit,
    probabilities: np.ndarray,
    targets: list[int],
) -> None:
    layers = ucry_angles(probabilities, targets)
    for position, (angles, target) in enumerate(zip(layers, targets)):
        sequence = ucry_sequence(angles)
        controls = list(reversed(targets[:position]))
        circuit.compose(sequence, qubits=[target] + controls, inplace=True)


def build_state_preparation(
    earthquake_probabilities: np.ndarray,
    rain_probabilities: np.ndarray,
) -> QuantumCircuit:
    circuit = QuantumCircuit(7, name="P")
    earthquake = np.asarray(earthquake_probabilities, dtype=float)
    if earthquake.size != 25:
        raise ValueError("earthquake probability vector must contain 25 values")
    earthquake_full = np.zeros(32, dtype=float)
    earthquake_full[:25] = earthquake
    add_preparation(circuit, earthquake_full, [6, 5, 4, 3, 2])
    add_preparation(circuit, np.asarray(rain_probabilities, dtype=float), [1, 0])
    counts = Counter(op.operation.name for op in circuit.data)
    if counts != Counter({"ry": 34, "cx": 32}):
        raise ValueError(f"unexpected preparation counts: {counts}")
    return circuit


def build_rain_preparation(rain_probabilities: np.ndarray) -> QuantumCircuit:
    circuit = QuantumCircuit(2, name="P_rain")
    add_preparation(circuit, np.asarray(rain_probabilities, dtype=float), [1, 0])
    counts = Counter(op.operation.name for op in circuit.data)
    if counts != Counter({"ry": 3, "cx": 2}):
        raise ValueError(f"unexpected rain preparation counts: {counts}")
    return circuit


def controlled_x(
    circuit: QuantumCircuit,
    controls: list[int],
    target: int,
    scratch: list[int],
    compute_only: bool = False,
    uncompute_only: bool = False,
) -> None:
    if len(controls) < 2:
        raise ValueError("controlled_x requires at least two controls")
    expected_scratch = len(controls) - 2
    if len(scratch) != expected_scratch:
        raise ValueError("scratch width does not match controlled_x decomposition")
    if len(controls) == 2:
        circuit.ccx(controls[0], controls[1], target)
        return
    if not uncompute_only:
        circuit.ccx(controls[0], controls[1], scratch[0])
        for index, control in enumerate(controls[2:-1], start=1):
            circuit.ccx(scratch[index - 1], control, scratch[index])
        circuit.ccx(scratch[-1], controls[-1], target)
    if not compute_only:
        for index in range(len(scratch) - 1, 0, -1):
            circuit.ccx(scratch[index - 1], controls[index + 1], scratch[index])
        circuit.ccx(controls[0], controls[1], scratch[0])


def build_event_oracle(negative_controls: tuple[int, ...] = (2, 3, 4)) -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="Uf")
    for qubit in negative_controls:
        circuit.x(qubit)
    controlled_x(circuit, DATA_QUBITS, FLAG_QUBIT, SCRATCH_QUBITS)
    for qubit in negative_controls:
        circuit.x(qubit)
    return circuit


def build_zero_reflection() -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="S0")
    for qubit in ACTIVE_QUBITS:
        circuit.x(qubit)
    circuit.h(FLAG_QUBIT)
    controlled_x(circuit, DATA_QUBITS, FLAG_QUBIT, SCRATCH_QUBITS)
    circuit.h(FLAG_QUBIT)
    for qubit in ACTIVE_QUBITS:
        circuit.x(qubit)
    return circuit


def build_data_only_reflection_mutant() -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="S0_data_only_mutant")
    for qubit in DATA_QUBITS:
        circuit.x(qubit)
    circuit.h(6)
    controlled_x(circuit, [0, 1, 2, 3, 4, 5], 6, SCRATCH_QUBITS[:4])
    circuit.h(6)
    for qubit in DATA_QUBITS:
        circuit.x(qubit)
    return circuit


def build_flag_mutant() -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="Uf_missing_negative_control_mutant")
    for qubit in (2, 3):
        circuit.x(qubit)
    controlled_x(circuit, DATA_QUBITS, FLAG_QUBIT, SCRATCH_QUBITS)
    for qubit in (2, 3):
        circuit.x(qubit)
    return circuit


def build_flag_phase(negate: bool = False) -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="Sf")
    if negate:
        circuit.x(FLAG_QUBIT)
        circuit.z(FLAG_QUBIT)
        circuit.x(FLAG_QUBIT)
    else:
        circuit.z(FLAG_QUBIT)
    return circuit


def build_complete_operator(preparation: QuantumCircuit, oracle: QuantumCircuit) -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="A")
    circuit.compose(preparation, qubits=DATA_QUBITS, inplace=True)
    circuit.compose(oracle, inplace=True)
    return circuit


def reverse_adjoint(circuit: QuantumCircuit) -> QuantumCircuit:
    adjoint = QuantumCircuit(circuit.num_qubits, name=circuit.name + "_dagger")
    for instruction in reversed(circuit.data):
        adjoint.append(
            instruction.operation.inverse(),
            [circuit.find_bit(qubit).index for qubit in instruction.qubits],
        )
    return adjoint


def build_grover(
    operator: QuantumCircuit,
    operator_dagger: QuantumCircuit,
    zero_reflection: QuantumCircuit,
    flag_phase: QuantumCircuit,
) -> QuantumCircuit:
    circuit = QuantumCircuit(13, name="Q")
    circuit.compose(flag_phase, inplace=True)
    circuit.compose(operator_dagger, inplace=True)
    circuit.compose(zero_reflection, inplace=True)
    circuit.compose(operator, inplace=True)
    return circuit


def component_counts(circuit: QuantumCircuit) -> dict:
    operations = Counter(op.operation.name for op in circuit.data)
    one_qubit = 0
    cx = 0
    toffoli = 0
    t_count = 0
    tdg_count = 0
    for instruction in circuit.data:
        name = instruction.operation.name
        if name == "ccx":
            toffoli += 1
            one_qubit += 9
            cx += 6
            t_count += 4
            tdg_count += 3
        elif name in {"x", "h", "z", "ry", "rz", "t", "tdg", "u"}:
            one_qubit += 1
        elif name in {"cx", "cz"}:
            cx += 1
        else:
            raise ValueError(f"unsupported primitive: {name}")
    expanded_depth = circuit.decompose(gates_to_decompose=["ccx"]).depth()
    return {
        "operations": dict(sorted(operations.items())),
        "one_qubit_expanded": one_qubit,
        "cx_expanded": cx,
        "toffoli_macro": toffoli,
        "expanded_primitives": one_qubit + cx,
        "t_count_toffoli_macros": t_count,
        "tdg_count_toffoli_macros": tdg_count,
        "depth_ccx_macro": circuit.depth(),
        "depth_expanded": expanded_depth,
        "size": circuit.size(),
        "qubits": circuit.num_qubits,
    }
