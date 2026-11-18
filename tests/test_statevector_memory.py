KIB = 1024
MIB = 1024**2
GIB = 1024**3
PIB = 1024**5


def raw_complex128_bytes(n_qubits: int) -> int:
    return (1 << n_qubits) * 16


def test_raw_complex128_statevector_memory_table():
    expected = {
        7: (128, 2 * KIB),
        13: (8192, 128 * KIB),
        24: (16_777_216, 256 * MIB),
        30: (1_073_741_824, 16 * GIB),
        32: (4_294_967_296, 64 * GIB),
        47: (140_737_488_355_328, 2 * PIB),
    }
    for n_qubits, (amplitudes, memory_bytes) in expected.items():
        assert 2**n_qubits == amplitudes
        assert raw_complex128_bytes(n_qubits) == memory_bytes
    assert raw_complex128_bytes(13) == 131_072
