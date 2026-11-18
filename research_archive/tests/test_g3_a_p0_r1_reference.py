"""G3-A-P0-R1 — independent exact reference (R1-01) and phase-sensitive loader
verification (R1-02)."""

import numpy as np

from adb_g3_a_p0.grover import GroverCircuit
from adb_g3_a_p0 import r1_reference as R

SCEN = ("adb-base", "adb-independent", "adb-common-shock")


def test_independent_reference_production():
    for sc in SCEN:
        gc = GroverCircuit(sc)
        r = R.independent_reference(gc)
        assert r["pass"], (sc, r)
        assert r["good_set_match"] is True
        assert r["diff_circuit_a_vs_frozen"] <= 1e-12
        assert r["diff_independent_mass_vs_frozen"] <= 1e-12


def test_independent_decoder_does_not_use_production_helpers():
    # the decoder must be self-contained (no production imports for decode)
    import inspect
    src = inspect.getsource(R.independent_exceed_indicator)
    assert "n2_observable_vectors" not in src
    assert "terminal_code" not in src
    assert "frozen_n2_terminal" not in src


def test_loader_amplitudes_match_sqrt_p_global_phase():
    for sc in SCEN:
        gc = GroverCircuit(sc)
        amp = R.loader_amplitudes(gc)
        assert np.allclose(amp.imag, 0.0, atol=1e-12)          # real output
        w = R.target_amplitudes(gc)
        res = R.phase_check(amp.real, w, tol=1e-9)
        assert res["ok"], (sc, res)
        assert res["relative_sign_ok"] is True
        # explicit: every nonzero amplitude has the SAME sign vs w
        nz = np.abs(w) > 1e-300
        ratios = amp.real[nz] / w[nz]
        assert np.abs(ratios[0]) > 0.5
        assert np.allclose(ratios, ratios[0], atol=1e-9)


def test_adjoint_on_complex_superpositions_full_vector():
    for sc in SCEN:
        gc = GroverCircuit(sc)
        r = R.complex_adjoint_full_vector(gc, seed=20261107, tol=1e-9)
        assert r["all_pass"], (sc, r)


def test_sign_flip_negative_controls_detected():
    for sc in SCEN:
        gc = GroverCircuit(sc)
        w = R.target_amplitudes(gc)
        for mode in ("good_phase", "single"):
            bad = R.sign_flip_state(gc, mode)
            res = R.phase_check(np.asarray(
                [R.read_amplitude(bad, i) for i in range(128)]).real, w, tol=1e-9)
            assert res["ok"] is False, (sc, mode, res)
            assert res["relative_sign_ok"] is False, (sc, mode)


def test_prior_three_negative_controls_still_detected():
    from adb_g3_a_p0 import run as R2
    for sc in SCEN:
        gc = GroverCircuit(sc)
        nc = R2.negative_controls(gc, 1)
        assert nc["all_detected"] is True, (sc, nc)


def test_original_and_new_suite_coexist():
    import pathlib
    assert pathlib.Path("tests/test_g3_a_p0_circuit.py").is_file()
    assert pathlib.Path("tests/test_g3_a_p0_decompositions.py").is_file()
    assert pathlib.Path("tests/test_g3_a_p0_r1_reference.py").is_file()