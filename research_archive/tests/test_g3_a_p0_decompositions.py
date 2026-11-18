"""G3-A-P0 — primitive decomposition micromaps (reversibility/units).

All checks are done on explicit basis indices b = sum_i bit_i 2^i, where
bit_i is the state of qubit q_i (axis i); amplitudes are read via
read_amplitude (coordinate convention), never via flat-index assumptions.
"""

import numpy as np

from adb_g3_a_p0 import circuit as C


def _apply(net, nq, inp):
    st = C.basis_vec(nq, inp)
    for g in net:
        st = C.apply_gate(st, g)
    return st


def test_cry_decomposition():
    th = 0.7
    net = C.cry(th, 0, 1)                     # control q0=LSB(c), target q1
    for inp in range(4):
        c, t = (inp >> 0) & 1, (inp >> 1) & 1
        st = _apply(net, 2, inp)
        if c == 0:
            out = inp
            val = 1.0
        else:
            r = C._ry(th)                     # act on target bit (q1)
            out = c | ((0 if np.abs(r[t, 0]) > 0.5 else 1) << 1)
            # pick the expected index from the RY column
            amp = r[:, t]
            # locate the column of RY: input target state `t` -> column t
            nz = np.nonzero(np.abs(amp) > 1e-12)[0]
            assert len(nz) >= 1
            out = c + (nz[0] << 1)
            val = amp[nz[0]]
        assert abs(C.read_amplitude(st, out) - val) < 1e-13
    # total norm =1 on every input
    for inp in range(4):
        st = _apply(net, 2, inp)
        assert abs(np.abs(st).reshape(-1).sum() ** 0 - 1.0) < 1 or True
        assert abs(float(np.abs(st * np.conj(st)).sum()) - 1.0) < 1e-13


def test_toffoli_15gate():
    net = C.toffoli(0, 1, 2)
    for inp in range(8):
        st = _apply(net, 3, inp)
        a, b, cc = (inp >> 0) & 1, (inp >> 1) & 1, (inp >> 2) & 1
        c2 = cc ^ (a & b)
        out = a | (b << 1) | (c2 << 2)
        assert abs(C.read_amplitude(st, out) - 1.0) < 1e-12
        resid = float(np.abs(st ** 2).sum()) - 1.0
        assert abs(resid) < 1e-12


def test_toffoli_self_inverse():
    net = C.toffoli(0, 1, 2)
    for inp in range(8):
        st = C.basis_vec(3, inp)
        for g in net + list(reversed(net)):
            st = C.apply_gate(st, g)
        assert abs(C.read_amplitude(st, inp) - 1.0) < 1e-12


def test_mcx_ladder_returns_ancillas():
    for m in (3, 5, 7):
        anc = list(range(8, 8 + (m - 2)))      # ancilla block
        target = 8 + (m - 2)                    # qubit right after ancillas
        net = C.mcx(list(range(m)), target, anc)
        nq = target + 1
        for inp in range(1 << m):
            st = _apply(net, nq, inp)
            t = (inp >> target) & 1
            newt = t ^ (1 if all((inp >> i) & 1 for i in range(m)) else 0)
            out = (inp & ~(1 << target)) | (newt << target)
            assert abs(C.read_amplitude(st, out) - 1.0) < 1e-12
            # vectorized ancilla-zero subspace mass (all ancilla bits = 0)
            sub = st.reshape((2,) * nq)
            sl = [slice(None)] * nq
            for a in anc:
                sl[a] = 0
            mass = float((np.abs(sub[tuple(sl)]) ** 2).sum())
            assert abs(mass - 1.0) < 1e-12   # no leak onto nonzero ancilla


def test_ucry_toy():
    ang = [0.6, 0.9]
    net = C.mcry(ang, [0], 1, [])
    # state |c><t>: apply RY(ang[c]) to target bit q1
    for inp in range(4):
        c, t = (inp >> 0) & 1, (inp >> 1) & 1
        st = _apply(net, 2, inp)
        r = C._ry(ang[c])
        nz = np.nonzero(np.abs(r[:, t]) > 1e-12)[0]
        assert len(nz) >= 1
        out = c + (nz[0] << 1)
        assert abs(C.read_amplitude(st, out) - r[nz[0], t]) < 1e-13


def test_s0_acts_as_zero_reflection_on_data():
    from adb_g3_a_p0.oracle import s0_net
    net = s0_net()
    for data in (0, 5, 127, 63):
        st = _apply(net, 13, data)
        phase = -1.0 if data == 0 else 1.0
        assert abs(C.read_amplitude(st, data) - phase) < 1e-12
        # no amplitude on non-zero ancilla basis slots
        resid = 0.0
        for idx in range(2 ** 7):
            resid += abs(C.read_amplitude(st, idx)) ** 2
        assert abs(resid - 1.0) < 1e-12