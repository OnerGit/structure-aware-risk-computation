# Evidence Summary

## Official source and seven-case execution

The official source archive was admitted by SHA256 and by an explicit canonical
path contract. Seven `problem_id` values reproduce exactly. All seven public
cases were executed through `python team_solver.py --request ... --output ...`
with the official `check_submission`, `strict_submission` and
`evaluation.tolerances` checks.

The solver reports `method=official_exact_classical_dp`,
`execution=classical_cpu`, `shots=0`, `qubits=0`, and actual
`model_evaluations`: `100` for each five DRB model and `421,968` for each ELT
model. The method label is broad. Scientifically, DRB is finite-state exact DP;
ELT is an exact catalogue row-weighted sum. This repository does not relabel
ELT as DP or either track as quantum.

## Metric correctness

- `35/35` estimates are present with official metric names and units.
- All `105/105` comparison cells pass official tolerance: `35/35` team vs
  official exact, `35/35` independent vs official exact, and `35/35` team vs
  independent. This is `35` metrics x `3` groups, not 105 independent trials.
- DRB independent check: frozen V1 Exact-DP, `25/25` rounded and `25/25` exact
  diagnostic cells.
- ELT independent check: direct NumPy NPZ row-sum; `10/10` ELT metric cells
  pass official tolerance.
- Negative controls: `16/16` rejected, with no tolerance relaxation.

## P1-B scoped quantum evidence

P1-B is a separate one-year annual ideal-circuit proof of concept for `adb-base`. It
was independently replayed at the component and netlist level with documented
scope limits:

- annual both-bonds-exhausted index `99`;
- independent official model target `1.879487424597577e-6`;
- emitted A-flag amplitude `1.8794874245975355e-6`, absolute error
  `4.1504614415460717e-20`;
- `13` peak logical qubits;
- `P=66`, `A=237`, `Q=658`, `A+3Q=2211`;
- `Q` expanded depth `345`, CCX-macro depth `141`;
- source and nine netlists unchanged.

It is not the seven-case solver and does not establish a three-year quantum
result. Its annual input is classically prepared and its ideal amplitudes are
not finite-shot QAE.

## Statevector scale

Raw complex128 storage is `2^n` amplitudes times `16` bytes. The 13-qubit
annual circuit uses `131,072` bytes (`128 KiB`) before simulator overhead. A
47-qubit dense state would require about `2 PiB` and is not simulated as a
statevector. `tests/test_statevector_memory.py` checks the declared values.

## Not established

No formal benchmark, private Holdout, ten-run ranking, geometry-score trial,
new quantum circuit in this revision, QPU execution, error-corrected resource
estimate, raw Poisson loader, quantum advantage or universal classical-hardness
result.
