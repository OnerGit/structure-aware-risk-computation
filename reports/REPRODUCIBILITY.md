# Reproducibility

## Frozen environment

Recorded assembly environment: Python `3.12.14`, NumPy `2.5.4`, SciPy `1.18.1`,
Qiskit `2.6.2`, pytest `8.4.2`. The official `qjlb` source is not installed. It
is bound from an explicit, digest-verified source root by the solver and
checks.

A clean-environment reproduction is stronger evidence than reuse of the frozen
venv. The final check creates a new Python virtual environment installed only
from this repository's `requirements.txt`, with `PYTHONPATH=src` and no hidden
project cache. The result is recorded outside this repository in the project
audit.

## R1 pre-publication gate (2026-11-18)

A fresh Python `3.12.14` virtual environment installed from `requirements.txt`
produced `19 passed` in `1.23 s` for `PYTHONPATH=src python -m pytest tests`.
The seven-case path re-executed in `4.05 s` total measured wall time, with
`191,928 KiB` peak case RSS, and reproduced `7/7` problem IDs, `35/35` metrics
and `105/105` scoped comparisons. Negative controls rejected `16/16`, the
P1-B replay matched `9/9` netlists byte-for-byte, and the V1 exact reference
passed `25/25` rounded and `25/25` exact diagnostic cells. These runs re-verify
the frozen evidence; they are not a new benchmark. Full logs and the private
gate report are retained in the release audit, outside this repository.

AI tools are not runtime dependencies. OpenCode/DeepSeek and the ChatGPT web
interface assisted with engineering, review and editing; reproduction uses
only the pinned local environment and supplied official inputs. See
`../AI_USE_DISCLOSURE.md`.

## External inputs

Required:

- official archive SHA256
  `89d4139f809c316294b6115f115724b2af2304cba1992774e2d5086e49f7c79b`;
- extracted directory
  `adb-challenge-quantum-joint-loss-scenario-public-main/`;
- set `QJLB_SOURCE_ROOT` to that extracted directory;
- set `ADB_OFFICIAL_ARCHIVE` to the archive path.

The repository does not vendor the official source, its seven scenario configs
or `seismic.npz`/`flood.npz`. The official source archive's own root
`SHA256SUMS` lists 37 entries; `36/37` match and `.gitignore` is absent upstream.
Do not claim `37/37`.

## Checks

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q

PYTHONPATH=src python scripts/audit_official_source.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --output results/official_source_admission.json

WORK="$(mktemp -d /tmp/adb-v2-seven-cases.XXXXXX)"
PYTHONPATH=src python scripts/run_seven_cases.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --output "$WORK"

PYTHONPATH=src python scripts/run_negative_controls.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --results "$WORK"

PYTHONPATH=src python scripts/verify_candidate.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE"
```

`run_seven_cases.py` refuses to overwrite an existing output directory. Use a
fresh output path or preserve the prior run outside the repository.

Expected outcomes:

- tests `19 passed` (`18` frozen tests plus the statevector-memory audit);
- reference ledger `PASS`: `7` cases, `35` metrics, `5` DRB / `2` ELT;
- source admission `PASS`: archive digest, `7/7` problem IDs, `27` canonical
  paths, data `4/4`, source `12/12`, scenarios `7/7`;
- seven-case execution `PASS` (all official case and protocol checks): `5/5` DRB and `2/2` ELT through the
  official CLI contract, all five metrics within official tolerance;
- comparisons `105/105` (`35` metrics x `3` groups);
- negative controls `PASS`: `16/16` rejected;
- P1-B integration `PASS`: all checks and `9/9` netlists byte-identical;
- V1 exact DRB reference `PASS`: `25/25` rounded and `25/25` exact diagnostic;
- overall `PASS` (all checks).

The recorded seven-case process run used `4.34 s` total measured wall time and
peak external RSS `190,796 KiB`. The separate verifier used `1.54 s` and peak
RSS `218,376 KiB`. The run budget was `900 s`; resource use is monitor-only, not
enforced by a hard memory cap.

## Cross-environment note

Floating-point matrix exponentials, convolutions and sums can differ in their
last bits across NumPy/SciPy/BLAS builds. An independent environment using
Python `3.13`, NumPy `2.3.5` and SciPy `1.17.0` reproduced the original seven
public cases and `17/17` non-Qiskit tests; the original Qiskit integration test
was not run there because Qiskit was unavailable. Cross-environment acceptance
is tolerance-based, not bitwise hash equality. The frozen bytes remain the
provenance record for this repository.
