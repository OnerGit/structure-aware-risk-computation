# Structure-Aware Risk Computation — Joint-Loss Risk Modelling Research Prototype

**Repository:** `structure-aware-risk-computation`  
**ADB Quantum Innovation Challenge 2026 · Phase 1**  
**Team:** Quantum Oner · **Research lead:** YONGJIAN, JIN

This repository provides an independently developed lightweight prototype and
reproducibility evidence for **correlated disaster-loss modelling**. It covers
three complementary scientific contributions:

1. **Strong classical reference:** exact three-year disaster-relief-bond (DRB)
   calculations and annual World Bank event-loss-table (ELT) reference sums,
   using the official public request/response interface.
2. **Route A — Gate-based quantum:** two separately validated, scoped proofs
   of concept: a verified one-year annual-factor reversible circuit and a
   historical terminal-distribution loading construction.
3. **Route B — Quantum-inspired tensor networks:** exact finite-model
   contraction and bounded TT/MPS truncation studies, with archived source,
   synthetic fixtures and recorded results.

The release is **classical-first**: the official benchmark and the strong
classical reference are the evidential baseline. The two quantum PoCs and the
TN experiments address **different targets and horizons**; they are not
head-to-head performance benchmarks. Phase 2 is a **conditional research plan
only**, not a completed result or a promise of quantum advantage.

## Phase 1 evidence at a glance

| Evidence | Result | Scope |
|---|---|---|
| Official classical benchmark | **7/7 cases; 35/35 metrics; 105/105 scoped comparisons** | 5 three-year DRB + 2 annual ELT, classical CPU |
| Annual reversible circuit | **13 logical qubits; P/A/Q = 66/237/658 expanded primitives; Q depth = 345** | One-year diagnostic event under the official `adb-base` annual law; ideal circuit |
| Historical terminal-distribution circuit | **13 logical qubits; A = 20,587; Q = 56,808 expanded gates; Q depth = 33,416** | Three-year exceedance event; precomputed classical Exact-DP terminal probabilities |
| Exact TN calculation | **25/25 official DRB metrics** | Finite two-country model, Exact-DP-equivalent result |
| Synthetic n3 cases | **F1 base, F2 independent, F3 common-shock** | Synthetic three-country diagnostics, not official three-country benchmarks |
| Standard TT-SVD | **54 scoped units** | 30 base-scenario metric/χ cells + 24 exceed-only cells in four further scenarios; finite-size |

The **105** comparisons are three scoped checks for each of **35** financial
metric cells, not 105 independent experiments. All numbers above come from
retained frozen evidence. The R1 pre-publication regression re-executed the
public path and reproduced them; it did not create a new scientific benchmark.

## Repository map

```text
src/                      Current seven-case reference and annual-circuit source
team_solver.py            Official request/response interface adapter
scripts/                  Input checks, seven-case executor, negative controls
scenarios/                Frozen public-case reference metadata (no official data)
tests/                    Deterministic reference and integration tests
results/                  Recorded official-case outputs, logs and QASM/netlists
evidence/p1b/             Original annual-circuit verification records
reports/                  Reproducibility and scope documentation
research_archive/src/     Frozen historical quantum/TN research modules
research_archive/results/ Recorded historical quantum/TN numerical results
research_archive/tests/   Historical verification test source
docs/                     Phase 1 scientific scope and Phase 2 research plan
THIRD_PARTY_NOTICES.md    Attributions and external-data rights
THIRD_PARTY_LICENSES/     Scoped upstream licence text (Apache-2.0)
AI_USE_DISCLOSURE.md      Disclosure of AI-assisted development and review
LICENSE_STATUS.md         Status of original-code licensing
SHA256SUMS.txt            Exact file hashes for this release candidate
```

## Environment and execution

**Recorded environment:** Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1,
Qiskit 2.6.2 and pytest 8.4.2. CPU-first classical execution and ideal
statevector simulation; **no physical QPU** was used. The one-year 13-qubit
statevector has 8,192 complex amplitudes (128 KiB of raw complex128 storage,
excluding simulator overhead). Resource metrics in `results/` are local
measurements, not hardware acceleration or runtime advantage.

**R1 pre-publication regression (2026-11-18):** a fresh Python 3.12.14
virtual environment installed from `requirements.txt` passed `19/19` repository
tests. The official seven-case path was re-executed against the hash-verified
external official inputs and reproduced `7/7` problem IDs, `35/35` metrics and
`105/105` scoped comparisons; negative controls rejected `16/16`, and the
P1-B netlist replay matched `9/9`. Full logs and the gate report are retained
in the private release audit, outside this repository.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q
```

### Official input requirement

The upstream official public benchmark source and World Bank ELT `.npz` files
are **not included** in this release. Obtain the official benchmark materials
from the challenge, verify the required archive SHA256, and then set:

```bash
export QJLB_SOURCE_ROOT=/absolute/path/to/adb-challenge-quantum-joint-loss-scenario-public-main
export ADB_OFFICIAL_ARCHIVE=/absolute/path/to/verified-official-archive.zip
```

Expected original official archive SHA256:
`89d4139f809c316294b6115f115724b2af2304cba1992774e2d5086e49f7c79b`.
The upstream source must provide `src/qjlb/model.py`, `scenarios/` and `data/`.
The scripts validate input hashes before use.

```bash
PYTHONPATH=src python scripts/audit_official_source.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --output /tmp/adb-official-audit.json

WORK="$(mktemp -d /tmp/adb-seven-cases.XXXXXX)"
PYTHONPATH=src python scripts/run_seven_cases.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --output "$WORK"

PYTHONPATH=src python scripts/run_negative_controls.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE" \
  --results "$WORK"

PYTHONPATH=src python scripts/verify_candidate.py \
  --source-root "$QJLB_SOURCE_ROOT" --archive "$ADB_OFFICIAL_ARCHIVE"
```

The expected outputs are the **recorded** frozen results. Reproducing them
requires the verified external official inputs and compatible dependencies.
See `reports/REPRODUCIBILITY.md` for per-check acceptance and recorded resource
measurements. The historical research materials in `research_archive/` are
separately described in `research_archive/README.md` and do not form part of
the official seven-case execution path.

## Reproducible evidence and known research scope

- Exact-DP is efficient on the official small two-country DRB models; the
  seven-case outputs are **classical**, not seven quantum solutions.
- The one-year diagnostic circuit is not the complete three-year contract.
  Its annual risk factors and preparation angles still require classical work.
- The terminal-loading PoC uses precomputed Exact-DP terminal probabilities.
- The historical exact TN contraction is DP-equivalent. TT-SVD results are
  finite-size and metric-scoped; no universal low-rank result is claimed.
- The candidate multi-year generative architecture, direct factor-level TN
  scaling, finite-shot QAE, and hardware studies remain future research.
- No end-to-end quantum or TN scaling advantage has been established.

For precise experimental scope and Phase 2 milestones, read
`docs/SCIENTIFIC_SCOPE.md` and `docs/PHASE2_RESEARCH_PLAN.md`.

## Third-party rights and AI assistance

Only original project code, tests, documented derived outputs and public
reference metadata are packaged. We do **not** redistribute the official ADB
source, official challenge brief or World Bank NPZ datasets. Refer to
`THIRD_PARTY_NOTICES.md` for data attributions. Our original code has no
selected open-source license (`LICENSE_STATUS.md`).

AI tools assisted with implementation, mathematical review, verification and
editing; the human researcher retained scientific decision-making and final
accountability. See `AI_USE_DISCLOSURE.md`. AI tools are **not** runtime
dependencies or additional team members.

This is an independent research prototype, not an ADB publication or an
endorsement of any kind.
