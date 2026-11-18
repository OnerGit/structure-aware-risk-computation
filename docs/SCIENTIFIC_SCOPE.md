# Scientific Scope and Evidence Boundaries

**Repository:** `structure-aware-risk-computation`

**Submission narrative:** correlated disaster-loss risk models, a strong
classical benchmark first, two separately scoped and validated quantum
proof-of-concepts, independent TN/MPS research, and a conditional Phase 2
programme. This file maps the frozen Phase 1 evidence to the public prototype;
it does not replace the separately submitted Technical Proposal and does not
introduce new experiments. The R1 gate re-executed the recorded checks on
2026-11-18; it did not add a new benchmark, sweep or scientific result.

## Official reference (classical)

- Five ADB disaster-relief-bond scenarios, each with five **three-year** risk
  metrics. Exact-DP and reference comparison are classical CPU calculations.
- Two World Bank ELT cases with five **annual rate-weighted** metrics each;
  different units and interpretation from the DRB cases.
- 35 financial cells, 105 scoped comparisons (three sets of 35), 16 negative
  controls rejected in the frozen S3 execution.

## Route A — reversible circuit PoCs

- **Terminal-distribution loading:** 13 logical qubits; A=20,587 and Q=56,808
  expanded gates; Q depth=33,416. The event concerns a three-year terminal-law
  exceedance probability under the scenario-specific reference scope, with
  classical Exact-DP input preprocessing.
- **Annual-factor preparation:** official `adb-base` **one-year** law, 13
  logical qubits; expanded P/A/Q gates 66/237/658, Q depth 345; diagnostic
  both-bonds-exhausted event probability approximately 1.8794874e-6. Frozen
  proof covers state preparation, adjoints, event flagging, reflection and
  ancilla cleanup in an ideal simulator. It is not the official three-year
  contract probability.
- **Proposed:** multi-year coherent-factor construction, reversible capped
  loss accumulation and contract predicates; not yet an integrated solver.

## Route B — TN/MPS bounded investigations

- Exact contraction of the two-country annual probability model matches the
  Exact-DP reference for 25/25 official DRB metric cells; maximum normalized
  absolute error recorded about 1.39e-17.
- Three **synthetic** three-country fixtures (base, independent, common-shock)
  are verified against separately constructed DP references; they are not
  official three-country ADB benchmark cases.
- Standard TT-SVD sweep comprises 54 scoped units: all five metrics on
  `adb-base` across six chi settings, plus the exceed metric on each of four
  other DRB scenarios across six chi settings. The chi=4 aggregate maximum
  metric absolute error in that scope is about 1.17e-15.
- Exact current TN is DP-equivalent. Later work will measure direct factor-level
  construction, payoff-coupled rank growth, truncation and complete costs.

## Comparisons and future claims

Current PoCs have **different observables and horizons**, so no direct
head-to-head numerical ranking is valid. The Phase 2 plan uses the same
financial laws, targets, accuracy and end-to-end cost definition for future
matched experiments. Neither quantum advantage nor general TN scaling
advantage has been established.
