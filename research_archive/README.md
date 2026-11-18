# Frozen Historical Research Snapshots — Route A and Route B

This folder contains **unaltered historical source and result files** copied
from the frozen research review package `ADB_M3_S1_R1_20261111_131354_REVIEW.zip`.
The original file identities are recorded in the local packaging audit; they
have not been refactored or merged into the main `src/` tree.

- `src/adb_g3_a_p0/`: reversible terminal-distribution loading PoC and
  Grover/exceedance circuit constructions.
- `src/adb_g3_b_p0/`: annual-factor TN contractions and corrected standard
  TT-SVD diagnostics, including synthetic F1/F2/F3 closeout.
- `src/adb_g1a`, `src/adb_g1b`, `src/adb_g2_s2_s1a`: read-only historical
  reference and annual-law dependencies needed to interpret the archived
  studies.
- `src/adb_g2_s2_s1b/`: the two G2-S2-S1B files needed by the archived
  negative-control tests (`__init__.py`, `mathdef.py`), restored unmodified
  from the original research source tree and verified against the frozen
  review package `MANIFEST.sha256`. Their SHA256 identities are
  `28336926c31a559266db75ab9f9e6c77955cca8d15f4236aebb86a55d722c56a`
  and
  `7b5ea7dc1c3b10a574c17fde9242ce011e5c14458f6a678487557d69b508cdd6`,
  matching both the package `MANIFEST.sha256` and the original research
  source tree. The other G2-S2-S1B modules (circuits, IQAE, resources and
  experiments) are outside this archive scope and are not included.
- `results/g3_a_p0` and `results/g3_b_p0`: original results, fixture data,
  resource records and chi=4 / 54-unit standard TT-SVD evidence.
- `tests/`: historical source of the corresponding verification tests.

The archived files are *not* replacements for the primary seven-case CLI
and annual-factor circuit path. To inspect historical imports, use a separate
process and make the local `research_archive/src` the sole `PYTHONPATH`.
For repeatability, tests may require the original frozen compatible numerical
environment; the values in `results/` are prior frozen outputs and no new
scientific runs were performed during package assembly.

The historical 54-unit TT-SVD result is a **post-factor-construction**, finite
model benchmark, not a complete factor-level scalable TN implementation.
Historical source filenames contain version labels strictly for technical
provenance, not as multiple competing public solutions.

**Historical test status (R2 packaging correction):** the four historical
negative-control tests that import `adb_g2_s2_s1b.mathdef` are now runnable.
With the restored files in place, all `47` historical tests pass under the
frozen compatible environment. This restoration adds only the two frozen
module files; it does not alter any existing archived source, test or result.
The public seven-case and P1-B paths remain independent of this archive.
