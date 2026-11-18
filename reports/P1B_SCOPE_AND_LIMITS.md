# P1-B Scope and Limits

P1-B addresses only the final annual event indicator for the official
`adb-base` input year. It is a gate-level reversible construction whose state
preparation consumes a classically computed annual `EQ25` factor and a
four-state rain law. The circuit does not construct those classical inputs from
raw event-loss tables, and it does not compute all three annual years or all
five DRB metrics.

The verified claims are:

- the intended annual target amplitude is prepared exactly in the ideal
  statevector model;
- the event oracle distinguishes the designated annual both-exhausted event
  from its negative-control mutant, including both flag-input branches;
- the phase, state-preparation, event-oracle and Q-block identities hold within
  the tested numerical tolerance;
- `A` is a reversible encoding and `A†` is its strict reversed adjoint;
- the corrected row-major rain order does not swap indices 1 and 2;
- all checked netlists regenerate byte-for-byte;
- component and expanded-gate budgets hold.

The unverified claims are finite-shot estimation precision, sample efficiency,
physical noise resilience, a hardware-native resource estimate, a raw quantum
Poisson loader, end-to-end three-year quantum execution, and any speedup over
classical methods. The result is evidence for a scoped circuit construction,
not a complete quantum solution of the ADB benchmark.
