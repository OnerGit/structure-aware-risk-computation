"""ADB G1-A exact joint-loss benchmark reproduction.

Independent, exact implementation of the official joint-loss model
(ADB Quantum Innovation Challenge Brief, pp. 2-4): per-country compound
Poisson earthquake triggers on a capped 25-state CTMC, a four-state
precipitation law, a 100-state annual law, and three-year capped convolution.
"""

from __future__ import annotations

__version__ = "0.1.0"