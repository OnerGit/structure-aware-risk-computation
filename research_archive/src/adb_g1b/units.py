"""Unit conversions for reporting (raw probability / percentage / pp, USD)."""

from __future__ import annotations


def prob_to_pp(p: float) -> float:
    return p * 100.0


def prob_to_percent(p: float) -> float:
    return p * 100.0


def pp_to_prob(pp: float) -> float:
    return pp / 100.0


def usd_to_usd_m(v: float) -> float:
    return v / 1e6


def usd_to_usd_k(v: float) -> float:
    return v / 1e3