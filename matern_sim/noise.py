"""Standardised measurement-error laws (E xi = 0, E xi^2 = 1), all in the
uniformly sub-Gaussian class of Assumption 4 with E xi^4 - 1 > 0.

  gauss    N(0,1)                                 E xi^4 = 3      (kappa4 = 0)
  unif     Uniform(-sqrt3, sqrt3)                 E xi^4 = 1.8    (platykurtic)
  mixnorm  0.9 N(0,s^2) + 0.1 N(0,9 s^2), s^2=1/1.8  E xi^4 = 8.33   (leptokurtic, symmetric)
  bern     (B - p)/sqrt(p q), B ~ Bernoulli(0.1)  E xi^4 = 8.11   (skewed, discrete)

mixnorm and bern have nearly the same fourth moment but very different shapes,
so their nugget limits should coincide (only E xi^4 enters Theorem 3.1).
"""
from __future__ import annotations

import math

import numpy as np

_P_BERN = 0.1
_MIX_W, _MIX_R = 0.1, 9.0          # contamination weight, variance ratio
_MIX_S = 1.0 / math.sqrt(1.0 - _MIX_W + _MIX_W * _MIX_R)


def _moments():
    pq = _P_BERN * (1 - _P_BERN)
    mix4 = 3.0 * (1 - _MIX_W + _MIX_W * _MIX_R**2) * _MIX_S**4
    return {
        "gauss": dict(m3=0.0, m4=3.0),
        "unif": dict(m3=0.0, m4=1.8),
        "mixnorm": dict(m3=0.0, m4=mix4),
        "bern": dict(m3=(1 - 2 * _P_BERN) / math.sqrt(pq), m4=(1 - 3 * pq) / pq),
    }


MOMENTS = _moments()
LAWS = tuple(MOMENTS)
LAW_ID = {name: i for i, name in enumerate(LAWS)}


def sample(law, rng, n):
    if law == "gauss":
        return rng.standard_normal(n)
    if law == "unif":
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), n)
    if law == "mixnorm":
        scale = np.where(rng.random(n) < _MIX_W, math.sqrt(_MIX_R), 1.0) * _MIX_S
        return scale * rng.standard_normal(n)
    if law == "bern":
        b = (rng.random(n) < _P_BERN).astype(float)
        return (b - _P_BERN) / math.sqrt(_P_BERN * (1 - _P_BERN))
    raise ValueError(f"unknown law {law!r}")


def kappa4(law, tau2):
    """Fourth cumulant of eps = tau xi:  tau^4 (E xi^4 - 3)."""
    return tau2**2 * (MOMENTS[law]["m4"] - 3.0)
