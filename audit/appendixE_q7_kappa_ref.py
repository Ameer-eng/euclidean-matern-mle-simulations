"""APPENDIX E QUESTIONS (2026-11-16), Q7: magnitudes of the dense reference values in
tests/test_kappa.dense_checks, so that the relative-error denominators can be judged.

This repeats the dense comparison of tests/test_kappa.py exactly: same cases,
parameters, seed and draw order. Per (grid, parameter) it prints the reference
kappa_tilde, numerator and denominator, and each relative error with the test's
floor (kappa: max(1e-12, |k|); numerator: max(1e-300, |num|)).

  python -m audit.appendixE_q7_kappa_ref > audit/out/appendixE/kappa_dense_refs.txt
"""
import math

import numpy as np

from matern_sim import noise
from matern_sim.kappa import ResidualCumulant, dense_kappa
from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel, dense_covariance
from tests.test_kappa import CASES, PARAMS


def main():
    rng = np.random.default_rng(2)
    rows = []
    for d, G in CASES:
        m = GridModel(d, G)
        rc = ResidualCumulant(m)
        for (s2, ell, nu, tau2) in PARAMS:
            th = theta_from_public(d, s2, ell, nu, tau2)
            C = dense_covariance(d, G, th)[0]
            y = np.linalg.cholesky(C) @ rng.standard_normal(m.n)
            y += math.sqrt(tau2) * noise.sample("mixnorm", rng, m.n)
            res = rc.compute(th, m.sym.forward(y))
            k, num, den = dense_kappa(d, G, th, y)
            rows.append((d, G, m.n, nu, tau2, k, num, den,
                         abs(res["kappa"] - k) / max(1e-12, abs(k)),
                         abs(res["numer"] - num) / max(1e-300, abs(num)),
                         abs(res["denom"] - den) / den))
            print(f"d={d} G={G} n={m.n:4d} nu={nu:<4g} tau2={tau2:<5g} kappa_ref={k:+.4e} "
                  f"numer_ref={num:+.4e} denom_ref={den:.4e} | rel err kappa {rows[-1][8]:.1e} "
                  f"numer {rows[-1][9]:.1e} denom {rows[-1][10]:.1e}", flush=True)
    a = np.array([r[5:] for r in rows], float)
    print(f"\nmin |kappa_ref| = {np.abs(a[:, 0]).min():.3e}; min |numer_ref| = {np.abs(a[:, 1]).min():.3e}; "
          f"floors active: kappa {int((np.abs(a[:, 0]) < 1e-12).sum())}, numer {int((np.abs(a[:, 1]) < 1e-300).sum())}")
    print(f"worst: kappa {a[:, 3].max():.1e}, numer {a[:, 4].max():.1e}, denom {a[:, 5].max():.1e}")


if __name__ == "__main__":
    main()
