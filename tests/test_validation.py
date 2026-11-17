"""Validate the blockwise implementation against dense linear algebra (small n).

Run:  python -m tests.test_validation
Checks (Appendix D.9 analogue):
  * Q is orthogonal (forward/inverse round trip, norm preservation);
  * blockwise log-likelihood == dense Cholesky log-likelihood, incl. far from theta0;
  * analytic gradient == dense trace formula and central finite differences;
  * Fisher information and the kurtosis matrix Dk == dense formulas.
"""
import math
import time

import numpy as np

from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel, dense_covariance, dense_loglik

CASES = [(3, 4), (3, 5), (4, 3), (4, 4), (5, 3), (3, 6), (2, 7)]
PARAMS = [(1.0, 0.4, 0.5, 0.25), (1.0, 0.4, 1.5, 0.25), (3.0, 0.05, 2.7, 0.01),
          (0.2, 1.9, 0.15, 5.0)]


def dense_grad_fisher(d, G, theta, y):
    C, dC = dense_covariance(d, G, theta)
    Ci = np.linalg.inv(C)
    a = Ci @ y
    g = np.array([-0.5 * (np.sum(Ci * dCp) - a @ dCp @ a) for dCp in dC])
    A = [Ci @ dCp for dCp in dC]
    I = np.array([[0.5 * np.sum(A[p] * A[q].T) for q in range(4)] for p in range(4)])
    diag = np.array([np.diag(Ap @ Ci) for Ap in A])
    return g, I, diag @ diag.T


def main():
    rng = np.random.default_rng(1)
    worst = dict(roundtrip=0, ll=0, grad=0, gradfd=0, fisher=0, kurt=0)
    for d, G in CASES:
        t0 = time.time()
        m = GridModel(d, G)
        y = rng.standard_normal(m.n)
        Z = m.sym.forward(y)
        worst["roundtrip"] = max(worst["roundtrip"], np.abs(m.sym.inverse(Z) - y).max(),
                                 abs(sum(np.sum(v * v) for v in Z.values()) - y @ y))
        for (s2, ell, nu, tau2) in PARAMS:
            th = theta_from_public(d, s2, ell, nu, tau2)
            yy = np.linalg.cholesky(dense_covariance(d, G, th)[0]) @ rng.standard_normal(m.n)
            Zy = m.sym.forward(yy)
            ll, g = m.loglik_grad(th, Zy)
            lld = dense_loglik(d, G, th, yy)
            gd, Id, Dd = dense_grad_fisher(d, G, th, yy)
            I, Dk = m.fisher(th, kurtosis_matrix=True)
            worst["ll"] = max(worst["ll"], abs(ll - lld) / max(1, abs(lld)))
            worst["grad"] = max(worst["grad"], np.abs(g - gd).max() / max(1, np.abs(gd).max()))
            worst["fisher"] = max(worst["fisher"], np.abs(I - Id).max() / np.abs(Id).max())
            worst["kurt"] = max(worst["kurt"], np.abs(Dk - Dd).max() / np.abs(Dd).max())
            h = 1e-5
            gfd = np.array([(m.loglik_grad(th + h * e, Zy, grad=False)
                             - m.loglik_grad(th - h * e, Zy, grad=False)) / (2 * h)
                            for e in np.eye(4)])
            worst["gradfd"] = max(worst["gradfd"], np.abs(g - gfd).max() / max(1, np.abs(g).max()))
        print(f"d={d} G={G} n={m.n}: blocks={m.sym.describe()['n_distinct_blocks']} "
              f"({time.time() - t0:.1f}s)", flush=True)
    print("worst relative errors:", {k: f"{v:.1e}" for k, v in worst.items()})
    tol = dict(roundtrip=1e-10, ll=1e-11, grad=1e-9, gradfd=1e-6, fisher=1e-9, kurt=1e-9)
    bad = {k: v for k, v in worst.items() if v > tol[k]}
    if bad:
        raise SystemExit(f"FAILED: {bad}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
