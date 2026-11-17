"""Validate the residual fourth-cumulant estimator (eq. A.54) against dense algebra.

Run:  python -m tests.test_kappa
Checks, on the small grids of tests/test_validation.py and at parameter values
including ones far from theta0:
  * numerator, denominator sum_ij A_ij^4 and kappa_tilde == dense computation;
  * the orbit-row diagonal == the grid diagonal of C^{-1} (internal consistency);
  * the log-likelihood returned for the data check == dense log-likelihood;
and, by Monte Carlo at the true covariance, that kappa_tilde is unbiased for the
fourth cumulant of each error law (Lemma A.12, eq. C.34).
"""
import math
import time

import numpy as np

from matern_sim import config, noise
from matern_sim.kappa import ResidualCumulant, dense_kappa
from matern_sim.kernel import theta_from_public
from matern_sim.likelihood import GridModel, dense_covariance, dense_loglik

CASES = [(3, 4), (3, 5), (4, 3), (4, 4), (5, 3), (3, 6), (2, 7), (2, 8)]
PARAMS = [(1.0, 0.4, 0.5, 0.25), (1.0, 0.4, 1.5, 0.25), (3.0, 0.05, 2.7, 0.01),
          (0.2, 1.9, 0.15, 5.0)]


def dense_checks():
    rng = np.random.default_rng(2)
    worst = dict(numer=0.0, denom=0.0, kappa=0.0, ll=0.0, diag=0.0)
    for d, G in CASES:
        t0 = time.time()
        m = GridModel(d, G)
        rc = ResidualCumulant(m)
        for (s2, ell, nu, tau2) in PARAMS:
            th = theta_from_public(d, s2, ell, nu, tau2)
            C = dense_covariance(d, G, th)[0]
            # non-Gaussian data so that the numerator is not ~0
            y = np.linalg.cholesky(C) @ rng.standard_normal(m.n)
            y += math.sqrt(tau2) * noise.sample("mixnorm", rng, m.n)
            res = rc.compute(th, m.sym.forward(y))
            k, num, den = dense_kappa(d, G, th, y)
            worst["numer"] = max(worst["numer"], abs(res["numer"] - num) / max(1e-300, abs(num)))
            worst["denom"] = max(worst["denom"], abs(res["denom"] - den) / den)
            worst["kappa"] = max(worst["kappa"], abs(res["kappa"] - k) / max(1e-12, abs(k)))
            lld = dense_loglik(d, G, th, y)
            worst["ll"] = max(worst["ll"], abs(res["loglik"] - lld) / max(1, abs(lld)))
            worst["diag"] = max(worst["diag"], res["diag_check"])
        print(f"d={d} G={G} n={m.n}: orbits={len(rc.rep_lin)} ({time.time() - t0:.1f}s)",
              flush=True)
    print("worst relative errors:", {k: f"{v:.1e}" for k, v in worst.items()})
    tol = dict(numer=1e-9, denom=1e-11, kappa=1e-9, ll=1e-11, diag=1e-11)
    return {k: v for k, v in worst.items() if v > tol[k]}


def unbiasedness(d=3, G=8, nu0=1.5, nrep=300):
    """At theta_hat = theta0 the estimator is exactly unbiased (eq. C.34)."""
    m = GridModel(d, G)
    rc = ResidualCumulant(m)
    th = theta_from_public(d, config.TRUTH["sigma2"], config.TRUTH["ell"], nu0,
                           config.TRUTH["tau2"])
    L = np.linalg.cholesky(dense_covariance(d, G, th)[0] - config.TRUTH["tau2"] * np.eye(m.n))
    rng = np.random.default_rng(3)
    bad = {}
    for law in noise.LAWS:
        k = np.empty(nrep)
        for r in range(nrep):
            y = L @ rng.standard_normal(m.n) + math.sqrt(config.TRUTH["tau2"]) * noise.sample(
                law, rng, m.n)
            k[r] = rc.compute(th, m.sym.forward(y))["kappa"]
        target = noise.kappa4(law, config.TRUTH["tau2"])
        z = (k.mean() - target) / (k.std(ddof=1) / math.sqrt(nrep))
        print(f"  {law:8s} kappa4={target:+.4f}  mean kappa_tilde={k.mean():+.4f} "
              f"(MC SE {k.std(ddof=1) / math.sqrt(nrep):.4f}, z={z:+.2f})", flush=True)
        if abs(z) > 4:
            bad[law] = z
    return bad


def main():
    bad = dense_checks()
    print(f"unbiasedness at the true covariance (n=8^3):", flush=True)
    bad.update(unbiasedness())
    if bad:
        raise SystemExit(f"FAILED: {bad}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
