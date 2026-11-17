"""Regression check for the joint-region validity rule (FOLLOW-UP Task 2, 2026-11-10).

Run:  python -m tests.test_joint_validity
1. Unit checks of matern_sim.aggregate.joint_wald: an indefinite covariance that
   gives a negative quadratic statistic (and so was counted as covering under the
   old e' solve(S, e) rule) is rejected and cannot count as a valid covering region;
   PD, slightly asymmetric (rounding), clearly asymmetric and non-finite inputs.
2. If results/summary/joint.csv exists: count consistency in every row, and, for the
   six largest-n settings, agreement of the rebuilt hess0 rows with the earlier
   independent audit audit/out/joint_check.csv (cov_PD_only, n_nonPD_used).
"""
import math
import os

import numpy as np

from matern_sim.aggregate import SYM_RTOL, joint_wald
from scipy.stats import chi2


def unit_checks():
    crit = chi2.ppf(0.95, 2)
    # indefinite: eigenvalues 1 and -1; e along the negative direction
    S = np.array([[0.0, 1.0], [1.0, 0.0]])
    e = np.array([1.0, -1.0])
    old = float(e @ np.linalg.solve(S, e))
    assert old < 0 and old <= crit, old           # the old rule counted this as covered
    st, q, _ = joint_wald(e, S)
    assert st == "not_pd" and math.isnan(q), (st, q)
    covered_valid = (st == "ok") and (q <= crit)
    assert not covered_valid
    # PD: statistic equals e' S^{-1} e
    S = np.array([[2.0, 0.3], [0.3, 0.5]])
    e = np.array([0.7, -0.2])
    st, q, a = joint_wald(e, S)
    assert st == "ok" and abs(q - e @ np.linalg.solve(S, e)) < 1e-12 and a == 0.0
    # rounding-level asymmetry is accepted (and symmetrised), recorded
    Sa = S.copy()
    Sa[0, 1] += 1e-14
    st, q, a = joint_wald(e, Sa)
    assert st == "ok" and 0 < a <= SYM_RTOL
    # genuine asymmetry is rejected
    Sb = S.copy()
    Sb[0, 1] += 1e-3
    assert joint_wald(e, Sb)[0] == "asymmetric"
    # singular PSD (rank deficient) is not PD
    assert joint_wald(e, np.array([[1.0, 1.0], [1.0, 1.0]]))[0] == "not_pd"
    # non-finite
    assert joint_wald(e, np.array([[np.nan, 0.0], [0.0, 1.0]]))[0] == "nonfinite"
    print("unit checks passed")


def table_checks():
    import pandas as pd
    p = "results/summary/joint.csv"
    if not os.path.exists(p):
        print("no joint.csv; table checks skipped")
        return
    j = pd.read_csv(p)
    assert {"n_valid", "n_cover_valid", "prob_valid_and_cover"} <= set(j.columns), \
        "joint.csv predates the validity rule"
    assert (j.n_cover_valid <= j.n_valid).all() and (j.n_valid <= j.nfit).all()
    inv = j.n_invalid_nonfinite + j.n_invalid_asymmetric + j.n_invalid_not_pd
    assert (inv + j.n_valid == j.nfit).all()
    ok = j.n_valid > 0
    assert np.allclose(j.coverage[ok], j.n_cover_valid[ok] / j.n_valid[ok])
    assert np.allclose(j.prob_valid_and_cover, j.n_cover_valid / j.nfit)
    print(f"joint.csv: {len(j)} rows consistent; {int(j.n_invalid_not_pd.sum())} non-PD "
          f"region-replicates excluded in total, {int(j.n_invalid_asymmetric.sum())} asymmetric")
    a = "audit/out/joint_check.csv"
    if os.path.exists(a):
        c = pd.read_csv(a)
        m = c.merge(j[j.variant == "joint"], on=["d", "nu0", "n", "law", "vector", "method"])
        assert len(m) == len(c), (len(m), len(c))
        assert np.allclose(m.cov_PD_only, m.coverage), \
            m.loc[~np.isclose(m.cov_PD_only, m.coverage), ["d", "nu0", "law", "method"]]
        assert (m.n_nonPD_used == m.n_invalid_not_pd).all()
        print(f"agrees with the earlier audit on {len(m)} largest-n rows "
              f"({int(m.n_nonPD_used.sum())} non-PD replicates)")


if __name__ == "__main__":
    unit_checks()
    table_checks()
