"""Step-size check of the finite-difference observed Hessian (Fitter.neg_hessian_fd).

This is the script that produced audit/out/hess_step_check.txt. It was run inline on
2026-11-09 and saved here verbatim on 2026-11-16; only this docstring and the
__main__ guard were added.

For three Gaussian data sets (replicate 0 of d=5 nu0=1/2 G=6, d=4 nu0=3/2 G=8 and
d=3 nu0=1/2 G=14), the data are regenerated from their seeds. The negative Hessian is
then recomputed at theta_hat and at theta0 with steps h = 1e-2, ..., 1e-6.

The step h is taken in the optimizer's scaled coordinates x, with theta - theta0 = T x
and x_j = sc_j (phi_j - phi0_j), where sc = sqrt(diag Fisher_phi(theta0)). The gradient
is analytic and is differenced centrally; the result is symmetrized and mapped to
-d2 ell / d w2 with w = (eta, nu, log a, log t).

Metric: max_{jk} |H_h[j,k] - H_ref[j,k]| / sqrt(H_ref[j,j] H_ref[k,k]), with
H_ref = H at h = 1e-3. The same scaling is used for the stored H (production step 1e-4)
against the recomputed h = 1e-4 value.

"data match" means |ell(theta_hat; regenerated data) - stored ell| < 1e-8 |stored ell|.
When it is False (a different CPU family from the original fit, see
results/kappa/fit_hosts.csv), the stored-vs-recomputed number compares different data
sets and is not a step-size error.

  python -m audit.hess_step_check
"""
import numpy as np

from matern_sim.fit import Fitter
from matern_sim.kappa_task import data_blocks, field_blocks
from matern_sim.likelihood import GridModel
from matern_sim.run_task import fit_path, truth


def main():
    for d, nu0, G in [(5, 0.5, 6), (4, 1.5, 8), (3, 0.5, 14)]:
        m = GridModel(d, G); th0 = truth(d, nu0)
        I0 = np.load(f"results/theory/d{d}_nu{nu0:g}_G{G}.npz")["I0"]
        F = Fitter(m, th0, nu0, I0); eig = m.signal_eigen(th0)
        z = np.load(fit_path("results", d, nu0, G, "joint", "gauss", 0))
        th = np.array(z["theta_hat"]); Z = data_blocks(m, field_blocks(eig, d, nu0, G, 0), d, nu0, G, 0, "gauss")
        match = abs(m.loglik_grad(th, Z, grad=False) - float(z["loglik"])) < 1e-8 * abs(float(z["loglik"]))
        for lab, pt in (("theta_hat", th), ("theta0", th0)):
            H = {h: F.neg_hessian_fd(Z, pt, h) for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)}
            ref = H[1e-3]
            D = np.sqrt(np.diag(ref))
            msg = " ".join(f"h={h:g}:{np.max(np.abs((H[h]-ref)/np.outer(D,D))):.1e}" for h in H)
            if lab == "theta_hat":
                st = np.max(np.abs((np.array(z['negH']) - H[1e-4]) / np.outer(D, D)))
                msg += f"  stored-vs-recomputed(h=1e-4) {st:.1e}"
            print(f"d={d} nu0={nu0} n={m.n} {lab} (data match {match}): scaled max diff vs h=1e-3: {msg}")


if __name__ == "__main__":
    main()
