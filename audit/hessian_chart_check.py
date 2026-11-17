"""AUDIT (2026-11-09): observed Hessian in the reported log chart vs the implemented
Jacobian transformation of the spectral-chart Hessian (Q5/Q6).

Reported charts zeta:  d = 3: (log c, log nu, log l, log tau2)   [App. Table 1, microergodic log]
                       d > 3: (log sigma2, log l, log nu, log tau2)  [variance-lengthscale log]
F: zeta -> theta = (eta, nu, log a, log t) (the code's chart).
  implemented   H_impl = G^T H_theta G,                 G = DF(zeta)
  formula       H_form = G^T H_theta G - sum_j s_j D^2 F_j,  s = grad_theta ell
  direct        H_dir  = central differences, in zeta, of the exact zeta-gradient G(zeta)^T s(F(zeta))
                H_dir2 = second differences of ell(F(zeta)) (function values only)
H_theta is computed exactly as in production (Fitter.neg_hessian_fd, HESS_STEP).
"""
import math, sys
import numpy as np
from matern_sim import config
from matern_sim.fit import Fitter
from matern_sim.kappa_task import data_blocks, field_blocks
from matern_sim.kernel import log_A
from matern_sim.likelihood import GridModel
from matern_sim.run_task import fit_path, truth
from scipy.special import digamma, polygamma


def F_and_derivs(zeta, d):
    """theta = F(zeta), G = DF (4x4), H[j] = D^2 F_j (4x4 each)."""
    if d <= 3:
        lc, lnu, ll, lt = zeta
        nu = math.exp(lnu)
        B, Bp = digamma(nu + d / 2) - digamma(nu), polygamma(1, nu + d / 2) - polygamma(1, nu)
        th = np.array([lc + log_A(d, nu), nu, -2 * ll, lt])
        G = np.zeros((4, 4)); G[0, 0] = 1; G[0, 1] = B * nu; G[1, 1] = nu; G[2, 2] = -2; G[3, 3] = 1
        H = np.zeros((4, 4, 4))
        H[0][1, 1] = Bp * nu * nu + B * nu          # d2/dlnu2 log A(e^lnu)
        H[1][1, 1] = nu
        return th, G, H
    ls, ll, lnu, lt = zeta
    nu = math.exp(lnu)
    B, Bp = digamma(nu + d / 2) - digamma(nu), polygamma(1, nu + d / 2) - polygamma(1, nu)
    th = np.array([ls + log_A(d, nu) - 2 * nu * ll, nu, -2 * ll, lt])
    G = np.zeros((4, 4)); G[0, 0] = 1; G[0, 1] = -2 * nu; G[0, 2] = B * nu - 2 * nu * ll
    G[1, 2] = nu; G[2, 1] = -2; G[3, 3] = 1
    H = np.zeros((4, 4, 4))
    H[0][2, 2] = Bp * nu * nu + B * nu - 2 * nu * ll
    H[0][1, 2] = H[0][2, 1] = -2 * nu
    H[1][2, 2] = nu
    return th, G, H


def zeta_of(theta, d):
    eta, nu, la, lt = theta
    if d <= 3:
        return np.array([eta - log_A(d, nu), math.log(nu), -0.5 * la, lt])
    return np.array([eta - log_A(d, nu) - nu * la, -0.5 * la, math.log(nu), lt])


def check(m, F, Z, theta, label, d, steps=(1e-2, 1e-3, 1e-4)):
    zeta = zeta_of(theta, d)
    th, G, H2 = F_and_derivs(zeta, d)
    assert np.allclose(th, theta, atol=1e-12)
    ll0, s = m.loglik_grad(theta, Z)
    Hth = F.neg_hessian_fd(Z, theta)
    H_impl = G.T @ Hth @ G
    H_form = H_impl - sum(s[j] * H2[j] for j in range(4))
    sd = 1 / np.sqrt(np.abs(np.diag(H_form)))          # step in units of ~1 SD
    out = {}
    for h in steps:
        Hd = np.empty((4, 4)); Hf = np.empty((4, 4))
        for k in range(4):
            e = np.zeros(4); e[k] = h * sd[k]
            gp = F_and_derivs(zeta + e, d)[1].T @ m.loglik_grad(F_and_derivs(zeta + e, d)[0], Z)[1]
            gm = F_and_derivs(zeta - e, d)[1].T @ m.loglik_grad(F_and_derivs(zeta - e, d)[0], Z)[1]
            Hd[:, k] = -(gp - gm) / (2 * e[k])
        Hd = 0.5 * (Hd + Hd.T)
        f = lambda z: m.loglik_grad(F_and_derivs(z, d)[0], Z, grad=False)
        for i in range(4):
            for k in range(4):
                ei = np.zeros(4); ei[i] = h * sd[i]; ek = np.zeros(4); ek[k] = h * sd[k]
                Hf[i, k] = -(f(zeta + ei + ek) - f(zeta + ei - ek) - f(zeta - ei + ek)
                             + f(zeta - ei - ek)) / (4 * ei[i] * ek[k])
        out[h] = (Hd, Hf)
    # scale-free comparison: D^-1/2 (A - B) D^-1/2 with D = diag of the direct Hessian
    def rel(A, B):
        D = np.sqrt(np.abs(np.diag(B)))
        return float(np.max(np.abs((A - B) / np.outer(D, D))))
    Hd = out[1e-3][0]
    ident = [0, 1, 3] if d <= 3 else [0, 1, 2, 3]
    var = lambda A: np.diag(np.linalg.inv(A))[ident]
    print(f"\n{label}: n={m.n} |score| in SD units = "
          f"{np.round(np.abs(G.T @ s) * sd, 2)}  (zeta-chart score x SD)")
    for h, (Hd_, Hf_) in out.items():
        print(f"  step {h:g} SD: direct-gradient vs direct-function {rel(Hd_, Hf_):.1e};"
              f"  impl vs direct {rel(H_impl, Hd_):.1e};  formula vs direct {rel(H_form, Hd_):.1e}")
    vi, vf, vd = var(H_impl), var(H_form), var(Hd)
    print(f"  variances of identifiable zeta (inverse of full matrix): impl {vi}, "
          f"with score term {vf}, direct {vd}")
    print(f"  ratio sqrt(var impl / var direct) = {np.sqrt(vi / vd)}")
    return rel(H_impl, Hd), rel(H_form, Hd)


def main():
    cases = [(4, 0.5, 4), (3, 0.5, 7), (5, 1.5, 3), (3, 1.5, 7)]
    for d, nu0, G in cases:
        m = GridModel(d, G)
        th0 = truth(d, nu0)
        I0 = np.load(f"results/theory/d{d}_nu{nu0:g}_G{G}.npz")["I0"]
        F = Fitter(m, th0, nu0, I0)
        eig = m.signal_eigen(th0)
        found = {}
        for rep in range(400):
            z = np.load(fit_path("results", d, nu0, G, "joint", "gauss", rep))
            ab = z["at_bound"]
            kind = "interior" if not ab.any() else ("nu_upper" if ab[1] and z["theta_hat"][1] > 2
                                                    else ("tau_lower" if ab[3] else None))
            if kind and kind not in found:
                found[kind] = (rep, np.array(z["theta_hat"]), float(z["loglik"]))
            if len(found) == 3:
                break
        for kind, (rep, th, llst) in found.items():
            Z = data_blocks(m, field_blocks(eig, d, nu0, G, rep), d, nu0, G, rep, "gauss")
            ok = abs(m.loglik_grad(th, Z, grad=False) - llst) < 1e-8 * abs(llst)
            check(m, F, Z, th, f"d={d} nu0={nu0} G={G} rep {rep}: {kind} fit (data match {ok})", d)
            if kind == "interior":
                check(m, F, Z, th0, f"d={d} nu0={nu0} G={G} rep {rep}: TRUTH theta0", d)


if __name__ == "__main__":
    main()
