"""Global quasi-MLE over the box Theta, observed Hessian, oracle and pair fits."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import LinAlgError
from scipy.optimize import minimize, minimize_scalar

from . import config


class Fitter:
    """L-BFGS-B in x = sc * (phi - phi0),  phi = (eta - 2 L_n nu, nu, log a, log t),
    sc = sqrt(diag Fisher_phi(theta0))  (Appendix D.3).  theta = T x + theta0."""

    def __init__(self, model, theta0, nu0, I0, box=None, lbfgs=None):
        self.model = model
        self.theta0 = np.asarray(theta0, float)
        d, n = model.d, model.n
        self.Ln = math.log(n) / (2.0 * nu0 + d)
        A = np.eye(4)
        A[0, 1] = -2.0 * self.Ln                     # phi = A theta
        Ainv = np.linalg.inv(A)
        Iphi = Ainv.T @ I0 @ Ainv
        self.sc = np.sqrt(np.diag(Iphi))
        self.T = Ainv @ np.diag(1.0 / self.sc)       # theta - theta0 = T x
        self.Tinv = np.linalg.inv(self.T)
        box = box or config.BOX
        self.lbfgs = dict(lbfgs or config.LBFGS)
        phi0 = A @ self.theta0
        nu_lo, nu_hi = box["nu"]
        lo = np.array([box["eta"][0] - 2 * self.Ln * nu_hi, nu_lo,
                       -2 * math.log(box["ell"][1]), math.log(box["tau2"][0])])
        hi = np.array([box["eta"][1] - 2 * self.Ln * nu_lo, nu_hi,
                       -2 * math.log(box["ell"][0]), math.log(box["tau2"][1])])
        self.xlo = self.sc * (lo - phi0)
        self.xhi = self.sc * (hi - phi0)
        self.box = box

    def theta(self, x):
        return self.theta0 + self.T @ x

    def x_of(self, theta):
        return self.Tinv @ (np.asarray(theta) - self.theta0)

    def objective(self, x, Z):
        th = self.theta(x)
        try:
            ll, g = self.model.loglik_grad(th, Z)
        except LinAlgError:
            return 1e300, np.zeros_like(x)
        return -ll, -(self.T.T @ g)

    def fit(self, Z, fix_nu=False, x_start=None):
        x0 = np.zeros(4) if x_start is None else np.asarray(x_start, float)
        bounds = list(zip(self.xlo, self.xhi))
        if fix_nu:
            # nu fixed at nu0: x_nu = 0 (eta still free through x_0)
            bounds[1] = (0.0, 0.0)
            x0[1] = 0.0
        t = time.time()
        res = minimize(self.objective, x0, args=(Z,), jac=True, method="L-BFGS-B",
                       bounds=bounds, options=self.lbfgs)
        th = self.theta(res.x)
        ll, g = self.model.loglik_grad(th, Z)
        eta, nu, la, lt = th
        tol = 1e-8
        at_bound = np.array([
            not (self.box["eta"][0] < eta < self.box["eta"][1]),
            (nu <= self.box["nu"][0] + tol) or (nu >= self.box["nu"][1] - tol),
            (res.x[2] <= self.xlo[2] + tol) or (res.x[2] >= self.xhi[2] - tol),
            (res.x[3] <= self.xlo[3] + tol) or (res.x[3] >= self.xhi[3] - tol),
        ])
        if fix_nu:
            at_bound[1] = False
        return dict(theta=th, x=res.x, loglik=ll, grad=g, success=bool(res.success),
                    message=str(res.message), nit=int(res.nit), nfev=int(res.nfev),
                    at_bound=at_bound, seconds=time.time() - t)

    def neg_hessian_fd(self, Z, theta_hat, h=None):
        """Observed negative Hessian of ell in spectral coordinates, by central
        differences of the analytic gradient in the scaled x coordinates."""
        h = config.HESS_STEP if h is None else h
        x = self.x_of(theta_hat)
        Hx = np.empty((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = h
            _, gp = self.objective(x + e, Z)
            _, gm = self.objective(x - e, Z)
            Hx[:, j] = (gp - gm) / (2 * h)          # Hessian of -ell in x
        Hx = 0.5 * (Hx + Hx.T)
        return self.Tinv.T @ Hx @ self.Tinv          # -d2 ell / d theta2


def oracle_nugget_fit(lam, s2, m, box=None):
    """argmax_t of the criterion with (eta, nu, a) at the truth."""
    box = box or config.BOX

    def negll(lt):
        v = lam + math.exp(lt)
        return 0.5 * np.sum(m * np.log(v) + s2 / v)

    lo, hi = math.log(box["tau2"][0]), math.log(box["tau2"][1])
    r = minimize_scalar(negll, bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-12, maxiter=500))
    return float(r.x), float(-r.fun)


def adjacent_pair_differences(y, G, d):
    """Disjoint pairs of axis-adjacent sites (spacing 1/G): (.., 2s) with (.., 2s+1)
    along the last axis; for odd G the leftover slice is paired along the next
    axis, recursively, so at most one site is unpaired.  All pairs have the same
    latent-increment variance, which makes the field contribution cancel in the
    fourth-cumulant estimate (an independent Gaussian increment has kappa4 = 0)."""
    a = np.asarray(y).reshape((G,) * d)
    out = []
    while a.ndim >= 1 and a.shape[-1] >= 2:
        m = a.shape[-1] // 2
        out.append((a[..., 1:2 * m:2] - a[..., 0:2 * m:2]).ravel())
        if a.shape[-1] % 2 == 0:
            break
        a = a[..., -1]
    return np.concatenate(out)


def pair_estimates(y, G, d):
    """Lemma A.12: t_tilde = sum D^2/(2m), kappa_hat = sum D^4/(2m) - 6 t_tilde^2."""
    D = adjacent_pair_differences(y, G, d)
    m = D.size
    t_pair = np.sum(D**2) / (2 * m)
    k_pair = np.sum(D**4) / (2 * m) - 6.0 * t_pair**2
    return float(t_pair), float(k_pair), int(m)
