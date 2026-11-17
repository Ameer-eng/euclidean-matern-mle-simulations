"""Matern covariance and its parameter derivatives, tabulated on the grid.

Parameterisation (paper eq. 3.1, 3.2): theta = (eta, nu, log a, log t),
  sigma^2 = e^eta / (A_d(nu) a^nu),  l = a^{-1/2},  tau^2 = t,
  K(r) = sigma^2 M_nu(r sqrt(a)),  M_nu(z) = 2^{1-nu} z^nu K_nu(z) / Gamma(nu).
No sqrt(2 nu) factor in the argument.

On the midpoint grid with spacing 1/G every lag has r = sqrt(D)/G with D an
integer in [0, d (G-1)^2]; covariance entries are table lookups K[D].
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import digamma, gammaln, kve

LOG2 = math.log(2.0)


def log_A(d, nu):
    """log A_d(nu) = (d/2) log(4 pi) + log Gamma(nu + d/2) - log Gamma(nu)."""
    return 0.5 * d * math.log(4.0 * math.pi) + gammaln(nu + 0.5 * d) - gammaln(nu)


def theta_from_public(d, sigma2, ell, nu, tau2):
    a = ell ** -2.0
    eta = log_A(d, nu) + math.log(sigma2) + nu * math.log(a)
    return np.array([eta, nu, math.log(a), math.log(tau2)])


def public_from_theta(d, theta):
    eta, nu, la, lt = theta
    log_s2 = eta - log_A(d, nu) - nu * la
    return dict(sigma2=math.exp(log_s2), ell=math.exp(-0.5 * la), nu=nu,
                tau2=math.exp(lt), c=math.exp(eta - log_A(d, nu)))


def _dlogK_dnu(nu, z, h=0.05):
    """d/dnu log K_nu(z) for z > 0 via the trapezoidal rule on
    K_nu(z) = int_0^inf exp(-z cosh u) cosh(nu u) du (spectrally accurate)."""
    z = np.asarray(z, float)
    zmin = float(z.min())
    anu = abs(nu)
    # upper limit: log-integrand has dropped by > 60 below its peak
    U = 5.0
    for _ in range(50):
        peak = max(0.0, anu * math.log(max(2.0 * anu / zmin, 1.0)))
        if zmin * (math.cosh(U) - 1.0) - anu * U > peak + 60.0:
            break
        U += 0.5
    u = np.arange(0.0, U + h, h)
    wts = np.full(u.shape, h)
    wts[0] = 0.5 * h
    x = anu * u
    logcosh = x + np.log1p(np.exp(-2.0 * x)) - LOG2
    out = np.empty_like(z)
    chunk = max(1, 2_000_000 // len(u))
    for s in range(0, z.size, chunk):
        zz = z.ravel()[s:s + chunk][:, None]
        lf = -zz * (np.cosh(u)[None, :] - 1.0) + logcosh[None, :]
        lf -= lf.max(axis=1, keepdims=True)
        e = np.exp(lf) * wts[None, :]
        out.ravel()[s:s + chunk] = (e * (u * np.tanh(x))[None, :]).sum(1) / e.sum(1)
    return np.sign(nu) * out if nu != 0 else np.zeros_like(out)


def matern_tables(theta, d, G, Dmax, which=("K", "Knu", "Ka")):
    """Rows of the returned (len(which), Dmax+1) array:
       K   = covariance,           Knu = dK/dnu,
       Ka  = dK/d(log a),          (dK/d eta = K;  dC/d log t = t I)."""
    eta, nu, la, lt = (float(v) for v in theta)
    D = np.arange(Dmax + 1, dtype=float)
    r = np.sqrt(D) / G
    z = r * math.exp(0.5 * la)
    log_s2 = eta - log_A(d, nu) - nu * la
    s2 = math.exp(log_s2)
    logc = (1.0 - nu) * LOG2 - gammaln(nu)
    pos = z > 0
    zp = z[pos]
    lz = np.log(zp)
    M = np.ones_like(z)
    with np.errstate(divide="ignore", under="ignore"):
        M[pos] = np.exp(logc + nu * lz + np.log(kve(nu, zp)) - zp)
    out = []
    for w in which:
        if w == "K":
            out.append(s2 * M)
        elif w == "Ka":
            Nz = np.zeros_like(z)  # z dM/dz = -c z^{nu+1} K_{nu-1}(z)
            with np.errstate(divide="ignore", under="ignore"):
                Nz[pos] = -np.exp(logc + (nu + 1.0) * lz + np.log(kve(nu - 1.0, zp)) - zp)
            out.append(s2 * (-nu * M + 0.5 * Nz))
        elif w == "Knu":
            s_nu = -(digamma(nu + 0.5 * d) - digamma(nu)) - la
            L = np.zeros_like(z)
            L[pos] = -LOG2 - digamma(nu) + lz + _dlogK_dnu(nu, zp)
            out.append(s2 * (M * s_nu + M * L))
        else:
            raise ValueError(w)
    return np.vstack(out)
