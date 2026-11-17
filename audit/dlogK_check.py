"""Check of kernel._dlogK_dnu (d/dnu log K_nu(z)).

Part 1 is the check recorded in audit/out/dlogK_check.txt (run inline on 2026-11-09,
saved here verbatim on 2026-11-16): at nu = 1/2, d/dnu log K_nu(z) = e^{2z} E_1(2z),
compared on z = logspace(-3, 2, 400).

Parts 2-3 were added on 2026-11-16 (Appendix E question 7):
  2. the same closed form on z in [1e-3, 120], which covers every lag the parameter
     box allows (z = r sqrt(a), sqrt(a) = 1/ell in [0.5, 50], r in [1/46, sqrt(5)]:
     z in [0.011, 111.8]); every saved fit and theta0 have z in [0.0149, 66.7];
  3. other orders nu, against a Richardson-extrapolated central difference of
     log kve(nu, z) in nu (reference accuracy ~1e-10, so this part checks agreement
     to that level only).

  python -m audit.dlogK_check > audit/out/appendixE/dlogK_check.txt
"""
import numpy as np
from scipy.special import exp1, kve

from matern_sim.kernel import _dlogK_dnu


def part1():
    z = np.logspace(-3, 2, 400)
    ref = np.exp(2 * z) * exp1(2 * z)        # d/dnu log K_nu(z) at nu = 1/2 (closed form)
    got = _dlogK_dnu(0.5, z)
    print("max relative error of d/dnu log K_nu at nu=1/2, z in [1e-3, 1e2]:",
          np.max(np.abs(got - ref) / np.abs(ref)))


def part2():
    for lo, hi in [(1e-3, 120.0), (0.011, 112.0), (0.0149, 66.7)]:
        z = np.logspace(np.log10(lo), np.log10(hi), 2000)
        ref = np.exp(2 * z) * exp1(2 * z)
        got = _dlogK_dnu(0.5, z)
        print(f"nu=1/2, z in [{lo:g}, {hi:g}] (2000 log-spaced points): max relative error "
              f"{np.max(np.abs(got - ref) / np.abs(ref)):.3e}")


def fd_ref(nu, z, h=1e-3):
    """Richardson-extrapolated central difference of log K_nu(z) in nu."""
    def cd(hh):
        return (np.log(kve(nu + hh, z)) - np.log(kve(nu - hh, z))) / (2 * hh)
    d1, d2 = cd(h), cd(h / 2)
    return (4 * d2 - d1) / 3


def part3():
    z = np.logspace(np.log10(0.011), np.log10(112.0), 400)
    for nu in (0.1, 0.15, 0.5, 1.0, 1.5, 2.7, 4.0):
        ref = fd_ref(nu, z)
        got = _dlogK_dnu(nu, z)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
        print(f"nu={nu:<4g} z in [0.011, 112]: max relative difference from FD reference "
              f"{rel.max():.2e} (min |reference| {np.abs(ref).min():.2e})")


if __name__ == "__main__":
    part1()
    part2()
    part3()
