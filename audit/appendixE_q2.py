"""APPENDIX E QUESTIONS (2026-11-16), Q2: folded (parity-basis) covariance entries on
odd grids.

Independent check of the formula implemented in assemble.assemble_parity_block, using
plain NumPy: build the dense signal covariance K[i, j] = k(D(i, j)) from the Matern
table on squared integer lags D, rotate it to the parity basis
P = P_axis (x) ... (x) P_axis (symmetry.axis_parity_matrix), and compare every entry
with
    w_x w_y sum_{c in {0,1}^d} s_c k(D_c(x, y)),
where
  * per axis q, the squared offset is (x_q - y_q)^2 if c_q = 0 and
    (G - 1 - x_q - y_q)^2 if c_q = 1;
  * s_c = prod over odd axes of (-1)^{c_q};
  * w_x = prod over even axes of (1/sqrt 2 if x_q is the centre index (G - 1)/2, else 1).
Here x and y index the (pair, centre) coordinates of the even/odd basis vectors.

The 1-D special cases (d = 1) are printed separately: pair-pair, pair-centre,
centre-centre and odd-odd.

  python -m audit.appendixE_q2
"""
import itertools
import math

import numpy as np

from matern_sim.kernel import matern_tables, theta_from_public
from matern_sim.symmetry import axis_parity_matrix


def kron_all(M, d):
    out = np.ones((1, 1))
    for _ in range(d):
        out = np.kron(out, M)
    return out


def check(d, G, nu):
    th = theta_from_public(d, 1.0, 0.4, nu, 0.25)
    tab = matern_tables(th, d, G, d * (G - 1) ** 2, which=("K",))[0]
    pts = np.array(list(itertools.product(range(G), repeat=d)))
    D = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    K = tab[D]
    P1 = axis_parity_matrix(G)
    me = (G + 1) // 2
    P = kron_all(P1, d)
    KP = P.T @ K @ P
    # basis column index along an axis: 0..me-1 even (pair j, or centre at me-1 if G odd),
    # me..G-1 odd (pair j - me)
    cols = list(itertools.product(range(G), repeat=d))
    centre = (G - 1) // 2 if G % 2 else None
    worst = 0.0
    for a, ca in enumerate(cols):
        for b, cb in enumerate(cols):
            par_a = [int(i >= me) for i in ca]
            par_b = [int(i >= me) for i in cb]
            if par_a != par_b:      # different parity patterns: block-diagonal zero
                worst = max(worst, abs(KP[a, b]))
                continue
            xa = [i - me if p else i for i, p in zip(ca, par_a)]   # pair index j along the axis
            xb = [i - me if p else i for i, p in zip(cb, par_b)]
            w = 1.0
            for q in range(d):
                if not par_a[q] and centre is not None and xa[q] == me - 1:
                    w /= math.sqrt(2)
                if not par_b[q] and centre is not None and xb[q] == me - 1:
                    w /= math.sqrt(2)
            s = 0.0
            for c in itertools.product((0, 1), repeat=d):
                Dc = sum((xa[q] - xb[q]) ** 2 if c[q] == 0 else (G - 1 - xa[q] - xb[q]) ** 2
                         for q in range(d))
                sign = (-1) ** sum(c[q] for q in range(d) if par_a[q])
                s += sign * tab[Dc]
            worst = max(worst, abs(w * s - KP[a, b]) / tab[0])
    return worst


def one_d(G=7, nu=0.5):
    th = theta_from_public(1, 1.0, 0.4, nu, 0.25)
    tab = matern_tables(th, 1, G, (G - 1) ** 2, which=("K",))[0]
    k = lambda h: tab[h * h]
    P1 = axis_parity_matrix(G)
    K = np.array([[k(i - j) for j in range(G)] for i in range(G)])
    KP = P1.T @ K @ P1
    me, c = (G + 1) // 2, (G - 1) // 2
    j, jp = 0, 1
    print(f"d=1, G={G}, centre index c={c}:")
    print(f"  pair j={j} vs pair j'={jp}: {KP[j, jp]:.15f}  formula k(j-j')+k(G-1-j-j') = {k(j - jp) + k(G - 1 - j - jp):.15f}")
    print(f"  pair j={j} vs centre     : {KP[j, me - 1]:.15f}  formula sqrt2 k(c-j)         = {math.sqrt(2) * k(c - j):.15f}")
    print(f"  centre vs centre         : {KP[me - 1, me - 1]:.15f}  formula k(0)                  = {k(0):.15f}")
    print(f"  odd j={j} vs odd j'={jp}   : {KP[me + j, me + jp]:.15f}  formula k(j-j')-k(G-1-j-j') = {k(j - jp) - k(G - 1 - j - jp):.15f}")
    print(f"  even-odd cross entries   : max |.| = {np.abs(KP[:me, me:]).max():.1e}")


if __name__ == "__main__":
    one_d()
    for d, G in [(1, 7), (2, 5), (2, 7), (3, 3), (3, 5), (4, 3), (2, 6), (3, 4)]:
        for nu in (0.5, 1.5):
            print(f"d={d} G={G} nu={nu}: max |formula - (P^T K P)| / k(0) = {check(d, G, nu):.1e}")
