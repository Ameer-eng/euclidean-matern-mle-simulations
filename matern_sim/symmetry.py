"""Hyperoctahedral symmetry of the regular midpoint grid (Appendix D.1).

The covariance of an isotropic field on the grid {(i - 1/2)/G}^d commutes with
the reflections i_k -> G-1-i_k and with permutations of the axes.  A fixed
orthogonal matrix Q (independent of theta) block-diagonalises C_n(theta):

  1. Per axis, the even/odd vectors (e_j +- e_{G-1-j})/sqrt(2) (and the centre
     vector for odd G) split C_n into 2^d parity blocks, one per pattern
     eps in {0,1}^d.  Patterns with the same number k of odd axes give the same
     block after relabelling axes, so only d+1 blocks are distinct.
  2. Within the canonical block (first k axes odd), the stabiliser
     H = S_k x S_{d-k} permutes reduced multi-indices.  Using Young's orthogonal
     form, the projector P^rho_{11} = (f/|H|) sum_h rho(h)_{11} pi(h) selects one
     copy of each irreducible representation rho; P^rho_{i1} maps it to copy i.
     The block restricted to every copy is the same matrix B_rho.

Q is never formed.  `GridSymmetry.forward` maps a grid vector to block
coordinates (Q^T y) and `GridSymmetry.inverse` maps back (Q z).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# Symmetric-group representations (Young's orthogonal form)
# --------------------------------------------------------------------------
def partitions(k):
    """Partitions of k in decreasing lexicographic order."""
    if k == 0:
        return [()]
    out = []

    def rec(rem, maxpart, prefix):
        if rem == 0:
            out.append(tuple(prefix))
            return
        for p in range(min(rem, maxpart), 0, -1):
            rec(rem - p, p, prefix + [p])

    rec(k, k, [])
    return out


def standard_tableaux(shape):
    """Standard Young tableaux of `shape`; tableau = tuple of (row, col) per entry."""
    k = sum(shape)
    out = []

    def rec(e, rowlen, cells):
        if e == k:
            out.append(tuple(cells))
            return
        for r in range(len(shape)):
            if rowlen[r] < shape[r] and (r == 0 or rowlen[r] < rowlen[r - 1]):
                rowlen[r] += 1
                rec(e + 1, rowlen, cells + [(r, rowlen[r] - 1)])
                rowlen[r] -= 1

    rec(0, [0] * len(shape), [])
    return out


def young_generators(shape):
    """Orthogonal matrices rho(s_i), s_i = (i, i+1), i = 0..k-2."""
    tabs = standard_tableaux(shape)
    index = {t: a for a, t in enumerate(tabs)}
    f = len(tabs)
    k = sum(shape)
    gens = []
    for i in range(k - 1):
        M = np.zeros((f, f))
        for a, T in enumerate(tabs):
            (r1, c1), (r2, c2) = T[i], T[i + 1]
            r = (c2 - r2) - (c1 - r1)  # axial distance (content difference)
            M[a, a] = 1.0 / r
            if abs(r) != 1:
                T2 = list(T)
                T2[i], T2[i + 1] = T[i + 1], T[i]
                M[index[tuple(T2)], a] = math.sqrt(1.0 - 1.0 / r**2)
        gens.append(M)
    return f, gens


def symmetric_group(k):
    """Elements of S_k (tuples p with p[q] = image of q) and Young irreps.

    Returns (elements, irreps) where irreps is a list of (shape, f, rho) with
    rho an array of shape (|S_k|, f, f) satisfying rho[g o h] = rho[g] rho[h],
    composition (g o h)(q) = g(h(q)).
    """
    ident = tuple(range(k))
    gens_perm = []
    for i in range(k - 1):
        p = list(range(k))
        p[i], p[i + 1] = p[i + 1], p[i]
        gens_perm.append(tuple(p))
    # BFS over the Cayley graph: record each element as s o parent.
    elements = [ident]
    parent = [(-1, -1)]
    seen = {ident: 0}
    q = 0
    while q < len(elements):
        g = elements[q]
        for s_idx, s in enumerate(gens_perm):
            h = tuple(s[g[j]] for j in range(k))
            if h not in seen:
                seen[h] = len(elements)
                elements.append(h)
                parent.append((q, s_idx))
        q += 1
    irreps = []
    for shape in partitions(k):
        f, gens = young_generators(shape)
        rho = np.empty((len(elements), f, f))
        rho[0] = np.eye(f)
        for e in range(1, len(elements)):
            par, s_idx = parent[e]
            rho[e] = gens[s_idx] @ rho[par]
        irreps.append((shape, f, rho))
    return elements, irreps


# --------------------------------------------------------------------------
# Grid structure
# --------------------------------------------------------------------------
def axis_parity_matrix(G):
    """G x G orthogonal matrix; columns = even basis (m_e) then odd basis (m_o)."""
    me, mo = (G + 1) // 2, G // 2
    P = np.zeros((G, G))
    s = 1.0 / math.sqrt(2.0)
    for j in range(mo):
        P[j, j] = s
        P[G - 1 - j, j] = s
        P[j, me + j] = s
        P[G - 1 - j, me + j] = -s
    if G % 2 == 1:
        P[(G - 1) // 2, me - 1] = 1.0
    return P


def _range_basis(P, tol=1e-9):
    """Orthonormal basis of the column space of a (small) projector, by
    modified Gram-Schmidt in column order (deterministic, continuous)."""
    vecs = []
    for j in range(P.shape[1]):
        v = P[:, j].copy()
        for _ in range(2):
            for u in vecs:
                v -= (u @ v) * u
        nv = np.linalg.norm(v)
        if nv > tol:
            vecs.append(v / nv)
    if not vecs:
        return np.zeros((P.shape[0], 0))
    return np.array(vecs).T


@dataclass
class Irrep:
    label: tuple            # (shape_odd, shape_even)
    f: int                  # dimension of the irrep = number of copies
    N: int                  # size of the block B_rho
    U: list                 # f CSC matrices (N_class x N), copy i of the basis
    U_csr: list = field(default_factory=list)  # same, CSR (row access)


@dataclass
class ParityClass:
    k: int                  # number of odd axes
    shape: tuple            # canonical reduced shape: (m_o,)*k + (m_e,)*(d-k)
    N: int
    X: np.ndarray           # (N, d) reduced multi-indices, C-order
    w: np.ndarray           # (N,) centre weights
    patterns: list          # parity patterns eps with k odd axes
    orders: list            # canonical axis order per pattern (odd axes first)
    irreps: list            # list[Irrep] with N > 0
    n_patterns: int = 0
    # Orbit representatives: B[x, y] = B[rep(x), h_x^{-1} . y] where x = h_x . rep(x),
    # so only the rows of the parity block at `reps` are ever computed.
    reps: np.ndarray = None   # (N_orb,) linear index of each orbit representative
    x_orb: np.ndarray = None  # (N,) orbit number of x
    x_ginv: np.ndarray = None # (N,) index (into act) of h_x^{-1}
    act: np.ndarray = None    # (|H|, N) act[h, lin(y)] = lin(h . y)


class GridSymmetry:
    """Block structure of isotropic covariances on the midpoint grid."""

    def __init__(self, d, G, verbose=False):
        self.d, self.G = int(d), int(G)
        self.n = self.G**self.d
        self.me, self.mo = (G + 1) // 2, G // 2
        self.P_axis = axis_parity_matrix(G)
        self.Dmax = d * (G - 1) ** 2
        self.classes = []
        for k in range(d + 1):
            cls = self._build_class(k, verbose)
            if cls is not None:
                self.classes.append(cls)
        self._check_completeness()

    # ---------------------------------------------------------------- build
    def _build_class(self, k, verbose):
        d, G, me, mo = self.d, self.G, self.me, self.mo
        shape = (mo,) * k + (me,) * (d - k)
        if min(shape, default=1) == 0:
            return None
        N = int(np.prod(shape))
        X = np.indices(shape).reshape(d, -1).T.astype(np.int64)
        w = np.ones(N)
        if G % 2 == 1:
            centre = me - 1
            for q in range(k, d):
                w *= np.where(X[:, q] == centre, 1.0 / math.sqrt(2.0), 1.0)
        patterns, orders = [], []
        for odd_axes in itertools.combinations(range(d), k):
            eps = tuple(1 if q in odd_axes else 0 for q in range(d))
            order = list(odd_axes) + [q for q in range(d) if q not in odd_axes]
            patterns.append(eps)
            orders.append(order)

        # group H = S_k x S_{d-k} acting on positions of the multi-index
        el1, irr1 = symmetric_group(k)
        el2, irr2 = symmetric_group(d - k)
        H = len(el1) * len(el2)
        perms = []
        for g1 in el1:
            for g2 in el2:
                perms.append(list(g1) + [k + g2[q] for q in range(d - k)])
        # action: (h.x)_{p(q)} = x_q ; act[h][lin(x)] = lin(h.x)
        act = np.empty((H, N), dtype=np.int64)
        for h, p in enumerate(perms):
            Y = np.empty_like(X)
            Y[:, p] = X
            act[h] = np.ravel_multi_index(Y.T, shape)
        # irreps of the product group
        prod_irreps = []
        for (s1, f1, rho1) in irr1:
            for (s2, f2, rho2) in irr2:
                f = f1 * f2
                # first column of rho(h) for every h (ordered like perms)
                col = np.empty((H, f))
                h = 0
                for a in range(len(el1)):
                    for b in range(len(el2)):
                        col[h] = np.kron(rho1[a][:, 0], rho2[b][:, 0])
                        h += 1
                prod_irreps.append(((s1, s2), f, col * (f / H)))

        # orbits = canonical representatives (sorted within each group)
        Xc = X.copy()
        Xc[:, :k] = np.sort(Xc[:, :k], axis=1)
        Xc[:, k:] = np.sort(Xc[:, k:], axis=1)
        key = np.ravel_multi_index(Xc.T, shape)
        _, inv = np.unique(key, return_inverse=True)
        order_idx = np.argsort(inv, kind="stable")
        bounds = np.flatnonzero(np.diff(inv[order_idx])) + 1
        orbits = np.split(order_idx, bounds)

        # representative of each orbit and, for every x, h_x^{-1} with x = h_x . rep
        pidx = {tuple(p): h for h, p in enumerate(perms)}
        hinv = np.empty(H, dtype=np.int64)
        for h, p in enumerate(perms):
            pinv = [0] * d
            for q in range(d):
                pinv[p[q]] = q
            hinv[h] = pidx[tuple(pinv)]
        reps = np.array([O[0] for O in orbits], dtype=np.int64)
        x_orb = np.empty(N, dtype=np.int64)
        x_ginv = np.empty(N, dtype=np.int64)
        for o, O in enumerate(orbits):
            images = act[:, O[0]]                                  # h . rep for all h
            h_x = np.argmax(images[:, None] == O[None, :], axis=0)  # first h with h.rep = x
            x_orb[O] = o
            x_ginv[O] = hinv[h_x]

        posmap = np.empty(N, dtype=np.int64)
        coo = [[[] for _ in range(f)] for (_, f, _) in prod_irreps]  # rows, cols, vals
        ncols = [0] * len(prod_irreps)
        for O in orbits:
            m = len(O)
            posmap[O] = np.arange(m)
            rows_h = posmap[act[:, O]]  # (H, m): position of h.y
            cols = np.arange(m)
            for r, (_, f, col) in enumerate(prod_irreps):
                Pm = np.zeros((f, m, m))
                for h in range(H):
                    Pm[:, rows_h[h], cols] += col[h][:, None]
                U1 = _range_basis(Pm[0])
                rr = U1.shape[1]
                if rr == 0:
                    continue
                for i in range(f):
                    Ui = U1 if i == 0 else Pm[i] @ U1
                    nz = np.abs(Ui) > 1e-14
                    ri, ci = np.nonzero(nz)
                    coo[r][i].append((O[ri], ci + ncols[r], Ui[ri, ci]))
                ncols[r] += rr

        irreps = []
        for r, (label, f, _) in enumerate(prod_irreps):
            if ncols[r] == 0:
                continue
            Us = []
            for i in range(f):
                rows = np.concatenate([c[0] for c in coo[r][i]])
                cols = np.concatenate([c[1] for c in coo[r][i]])
                vals = np.concatenate([c[2] for c in coo[r][i]])
                Us.append(sp.csc_matrix((vals, (rows, cols)), shape=(N, ncols[r])))
            irreps.append(Irrep(label=label, f=f, N=ncols[r], U=Us,
                                U_csr=[u.tocsr() for u in Us]))
        if verbose:
            print(f"  class k={k}: N={N}, |H|={H}, blocks "
                  + ", ".join(f"{ir.N}x{ir.f}" for ir in irreps), flush=True)
        return ParityClass(k=k, shape=shape, N=N, X=X, w=w, patterns=patterns,
                           orders=orders, irreps=irreps,
                           n_patterns=len(patterns), reps=reps, x_orb=x_orb,
                           x_ginv=x_ginv, act=act)

    def _check_completeness(self):
        tot = 0
        for cls in self.classes:
            s = sum(ir.N * ir.f for ir in cls.irreps)
            if s != cls.N:
                raise RuntimeError(f"incomplete decomposition in class k={cls.k}: {s} != {cls.N}")
            tot += cls.N * cls.n_patterns
        if tot != self.n:
            raise RuntimeError(f"parity classes cover {tot} != n={self.n}")

    # ------------------------------------------------------------- summary
    @property
    def blocks(self):
        """List of (class index, irrep index, N_rho, multiplicity)."""
        out = []
        for ci, cls in enumerate(self.classes):
            for ri, ir in enumerate(cls.irreps):
                out.append((ci, ri, ir.N, cls.n_patterns * ir.f))
        return out

    def describe(self):
        b = self.blocks
        return dict(n=self.n, n_distinct_blocks=len(b),
                    largest_block=max(x[2] for x in b),
                    sum_mult_N=sum(x[2] * x[3] for x in b))

    # ---------------------------------------------------------- transforms
    def _axis_apply(self, arr, M):
        for q in range(self.d):
            arr = np.moveaxis(np.tensordot(M, arr, axes=([1], [q])), 0, q)
        return arr

    def _slices(self, eps):
        me, G = self.me, self.G
        return tuple(slice(me, G) if e else slice(0, me) for e in eps)

    def forward(self, y):
        """Grid vector (length n, C-order) -> dict (ci, ri) -> (N_rho, mult)."""
        c = self._axis_apply(np.asarray(y, float).reshape((self.G,) * self.d),
                             self.P_axis.T)
        Z = {}
        for ci, cls in enumerate(self.classes):
            Y = np.empty((cls.N, cls.n_patterns))
            for p, (eps, order) in enumerate(zip(cls.patterns, cls.orders)):
                Y[:, p] = c[self._slices(eps)].transpose(order).reshape(-1)
            for ri, ir in enumerate(cls.irreps):
                Z[(ci, ri)] = np.concatenate([ir.U[i].T @ Y for i in range(ir.f)],
                                             axis=1)
        return Z

    def inverse(self, Z):
        """Inverse of `forward` (block coordinates -> grid vector)."""
        c = np.zeros((self.G,) * self.d)
        for ci, cls in enumerate(self.classes):
            Y = np.zeros((cls.N, cls.n_patterns))
            for ri, ir in enumerate(cls.irreps):
                Zb = Z[(ci, ri)]
                P = cls.n_patterns
                for i in range(ir.f):
                    Y += ir.U[i] @ Zb[:, i * P:(i + 1) * P]
            for p, (eps, order) in enumerate(zip(cls.patterns, cls.orders)):
                sub = Y[:, p].reshape(cls.shape).transpose(np.argsort(order))
                c[self._slices(eps)] = sub
        return self._axis_apply(c, self.P_axis).reshape(-1)

    def grid_diagonal(self, red_diag):
        """Map per-class reduced diagonals to the grid diagonal of Q M Q^T.

        red_diag[ci] is the length-N diagonal (canonical reduced coordinates)
        of U M U^T summed over irreps and copies of class ci.
        """
        G, d, mo = self.G, self.d, self.mo
        g = np.arange(G)
        centre = g == G - 1 - g
        je = np.minimum(g, G - 1 - g)              # even reduced index (centre -> m_e-1)
        jo = np.minimum(je, max(mo - 1, 0))        # odd reduced index (centre has weight 0)
        w2e = np.where(centre, 1.0, 0.5)
        w2o = np.where(centre, 0.0, 0.5)

        def along(v, q):
            return v.reshape([G if r == q else 1 for r in range(d)])

        out = np.zeros((G,) * d)
        for ci, cls in enumerate(self.classes):
            dk = red_diag[ci]
            for eps, order in zip(cls.patterns, cls.orders):
                idx = [along(jo if eps[q] else je, q) for q in order]
                lin = np.ravel_multi_index(np.broadcast_arrays(*idx), cls.shape)
                wt = np.ones((1,) * d)
                for q in range(d):
                    wt = wt * along(w2o if eps[q] else w2e, q)
                out += wt * dk[lin]
        return out.reshape(-1)
