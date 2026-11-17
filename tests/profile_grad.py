"""Stage-by-stage timing of one log-likelihood + gradient evaluation.

  NUMBA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m tests.profile_grad 3 32
"""
import math
import sys
import time

import numpy as np

from matern_sim.assemble import assemble_rows, project_block_rep, sum_prod
from matern_sim.kernel import matern_tables
from matern_sim.likelihood import GridModel, chol_logdet_inv
from matern_sim.run_task import truth


def main():
    d, G = int(sys.argv[1]), int(sys.argv[2])
    m = GridModel(d, G)
    th = truth(d, 0.5)
    Z = m.sym.forward(np.random.default_rng(0).standard_normal(m.n))
    m.loglik_grad(th, Z)  # JIT + warm-up
    T = dict(tables=0.0, assemble=0.0, project=0.0, chol_inv=0.0, traces=0.0, gemm=0.0)
    t_all = time.time()
    t = time.time()
    tabs = matern_tables(th, d, G, m.Dmax, ("K", "Knu", "Ka"))
    T["tables"] += time.time() - t
    tt = math.exp(th[3])
    for ci, cls in enumerate(m.sym.classes):
        t = time.time()
        P = np.empty((3, len(cls.reps), cls.N))
        assemble_rows(cls.X, cls.reps, cls.k, G, cls.w, tabs, P)
        T["assemble"] += time.time() - t
        for ri, ir in enumerate(cls.irreps):
            t = time.time()
            U = ir.U[0]
            Tb = np.empty((ir.N, cls.N))
            S = np.empty((3, ir.N, ir.N))
            for tb in range(3):
                project_block_rep(P[tb], cls.x_orb, cls.x_ginv, cls.act,
                                  U.indptr, U.indices, U.data, Tb, S[tb])
            T["project"] += time.time() - t
            t = time.time()
            B = S[0].copy()
            B[np.diag_indices_from(B)] += tt
            logdet, Binv = chol_logdet_inv(B)
            T["chol_inv"] += time.time() - t
            Zb = Z[(ci, ri)]
            t = time.time()
            W = Binv @ Zb
            DW = [S[p] @ W for p in range(3)]
            T["gemm"] += time.time() - t
            t = time.time()
            for p in range(3):
                sum_prod(Binv, S[p])
                np.sum(W * DW[p])
            np.trace(Binv)
            T["traces"] += time.time() - t
    tot = time.time() - t_all
    print(f"d={d} G={G} n={m.n} largest_block={m.sym.describe()['largest_block']} "
          f"total={tot:.2f}s  " + "  ".join(f"{k}={v:.2f}" for k, v in T.items()), flush=True)


if __name__ == "__main__":
    main()
