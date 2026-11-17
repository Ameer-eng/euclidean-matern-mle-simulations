"""Observed negative Hessian at the truth, -d2 ell(theta0), for stored fits.

  python -m matern_sim.hess0_task --d 3 --nu0 0.5 --G 22 --reps 0:400 \
         --laws gauss,unif,mixnorm,bern --variants joint --family amd --cpus 4

Same arguments as kappa_task.  For every fit results/fits/<tag>/<variant>/<law>/
repNNNN.npz without results/hess0/<tag>/<variant>/<law>/repNNNN.npz, the
replicate's data are regenerated from its seeds exactly as in run_task, and the
observed Hessian at theta0 is computed by the same central differences as the
stored Hessian at theta_hat (Fitter.neg_hessian_fd: step HESS_STEP in the
optimiser's Fisher-scaled coordinates).  Used for the theta0 versions of the
Wald standard deviations (Section 4, Appendix D.7).

Batched: theta0 is the same for every replicate, so each of the eight
perturbed parameter points needs one Cholesky inverse per distinct block, which
then serves all the task's data sets at once (one matrix product per block and
derivative); the per-fit cost is a few matrix products.

Data check: the regenerated block data must reproduce the stored oracle-nugget
criterion loglik_oracle at the stored logt_oracle (relative tolerance 1e-9; the
same eigendecomposition generates the field, so a different eigenvector sign
changes this criterion by as much as it changes any quadratic form in the data).
Exact regeneration needs the CPU family and thread count of the fit; --family
and --cpus select those fits as in kappa_task.  A fit that fails the check gets
a marker results/hess0_mismatch/.../repNNNN.json and the task exits with status 3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np

from . import config
from .assemble import sum_prod
from .fit import Fitter
from .kappa_task import cpu_family, data_blocks, field_blocks, load_hosts
from .likelihood import GridModel, chol_logdet_inv, oracle_nugget_profile
from .run_task import (VARIANTS, atomic_savez, ensure_theory, fit_path, log,
                       maxrss_gb, tag, truth)

ORACLE_RTOL = 1e-9


def hess0_path(out, d, nu0, G, variant, law, rep):
    return os.path.join(out, "hess0", tag(d, nu0, G), variant, law, f"rep{rep:04d}.npz")


def hess0_mismatch_path(out, d, nu0, G, variant, law, rep):
    return os.path.join(out, "hess0_mismatch", tag(d, nu0, G), variant, law,
                        f"rep{rep:04d}.json")


def pending(out, d, nu0, G, reps, variants, laws, family="any", cpus=0, hosts=None):
    """[(rep, variant, law)] of fits with no hess0 file or marker, restricted (unless
    family == 'any') to fits made on `family` with `cpus` threads."""
    todo = []
    for rep in reps:
        for v in variants:
            for law in laws:
                if not os.path.exists(fit_path(out, d, nu0, G, v, law, rep)):
                    continue
                if os.path.exists(hess0_path(out, d, nu0, G, v, law, rep)) or \
                        os.path.exists(hess0_mismatch_path(out, d, nu0, G, v, law, rep)):
                    continue
                if family != "any":
                    fam, c = (hosts or {}).get((d, nu0, G, rep, law, v), ("unknown", 0))
                    if fam != family or (cpus and family != "unknown" and c != cpus):
                        continue
                todo.append((rep, v, law))
    return todo


def oracle_criterion(eig, Z, logt):
    """The oracle-nugget criterion of fit.oracle_nugget_fit at log t = logt."""
    lam, s2, m = oracle_nugget_profile(eig, Z)
    v = lam + math.exp(logt)
    return float(-0.5 * np.sum(m * np.log(v) + s2 / v))


def batched_gradients(model, theta, Zall, nfit):
    """Gradient of ell at theta for nfit data sets at once; Zall[key] has shape
    (N, nfit, mult).  Same arithmetic as GridModel.loglik_grad."""
    t = math.exp(theta[3])
    g = np.zeros((nfit, 4))
    for ci, ri, mult, S in model.iter_blocks(theta, ("K", "Knu", "Ka")):
        Zb = Zall[(ci, ri)]
        N = Zb.shape[0]
        B = S[0].copy()
        B[np.diag_indices_from(B)] += t
        _, Binv = chol_logdet_inv(B)
        W = Binv @ Zb.reshape(N, nfit * mult)
        for p in range(3):
            dB = S[p]
            det = mult * sum_prod(Binv, dB)
            q = np.sum((W * (dB @ W)).reshape(N, nfit, mult), axis=(0, 2))
            g[:, p] -= 0.5 * (det - q)
        q = np.sum((W * W).reshape(N, nfit, mult), axis=(0, 2))
        g[:, 3] -= 0.5 * t * (mult * np.trace(Binv) - q)
        del S, B, Binv, W
    return g


def run(args):
    d, nu0, G = args.d, args.nu0, args.G
    t_start = time.time()
    r0, r1 = (int(v) for v in args.reps.split(":"))
    laws, variants = args.laws.split(","), args.variants.split(",")
    here = cpu_family()
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    if args.family not in ("any", "unknown") and here != args.family:
        log(f"node {socket.gethostname()} is {here}, not {args.family}: refusing to run")
        return 2
    if args.cpus and threads != str(args.cpus):
        log(f"WARNING: OPENBLAS_NUM_THREADS={threads}, fits used {args.cpus} threads")
    hosts = load_hosts(args.out) if args.family != "any" else None
    todo = pending(args.out, d, nu0, G, range(r0, r1), variants, laws,
                   args.family, args.cpus, hosts)
    log(f"hess0 {tag(d, nu0, G)} reps {r0}:{r1} on {socket.gethostname()} ({here}), "
        f"threads={threads}, fits from family={args.family} "
        f"cpus={args.cpus or 'any'}: {len(todo)} fits to do")
    if not todo:
        return 0
    t = time.time()
    model = GridModel(d, G)
    theta0 = truth(d, nu0)
    I0 = ensure_theory(model, d, nu0, G, args.out, theta0, write=False)
    fitter = Fitter(model, theta0, nu0, I0)
    eig = model.signal_eigen(theta0)
    log(f"setup {time.time() - t:.1f}s: {model.sym.describe()}")

    nbad = 0
    for c0 in range(0, len(todo), args.batch):
        batch = todo[c0:c0 + args.batch]
        t = time.time()
        keys = sorted(eig)
        shapes = {k: (eig[k][0].shape[0], eig[k][2]) for k in keys}
        Zall = {k: np.empty((N, len(batch), m)) for k, (N, m) in shapes.items()}
        good, Zf, last_rep = [], None, None
        for (rep, variant, law) in batch:
            if rep != last_rep:
                Zf, last_rep = field_blocks(eig, d, nu0, G, rep), rep
            Z = data_blocks(model, Zf, d, nu0, G, rep, law)
            with np.load(fit_path(args.out, d, nu0, G, variant, law, rep)) as z:
                ll_or, lt_or = float(z["loglik_oracle"]), float(z["logt_oracle"])
            ll = oracle_criterion(eig, Z, lt_or)
            rel = abs(ll - ll_or) / max(1.0, abs(ll_or))
            if not rel <= ORACLE_RTOL:
                nbad += 1
                log(f"rep {rep} {law:7s} {variant:7s} DATA MISMATCH: oracle criterion "
                    f"{ll:.10f} vs stored {ll_or:.10f} (rel {rel:.1e}); not written")
                mp = hess0_mismatch_path(args.out, d, nu0, G, variant, law, rep)
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                with open(mp, "w") as fh:
                    json.dump(dict(host=socket.gethostname(), family=here,
                                   threads=threads, crit=ll, crit_fit=ll_or, rel=rel), fh)
                continue
            j = len(good)
            for k in keys:
                Zall[k][:, j, :] = Z[k]
            good.append((rep, variant, law, rel))
        if not good:
            continue
        nf = len(good)
        Zall = {k: np.ascontiguousarray(v[:, :nf, :]) for k, v in Zall.items()}
        t_data = time.time() - t

        # central differences of the gradient of -ell in the scaled x coordinates
        h = config.HESS_STEP
        Hx = np.empty((nf, 4, 4))
        for jj in range(4):
            e = np.zeros(4)
            e[jj] = h
            gp = batched_gradients(model, fitter.theta(e), Zall, nf)
            gm = batched_gradients(model, fitter.theta(-e), Zall, nf)
            # objective gradient in x is -(T^T g)
            Hx[:, :, jj] = -((gp - gm) @ fitter.T) / (2 * h)
        Hx = 0.5 * (Hx + np.transpose(Hx, (0, 2, 1)))
        negH0 = np.einsum("ji,mjk,kl->mil", fitter.Tinv, Hx, fitter.Tinv)
        t_hess = time.time() - t - t_data
        for i, (rep, variant, law, rel) in enumerate(good):
            atomic_savez(hess0_path(args.out, d, nu0, G, variant, law, rep),
                         negH0=negH0[i], theta0=theta0, oracle_rel_diff=rel,
                         d=d, nu0=nu0, G=G, n=model.n, rep=rep, law=law, variant=variant,
                         host=socket.gethostname(), family=here, threads=threads,
                         seconds_batch=time.time() - t, batch_size=nf)
        log(f"batch {c0 // args.batch + 1}: {nf} fits, data {t_data:.1f}s, "
            f"Hessians {t_hess:.1f}s, maxRSS {maxrss_gb():.1f} GB")
        del Zall
    log(f"task done in {(time.time() - t_start) / 60:.1f} min"
        + (f"; {nbad} DATA MISMATCHES" if nbad else ""))
    return 3 if nbad else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--nu0", type=float, required=True)
    p.add_argument("--G", type=int, required=True)
    p.add_argument("--reps", default="0:1", help="half-open range r0:r1")
    p.add_argument("--laws", default=",".join(config.LAWS))
    p.add_argument("--variants", default="joint")
    p.add_argument("--out", default="results")
    p.add_argument("--family", default="any")
    p.add_argument("--cpus", type=int, default=0)
    p.add_argument("--batch", type=int, default=800, help="data sets per batch")
    args = p.parse_args(argv)
    for v in args.variants.split(","):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
