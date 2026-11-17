"""AUDIT (2026-11-09): gradient of ell at theta0 for stored fits.

Mirrors matern_sim.hess0_task (same data regeneration, same oracle-criterion data
check, same --family/--cpus pinning) but evaluates only the score at theta0, in the
spectral chart theta = (eta, nu, log a, log t).  Needed to form the observed Hessian
at theta0 directly in the reported log coordinates,
    H_zeta = G^T H_theta G - sum_j (d ell / d theta_j) D^2 F_j(zeta),
whose score term the production code (aggregate.py) omits.

  python -m audit.score0_task --d 5 --nu0 0.5 --G 10 --reps 0:400 --laws gauss \
         --family amd_avx512 --cpus 8 --save audit/out/score0
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

from matern_sim.hess0_task import batched_gradients, oracle_criterion
from matern_sim.kappa_task import cpu_family, data_blocks, field_blocks, load_hosts
from matern_sim.likelihood import GridModel
from matern_sim.run_task import atomic_savez, fit_path, log, maxrss_gb, tag, truth

ORACLE_RTOL = 1e-9
LOCAL = "audit/out/score0_local"


def spath(save, d, nu0, G, variant, law, rep, mismatch=False):
    return os.path.join(save, "mismatch" if mismatch else "ok", tag(d, nu0, G), variant,
                        law, f"rep{rep:04d}." + ("json" if mismatch else "npz"))


def pending(out, save, d, nu0, G, reps, variants, laws, family, cpus, hosts):
    todo = []
    for rep in reps:
        for v in variants:
            for law in laws:
                if not os.path.exists(fit_path(out, d, nu0, G, v, law, rep)):
                    continue
                if os.path.exists(spath(save, d, nu0, G, v, law, rep)) or \
                        os.path.exists(spath(save, d, nu0, G, v, law, rep, True)):
                    continue
                # already computed (data check passed) by the local run
                if save != LOCAL and os.path.exists(spath(LOCAL, d, nu0, G, v, law, rep)):
                    continue
                if family != "any":
                    fam, c = (hosts or {}).get((d, nu0, G, rep, law, v), ("unknown", 0))
                    if fam != family or (cpus and family != "unknown" and c != cpus):
                        continue
                todo.append((rep, v, law))
    return todo


def run(a):
    d, nu0, G = a.d, a.nu0, a.G
    t_start = time.time()
    r0, r1 = (int(v) for v in a.reps.split(":"))
    here = cpu_family()
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    if a.family not in ("any", "unknown") and here != a.family:
        log(f"node {socket.gethostname()} is {here}, not {a.family}: refusing to run")
        return 2
    hosts = load_hosts(a.out) if a.family != "any" else None
    todo = pending(a.out, a.save, d, nu0, G, range(r0, r1), a.variants.split(","),
                   a.laws.split(","), a.family, a.cpus, hosts)
    log(f"score0 {tag(d, nu0, G)} reps {r0}:{r1} on {socket.gethostname()} ({here}) "
        f"threads={threads}: {len(todo)} fits")
    if not todo:
        return 0
    model = GridModel(d, G)
    theta0 = truth(d, nu0)
    eig = model.signal_eigen(theta0)
    nbad = 0
    for c0 in range(0, len(todo), a.batch):
        batch = todo[c0:c0 + a.batch]
        keys = sorted(eig)
        Zall = {k: np.empty((eig[k][0].shape[0], len(batch), eig[k][2])) for k in keys}
        good, last = [], None
        for (rep, v, law) in batch:
            if rep != last:
                Zf, last = field_blocks(eig, d, nu0, G, rep), rep
            Z = data_blocks(model, Zf, d, nu0, G, rep, law)
            with np.load(fit_path(a.out, d, nu0, G, v, law, rep)) as z:
                ll_or, lt_or = float(z["loglik_oracle"]), float(z["logt_oracle"])
            rel = abs(oracle_criterion(eig, Z, lt_or) - ll_or) / max(1.0, abs(ll_or))
            if not rel <= ORACLE_RTOL:
                nbad += 1
                p = spath(a.save, d, nu0, G, v, law, rep, True)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    json.dump(dict(host=socket.gethostname(), family=here, rel=rel), fh)
                continue
            for k in keys:
                Zall[k][:, len(good), :] = Z[k]
            good.append((rep, v, law, rel))
        if not good:
            continue
        nf = len(good)
        Zall = {k: np.ascontiguousarray(x[:, :nf, :]) for k, x in Zall.items()}
        g0 = batched_gradients(model, theta0, Zall, nf)      # d ell / d theta at theta0
        for i, (rep, v, law, rel) in enumerate(good):
            atomic_savez(spath(a.save, d, nu0, G, v, law, rep), score0=g0[i],
                         theta0=theta0, oracle_rel_diff=rel, d=d, nu0=nu0, G=G,
                         n=model.n, rep=rep, law=law, variant=v,
                         host=socket.gethostname(), family=here, threads=threads)
        log(f"batch {c0 // a.batch + 1}: {nf} fits, maxRSS {maxrss_gb():.1f} GB")
        del Zall
    log(f"done in {(time.time() - t_start) / 60:.1f} min; {nbad} mismatches")
    return 3 if nbad else 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--nu0", type=float, required=True)
    p.add_argument("--G", type=int, required=True)
    p.add_argument("--reps", default="0:1")
    p.add_argument("--laws", default="gauss,unif,mixnorm,bern")
    p.add_argument("--variants", default="joint")
    p.add_argument("--out", default="results")
    p.add_argument("--save", default="audit/out/score0")
    p.add_argument("--family", default="any")
    p.add_argument("--cpus", type=int, default=0)
    p.add_argument("--batch", type=int, default=800)
    sys.exit(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
