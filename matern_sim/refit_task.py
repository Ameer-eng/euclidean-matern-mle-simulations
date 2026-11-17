"""Restart rule for fits that stopped short of stationarity (added 2026-11-10).

A saved joint fit is flagged when its KKT residual -- the largest projected
gradient of -ell in the optimiser coordinates x (L-BFGS-B convention at active
bounds) -- exceeds PG_TOL = 1e-3.  In the 67,200-fit campaign this flags 143
fits, all at n <= 10^4, all stopped by the relative-reduction (ftol) test.

For each flagged fit the replicate's data are regenerated from its seeds and
checked as in kappa_task (Gaussian log-likelihood at the stored theta_hat, relative
tolerance 1e-8).  L-BFGS-B is then restarted, with the production scaling, box and
settings (same Fitter as run_task), from
  RESTART  the stopping point of the production fit,
  S1-S3    (sigma2, l, nu, tau2) = (4, 0.1, 2.5, 0.02), (0.25, 1.5, 0.3, 2), (2, 0.8, 1, 0.05),
  DATA     nu = 1, l = 0.25, tau2 = t_pair, sigma2 = max(mean y^2 - t_pair, 0.1 mean y^2).
The candidate with the highest log-likelihood (production fit included) is kept;
while its KKT residual is still > PG_TOL it is restarted from itself (at most
MAX_POLISH times, stopping when the log-likelihood no longer increases).  Since
RESTART starts at the production estimate and L-BFGS-B never increases -ell, the
kept log-likelihood is >= the production one.

Outputs, for every flagged fit whose data check passes:
  results/fits/...      replaced (theta_hat, x_hat, loglik, grad, success, message,
                        nit, at_bound, negH, fisher_hat from the kept candidate;
                        nfev stays the production count; refit_* fields added)
  results/kappa/...     replaced (kappa_tilde at the new theta_hat)
  results/fits_prerefit/..., results/kappa_prerefit/...   the original files
A fit whose data do not reproduce on this node gets a marker
results/refit_mismatch/<tag>/joint/<law>/repNNNN.json (family recorded) and is
retried only on a node of a different CPU family.

  python -m matern_sim.refit_task --list            # flagged fits and their node family
  python -m matern_sim.refit_task --family amd --cpus 1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import socket
import sys
import time

import numpy as np

from . import config, noise
from .fit import Fitter
from .kappa import ResidualCumulant
from .kappa_task import (LL_RTOL, cpu_family, data_blocks, field_blocks, kappa_path,
                         load_hosts)
from .kernel import theta_from_public
from .likelihood import GridModel
from .run_task import atomic_savez, fit_path, log, tag, truth

PG_TOL = 1e-3
MAX_POLISH = 3
FAR = {"S1": (4.0, 0.1, 2.5, 0.02), "S2": (0.25, 1.5, 0.3, 2.0), "S3": (2.0, 0.8, 1.0, 0.05)}


def kkt_residual(F, x, grad_theta):
    """max |projected gradient| of -ell in x (L-BFGS-B convention at active bounds)."""
    gx = -(F.T.T @ np.asarray(grad_theta, float))
    pg = gx.copy()
    lo, hi = x <= F.xlo + 1e-10, x >= F.xhi - 1e-10
    pg[lo] = np.minimum(gx[lo], 0.0)
    pg[hi] = np.maximum(gx[hi], 0.0)
    return float(np.max(np.abs(pg)))


def mismatch_path(out, d, nu0, G, law, rep):
    return os.path.join(out, "refit_mismatch", tag(d, nu0, G), "joint", law,
                        f"rep{rep:04d}.json")


def flagged(out):
    """[(d, nu0, G, law, rep, pg)] of joint fits with KKT residual > PG_TOL that
    have not been refitted yet (fits with refit_* fields are skipped)."""
    fitters, res = {}, []
    for f in sorted(glob.glob(os.path.join(out, "fits", "*", "joint", "*", "rep*.npz"))):
        with np.load(f) as z:
            if "refit" in z.files:
                continue
            d, nu0, G = int(z["d"]), float(z["nu0"]), int(z["G"])
            key = (d, nu0, G)
            if key not in fitters:
                th = np.load(os.path.join(out, "theory", tag(d, nu0, G) + ".npz"))
                fitters[key] = Fitter(_Shape(d, int(z["n"])), th["theta0"], nu0, th["I0"])
            pg = kkt_residual(fitters[key], z["x_hat"], z["grad"])
            if pg > PG_TOL:
                res.append((d, nu0, G, str(z["law"]), int(z["rep"]), pg))
    return res


class _Shape:
    """Minimal stand-in for GridModel when only the Fitter's box/scaling is needed."""
    def __init__(self, d, n):
        self.d, self.n = d, n


def refit_one(model, F, rc, eig, d, nu0, G, law, rep, out):
    fp = fit_path(out, d, nu0, G, "joint", law, rep)
    with np.load(fp) as z:
        orig = {k: z[k] for k in z.files}
    Z = data_blocks(model, field_blocks(eig, d, nu0, G, rep), d, nu0, G, rep, law)
    ll_chk = model.loglik_grad(orig["theta_hat"], Z, grad=False)
    rel = abs(ll_chk - float(orig["loglik"])) / max(1.0, abs(float(orig["loglik"])))
    if not rel <= LL_RTOL:
        return dict(status="mismatch", rel=rel)
    pg0 = kkt_residual(F, orig["x_hat"], orig["grad"])
    t0 = time.time()
    y2 = sum(float(np.sum(v * v)) for v in Z.values()) / model.n
    t_pair = float(orig["t_pair"])
    th_data = theta_from_public(d, max(y2 - t_pair, 0.1 * y2), 0.25, 1.0,
                                min(max(t_pair, 1.1e-3), 9.9))
    starts = {"RESTART": np.array(orig["x_hat"], float)}
    for k, (s, l, nu, t) in FAR.items():
        starts[k] = F.x_of(theta_from_public(d, s, l, nu, t))
    starts["DATA"] = F.x_of(th_data)
    cands = {"PRODUCTION": dict(loglik=float(orig["loglik"]), pg=pg0, nfev=0)}
    results = {}
    nfev = 0
    for name, x0 in starts.items():
        r = F.fit(Z, x_start=x0)
        nfev += r["nfev"]
        results[name] = r
        cands[name] = dict(loglik=float(r["loglik"]), pg=kkt_residual(F, r["x"], r["grad"]),
                           nfev=int(r["nfev"]))
    best = max(results, key=lambda k: results[k]["loglik"])
    r = results[best]
    if r["loglik"] < float(orig["loglik"]):          # cannot happen (RESTART), kept as a guard
        return dict(status="no_gain", rel=rel)
    polish = 0
    while kkt_residual(F, r["x"], r["grad"]) > PG_TOL and polish < MAX_POLISH:
        r2 = F.fit(Z, x_start=r["x"])
        nfev += r2["nfev"]
        polish += 1
        cands[f"POLISH{polish}"] = dict(loglik=float(r2["loglik"]),
                                        pg=kkt_residual(F, r2["x"], r2["grad"]),
                                        nfev=int(r2["nfev"]))
        if not r2["loglik"] > r["loglik"]:
            break
        r = r2
    th = r["theta"]
    negH = F.neg_hessian_fd(Z, th)
    fisher_hat = model.fisher(th)
    kr = rc.compute(th, Z)
    krel = abs(kr["loglik"] - r["loglik"]) / max(1.0, abs(r["loglik"]))
    pg1 = kkt_residual(F, r["x"], r["grad"])
    here = cpu_family()
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    new = dict(orig)
    new.update(theta_hat=th, x_hat=r["x"], loglik=r["loglik"], grad=r["grad"],
               success=r["success"], message=r["message"], nit=r["nit"],
               at_bound=r["at_bound"], negH=negH, fisher_hat=fisher_hat,
               refit=True, refit_start=best, refit_polish=polish,
               refit_pg_tol=PG_TOL, refit_pg_before=pg0, refit_pg_after=pg1,
               loglik_prerefit=orig["loglik"], theta_hat_prerefit=orig["theta_hat"],
               refit_nfev=nfev, refit_seconds=time.time() - t0,
               refit_data_rel=rel, refit_candidates=json.dumps(cands),
               refit_host=socket.gethostname(), refit_family=here, refit_threads=threads)
    # back up the originals, then replace fit and kappa files
    kp = kappa_path(out, d, nu0, G, "joint", law, rep)
    for src, sub in ((fp, "fits"), (kp, "kappa")):
        dst = src.replace(os.sep + sub + os.sep, os.sep + sub + "_prerefit" + os.sep, 1)
        if os.path.exists(src) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    atomic_savez(fp, **new)
    atomic_savez(kp, kappa_resid=kr["kappa"], numer=kr["numer"], denom=kr["denom"],
                 denom_over_n=kr["denom_over_n"], t_hat=kr["t_hat"], sum_r4=kr["sum_r4"],
                 sum_v=kr["sum_v"], loglik=kr["loglik"], loglik_fit=r["loglik"],
                 loglik_rel_diff=krel, diag_check=kr["diag_check"], theta_hat=th,
                 d=d, nu0=nu0, G=G, n=model.n, rep=rep, law=law, variant="joint",
                 kappa4_true=noise.kappa4(law, config.TRUTH["tau2"]),
                 host=socket.gethostname(), family=here, threads=threads, refit=True)
    return dict(status="ok", rel=rel, best=best, gain=r["loglik"] - float(orig["loglik"]),
                pg0=pg0, pg1=pg1, polish=polish, nfev=nfev, krel=krel)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    p.add_argument("--family", default="any",
                   help="only fits made on this CPU family (fit_hosts.csv); 'any': all")
    p.add_argument("--cpus", type=int, default=0, help="only fits made with this many threads")
    p.add_argument("--list", action="store_true", help="list flagged fits and exit")
    a = p.parse_args(argv)
    here = cpu_family()
    hosts = load_hosts(a.out)
    todo = []
    for (d, nu0, G, law, rep, pg) in flagged(a.out):
        fam, c = hosts.get((d, nu0, G, rep, law, "joint"), ("unknown", 0))
        mp = mismatch_path(a.out, d, nu0, G, law, rep)
        if os.path.exists(mp):
            with open(mp) as fh:
                if json.load(fh).get("family") == here:
                    continue
        if a.family != "any" and (fam != a.family or (a.cpus and c != a.cpus)):
            continue
        todo.append((d, nu0, G, law, rep, pg, fam, c))
    if a.list:
        for t in todo:
            print(*t)
        print(f"{len(todo)} flagged fits")
        return 0
    if a.family not in ("any", "unknown") and here != a.family:
        log(f"node {socket.gethostname()} is {here}, not {a.family}: refusing to run")
        return 2
    log(f"refit on {socket.gethostname()} ({here}, threads "
        f"{os.environ.get('OPENBLAS_NUM_THREADS', '?')}): {len(todo)} flagged fits")
    nbad = 0
    cache = {}
    for (d, nu0, G, law, rep, pg, fam, c) in sorted(todo):
        key = (d, nu0, G)
        if key not in cache:
            cache.clear()
            model = GridModel(d, G)
            theta0 = truth(d, nu0)
            I0 = np.load(os.path.join(a.out, "theory", tag(d, nu0, G) + ".npz"))["I0"]
            cache[key] = (model, Fitter(model, theta0, nu0, I0), ResidualCumulant(model),
                          model.signal_eigen(theta0))
        model, F, rc, eig = cache[key]
        r = refit_one(model, F, rc, eig, d, nu0, G, law, rep, a.out)
        if r["status"] == "mismatch":
            nbad += 1
            mp = mismatch_path(a.out, d, nu0, G, law, rep)
            os.makedirs(os.path.dirname(mp), exist_ok=True)
            with open(mp, "w") as fh:
                json.dump(dict(host=socket.gethostname(), family=here, rel=r["rel"],
                               fit_family=fam, fit_cpus=c), fh)
            log(f"{tag(d, nu0, G)} {law:7s} rep {rep}: DATA MISMATCH (rel {r['rel']:.1e}), "
                f"fit made on {fam}/{c}; skipped")
            continue
        log(f"{tag(d, nu0, G)} {law:7s} rep {rep}: pg {r.get('pg0', pg):.2e} -> "
            f"{r.get('pg1', float('nan')):.2e}, best {r.get('best')}, "
            f"gain {r.get('gain', float('nan')):.3e}, polish {r.get('polish')}, "
            f"nfev {r.get('nfev')}, kappa ll check {r.get('krel', float('nan')):.0e}")
    log(f"done; {nbad} data mismatches")
    return 3 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
