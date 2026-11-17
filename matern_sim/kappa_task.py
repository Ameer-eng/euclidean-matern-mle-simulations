"""Residual fourth-cumulant estimate (eq. A.54) for stored fits.

  python -m matern_sim.kappa_task --d 3 --nu0 0.5 --G 22 --reps 0:20 \
         --laws gauss,unif,mixnorm,bern --variants joint --out results

Same arguments as run_task.  For every existing fit
results/fits/<tag>/<variant>/<law>/repNNNN.npz without a matching
results/kappa/<tag>/<variant>/<law>/repNNNN.npz, the replicate's data are
regenerated from its seeds exactly as in run_task, and kappa_tilde is computed at
the stored theta_hat.  The Gaussian log-likelihood at theta_hat is recomputed on
the regenerated data and must equal the stored one (relative tolerance 1e-8).

Exact regeneration needs the CPU family and thread count of the fit (see
fit_hosts.py).  With --family F --cpus C the task handles only the fits that
results/kappa/fit_hosts.csv attributes to family F and C threads, and refuses to
run on a node of another family.  A fit whose data still do not match gets a
marker results/kappa_mismatch/<tag>/<variant>/<law>/repNNNN.json, is skipped by
later plans, and makes the task exit with status 3.
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

from . import config, noise
from .kappa import ResidualCumulant
from .likelihood import GridModel, draw_field_blocks
from .run_task import VARIANTS, atomic_savez, fit_path, log, maxrss_gb, tag, truth

LL_RTOL = 1e-8
ABORT_AFTER = 3          # stop a task whose first fits all fail the data check


def kappa_path(out, d, nu0, G, variant, law, rep):
    return os.path.join(out, "kappa", tag(d, nu0, G), variant, law, f"rep{rep:04d}.npz")


def field_blocks(eig, d, nu0, G, rep):
    """Latent field of replicate `rep` in block coordinates (as in run_task)."""
    ss = [config.ROOT_SEED, d, int(round(10 * nu0)), G, rep]
    rng_f = np.random.Generator(np.random.PCG64(np.random.SeedSequence(ss + [0])))
    return draw_field_blocks(eig, rng_f)


def data_blocks(model, Zf, d, nu0, G, rep, law):
    """Block data Q^T Y of replicate `rep` under `law` (as in run_task)."""
    ss = [config.ROOT_SEED, d, int(round(10 * nu0)), G, rep]
    rng_e = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(ss + [1, noise.LAW_ID[law]])))
    eps = math.sqrt(config.TRUTH["tau2"]) * noise.sample(law, rng_e, model.n)
    Ze = model.sym.forward(eps)
    return {k: Zf[k] + Ze[k] for k in Zf}


def mismatch_path(out, d, nu0, G, variant, law, rep):
    return os.path.join(out, "kappa_mismatch", tag(d, nu0, G), variant, law,
                        f"rep{rep:04d}.json")


def cpu_family():
    """intel / intel_avx512 / amd / amd_avx512 for this node (cf. fit_hosts)."""
    vendor, avx512 = "intel", False
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("vendor_id") and "AMD" in line:
                    vendor = "amd"
                elif line.startswith("flags"):
                    avx512 = "avx512f" in line.split()
                    break
    except OSError:
        return "unknown"
    return vendor + ("_avx512" if avx512 else "")


def load_hosts(out):
    """{(d, nu0, G, rep, law, variant): (family, cpus)} from fit_hosts.csv."""
    import csv
    path = os.path.join(out, "kappa", "fit_hosts.csv")
    hosts = {}
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                cpus = int(float(r["cpus"])) if r["cpus"] not in ("", "nan") else 0
                hosts[(int(r["d"]), float(r["nu0"]), int(r["G"]), int(r["rep"]), r["law"],
                       r["variant"])] = (r["family"], cpus)
    return hosts


def pending(out, d, nu0, G, reps, variants, laws, family="any", cpus=0, hosts=None):
    """{rep: [(variant, law)]} of fits that exist, have no kappa file and no
    mismatch marker, and (unless family == 'any') were made on `family` with
    `cpus` threads according to `hosts`.  family == 'unknown' selects the fits
    whose node is not in `hosts` (they run on any node)."""
    todo = {}
    for rep in reps:
        jobs = []
        for v in variants:
            for law in laws:
                if not os.path.exists(fit_path(out, d, nu0, G, v, law, rep)):
                    continue
                if os.path.exists(kappa_path(out, d, nu0, G, v, law, rep)) or \
                        os.path.exists(mismatch_path(out, d, nu0, G, v, law, rep)):
                    continue
                if family != "any":
                    fam, c = (hosts or {}).get((d, nu0, G, rep, law, v), ("unknown", 0))
                    if fam != family or (cpus and family != "unknown" and c != cpus):
                        continue
                jobs.append((v, law))
        if jobs:
            todo[rep] = jobs
    return todo


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
    log(f"kappa {tag(d, nu0, G)} reps {r0}:{r1} on {socket.gethostname()} ({here}), "
        f"threads={threads}, fits from family={args.family} cpus={args.cpus or 'any'}: "
        f"{sum(len(j) for j in todo.values())} fits to do")
    if not todo:
        return 0
    t = time.time()
    model = GridModel(d, G)
    theta0 = truth(d, nu0)
    eig = model.signal_eigen(theta0)
    rc = ResidualCumulant(model)
    log(f"setup {time.time() - t:.1f}s: {model.sym.describe()}, "
        f"{len(rc.rep_lin)} orbits, row buffer {rc.memory_gb():.2f} GB")
    nbad = ngood = 0
    for rep, jobs in todo.items():
        if nbad >= ABORT_AFTER and ngood == 0:
            log(f"first {nbad} fits all mismatched on this node: giving up "
                "(remaining fits stay pending)")
            break
        Zf = field_blocks(eig, d, nu0, G, rep)
        for law in dict.fromkeys(law for _, law in jobs):
            Z = data_blocks(model, Zf, d, nu0, G, rep, law)
            for variant in (v for v, l in jobs if l == law):
                t = time.time()
                fp = fit_path(args.out, d, nu0, G, variant, law, rep)
                with np.load(fp) as z:
                    theta_hat = np.array(z["theta_hat"], float)
                    ll_fit = float(z["loglik"])
                res = rc.compute(theta_hat, Z)
                rel = abs(res["loglik"] - ll_fit) / max(1.0, abs(ll_fit))
                if not rel <= LL_RTOL:
                    nbad += 1
                    log(f"rep {rep} {law:7s} {variant:7s} DATA MISMATCH: loglik "
                        f"{res['loglik']:.6f} vs stored {ll_fit:.6f} (rel {rel:.1e}); "
                        "not written")
                    mp = mismatch_path(args.out, d, nu0, G, variant, law, rep)
                    os.makedirs(os.path.dirname(mp), exist_ok=True)
                    with open(mp, "w") as fh:
                        json.dump(dict(host=socket.gethostname(), family=here,
                                       threads=threads, loglik=res["loglik"],
                                       loglik_fit=ll_fit, rel=rel), fh)
                    continue
                ngood += 1
                atomic_savez(
                    kappa_path(args.out, d, nu0, G, variant, law, rep),
                    kappa_resid=res["kappa"], numer=res["numer"], denom=res["denom"],
                    denom_over_n=res["denom_over_n"], t_hat=res["t_hat"],
                    sum_r4=res["sum_r4"], sum_v=res["sum_v"],
                    loglik=res["loglik"], loglik_fit=ll_fit, loglik_rel_diff=rel,
                    diag_check=res["diag_check"], theta_hat=theta_hat,
                    d=d, nu0=nu0, G=G, n=model.n, rep=rep, law=law, variant=variant,
                    kappa4_true=noise.kappa4(law, config.TRUTH["tau2"]),
                    host=socket.gethostname(), family=here, threads=threads,
                    seconds=time.time() - t)
                log(f"rep {rep} {law:7s} {variant:7s} kappa={res['kappa']:+.4f} "
                    f"(true {noise.kappa4(law, config.TRUTH['tau2']):+.4f}) "
                    f"ll diff {rel:.0e}  {time.time() - t:.1f}s  "
                    f"maxRSS {maxrss_gb():.1f} GB")
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
    p.add_argument("--family", default="any",
                   help="only fits made on this CPU family (see fit_hosts.py); "
                        "'unknown': only fits whose node is unknown; 'any': all")
    p.add_argument("--cpus", type=int, default=0,
                   help="only fits made with this many threads (0 = any)")
    args = p.parse_args(argv)
    for v in args.variants.split(","):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
