"""Run one task: replicates [r0, r1) of setting (d, nu0, G) for a list of error laws.

  python -m matern_sim.run_task --d 3 --nu0 0.5 --G 22 --reps 0:20 \
         --laws gauss,unif,mixnorm,bern --out results

Outputs (one file per fit, written atomically; existing files are skipped, so a
re-submitted task resumes where it stopped):
  results/fits/<tag>/<variant>/<law>/repNNNN.npz
  results/theory/<tag>.npz      I_n(theta0) and kurtosis matrix (once per setting)
Modes:  --theory-only (only the theory file)   --bench (timing for the manifest)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import sys
import time

import numpy as np

from . import config, noise
from .fit import Fitter, oracle_nugget_fit, pair_estimates
from .kernel import theta_from_public
from .likelihood import GridModel, draw_field_blocks, oracle_nugget_profile

VARIANTS = ("joint", "knownnu")


def tag(d, nu0, G):
    return f"d{d}_nu{nu0:g}_G{G}"


def fit_path(out, d, nu0, G, variant, law, rep):
    return os.path.join(out, "fits", tag(d, nu0, G), variant, law, f"rep{rep:04d}.npz")


def theory_path(out, d, nu0, G):
    return os.path.join(out, "theory", tag(d, nu0, G) + ".npz")


def atomic_savez(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **kw)
    os.replace(tmp, path)


def maxrss_gb():
    """Peak resident memory in GB (ru_maxrss is in KB on Linux, bytes on macOS)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3 if sys.platform == "darwin" else 1024**2)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def truth(d, nu0):
    return theta_from_public(d, config.TRUTH["sigma2"], config.TRUTH["ell"], nu0,
                             config.TRUTH["tau2"])


def ensure_theory(model, d, nu0, G, out, theta0, write=True):
    """I_n(theta0) (needed to scale the optimiser).  With write=True (theory
    stage) also compute the kurtosis matrix and save the theory file; fit tasks
    use write=False so concurrent array tasks do not all repeat that work."""
    path = theory_path(out, d, nu0, G)
    if os.path.exists(path):
        th = np.load(path)
        return th["I0"]
    if not write:
        return model.fisher(theta0)
    t = time.time()
    I0, Dk0 = model.fisher(theta0, kurtosis_matrix=True)
    atomic_savez(path, I0=I0, Dk0=Dk0, theta0=theta0, n=model.n, d=d, G=G, nu0=nu0,
                 tau2=config.TRUTH["tau2"], Ln=math.log(model.n) / (2 * nu0 + d),
                 blocks=np.array(model.sym.blocks))
    log(f"theory written ({time.time() - t:.1f}s): {path}")
    return I0


def run(args):
    d, nu0, G = args.d, args.nu0, args.G
    t_start = time.time()
    log(f"setting {tag(d, nu0, G)} on {socket.gethostname()}, "
        f"threads={os.environ.get('NUMBA_NUM_THREADS', '?')}")
    model = GridModel(d, G)
    log(f"symmetry built: {model.sym.describe()}")
    theta0 = truth(d, nu0)
    I0 = ensure_theory(model, d, nu0, G, args.out, theta0, write=args.theory_only)
    if args.theory_only:
        return
    laws = args.laws.split(",")
    variants = args.variants.split(",")
    r0, r1 = (int(v) for v in args.reps.split(":"))

    todo = {rep: [(v, law) for v in variants for law in laws
                  if not os.path.exists(fit_path(args.out, d, nu0, G, v, law, rep))]
            for rep in range(r0, r1)}
    todo = {rep: jobs for rep, jobs in todo.items() if jobs}
    if not todo:
        log("nothing to do")
        return
    t = time.time()
    eig = model.signal_eigen(theta0)
    fitter = Fitter(model, theta0, nu0, I0)
    tau0 = math.sqrt(config.TRUTH["tau2"])
    log(f"signal eigendecomposition + fitter ready ({time.time() - t:.1f}s); "
        f"{sum(len(j) for j in todo.values())} fits to do")

    for rep, jobs in todo.items():
        ss = [config.ROOT_SEED, d, int(round(10 * nu0)), G, rep]
        rng_f = np.random.Generator(np.random.PCG64(np.random.SeedSequence(ss + [0])))
        Zf = draw_field_blocks(eig, rng_f)
        Fgrid = model.sym.inverse(Zf)
        for law in dict.fromkeys(law for _, law in jobs):
            rng_e = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence(ss + [1, noise.LAW_ID[law]])))
            eps = tau0 * noise.sample(law, rng_e, model.n)
            Ze = model.sym.forward(eps)
            Z = {k: Zf[k] + Ze[k] for k in Zf}
            y = Fgrid + eps
            t_pair, k_pair, m_pairs = pair_estimates(y, G, d)
            lt_or, ll_or = oracle_nugget_fit(*oracle_nugget_profile(eig, Z))
            for variant in (v for v, l in jobs if l == law):
                t = time.time()
                res = fitter.fit(Z, fix_nu=(variant == "knownnu"))
                t_fit = time.time() - t
                negH = fitter.neg_hessian_fd(Z, res["theta"])
                fisher_hat = model.fisher(res["theta"])
                atomic_savez(
                    fit_path(args.out, d, nu0, G, variant, law, rep),
                    theta_hat=res["theta"], x_hat=res["x"], loglik=res["loglik"],
                    grad=res["grad"], success=res["success"], message=res["message"],
                    nit=res["nit"], nfev=res["nfev"], at_bound=res["at_bound"],
                    negH=negH, fisher_hat=fisher_hat,
                    t_pair=t_pair, kappa_pair=k_pair, m_pairs=m_pairs,
                    logt_oracle=lt_or, loglik_oracle=ll_or,
                    theta0=theta0, d=d, nu0=nu0, G=G, n=model.n, rep=rep, law=law,
                    variant=variant, kappa4_true=noise.kappa4(law, config.TRUTH["tau2"]),
                    m4_true=noise.MOMENTS[law]["m4"], seed_root=config.ROOT_SEED,
                    seconds_fit=t_fit, seconds_total=time.time() - t)
                log(f"rep {rep} {law:7s} {variant:7s} nfev={res['nfev']:3d} "
                    f"conv={res['success']} bound={res['at_bound'].astype(int)} "
                    f"fit {t_fit:.0f}s total {time.time() - t:.0f}s  "
                    f"maxRSS {maxrss_gb():.1f} GB")
    log(f"task done in {(time.time() - t_start) / 60:.1f} min")


def bench(args):
    """Time one log-likelihood+gradient and one Fisher evaluation at theta0."""
    d, nu0, G = args.d, args.nu0, args.G
    warm = GridModel(d, 3)                                   # JIT warm-up
    warm.loglik_grad(truth(d, nu0), warm.sym.forward(np.ones(3**d)))
    warm.fisher(truth(d, nu0), kurtosis_matrix=True)
    t = time.time()
    model = GridModel(d, G)
    t_sym = time.time() - t
    theta0 = truth(d, nu0)
    Z = model.sym.forward(np.random.default_rng(0).standard_normal(model.n))
    t = time.time()
    model.loglik_grad(theta0, Z)
    t_grad = time.time() - t
    t = time.time()
    model.loglik_grad(theta0, Z)
    t_grad2 = time.time() - t
    t = time.time()
    model.fisher(theta0)
    t_fisher = time.time() - t
    t = time.time()
    model.signal_eigen(theta0)
    t_eig = time.time() - t
    info = dict(d=d, nu0=nu0, G=G, n=model.n, threads=os.environ.get("NUMBA_NUM_THREADS"),
                host=socket.gethostname(), cpu=_cpu_model(), t_symmetry=t_sym,
                t_grad=min(t_grad, t_grad2), t_fisher=t_fisher, t_eigen=t_eig,
                maxrss_gb=maxrss_gb(),
                **{k: v for k, v in model.sym.describe().items() if k != "n"})
    path = os.path.join(args.out, "bench", tag(d, nu0, G) + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(info, fh, indent=1)
    log(json.dumps(info))


def _cpu_model():
    if sys.platform == "darwin":
        try:
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or "?"
        except OSError:
            return "?"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--nu0", type=float, required=True)
    p.add_argument("--G", type=int, required=True)
    p.add_argument("--reps", default="0:1", help="half-open range r0:r1")
    p.add_argument("--laws", default=",".join(config.LAWS))
    p.add_argument("--variants", default="joint", help="joint and/or knownnu")
    p.add_argument("--out", default="results")
    p.add_argument("--theory-only", action="store_true")
    p.add_argument("--bench", action="store_true")
    args = p.parse_args(argv)
    for v in args.variants.split(","):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}")
    for law in args.laws.split(","):
        if law not in noise.LAWS:
            raise SystemExit(f"unknown law {law}")
    (bench if args.bench else run)(args)


if __name__ == "__main__":
    main()
