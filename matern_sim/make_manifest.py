"""Write Slurm array manifests and a submission plan.

  python -m matern_sim.make_manifest bench            # timing jobs, one per (d, G)
  python -m matern_sim.make_manifest theory           # I_n(theta0), kurtosis matrix
  python -m matern_sim.make_manifest fit [--laws ...] [--variants ...] [--nrep 400]
  python -m matern_sim.make_manifest kappa [--name ...]  # residual kappa (eq. A.54)
                                                        # for stored fits lacking it
  python -m matern_sim.make_manifest hess0 [--name ...]  # observed Hessian at theta0

Each stage writes one manifest per (d, G) (lines = run_task arguments) and a plan
manifests/<stage>.plan with one line per array job:
  manifest  job_name  cpus  mem_GB  time_min  n_tasks  [module  [sbatch options]]
(module defaults to matern_sim.run_task; the kappa stage uses matern_sim.kappa_task
and pins each array job to the CPU family of the fits it regenerates).
Fit manifests are sized from the bench results (results/bench/*.json), so run the
bench stage first.  slurm/submit.sh submits a plan.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

from . import config

def evals_per_fit(n, variant):
    """Expected gradient evaluations per fit: L-BFGS-B (paper App. D.3: 20-30 at
    large n, 35-67 at the smallest n; upper ends used here) + 8 for the
    finite-difference Hessian + 1 final evaluation."""
    if variant == "knownnu":
        lbfgs = 25 if n >= 10_000 else 45
    else:
        lbfgs = 35 if n >= 10_000 else 70
    return lbfgs + 9


def bench_cpus_for(n):
    """CPUs used by the benchmark jobs (kept fixed so bench timings stay comparable)."""
    if n <= 3_000:
        return 2
    if n <= 20_000:
        return 8
    return 16


def cpus_for(n):
    """CPUs per fit task.  Measured (pinned, OPENBLAS_THREAD_TIMEOUT=4): 4-core tasks
    do 15-35% more work per core than 8-core ones (d=3, G=32/46; d=4, G=18), and
    Longleaf caps each user at 1000 CPUs, so more, smaller tasks finish the campaign
    sooner."""
    return 1 if n <= 3_000 else 4


def scaled_t_grad(b, cpus):
    """Bench time per gradient, rescaled (linearly, i.e. conservatively) when the
    production task uses fewer threads than the bench job did."""
    bt = int(b.get("threads") or 1)
    return b["t_grad"] * max(1.0, bt / cpus)


def _settings(grids, dims=None, only=None):
    for d, Gs in grids.items():
        if dims and d not in dims:
            continue
        for G in Gs:
            if only and f"d{d}G{G}" not in only:
                continue
            yield d, G


def _range_done(out, d, nu0, G, r0, r1, laws, variants):
    from .run_task import fit_path
    return all(os.path.exists(fit_path(out, d, nu0, G, v, law, r))
               for r in range(r0, r1) for v in variants for law in laws)


def _hm(minutes):
    minutes = int(math.ceil(minutes / 15.0) * 15)
    return minutes


def write_plan(stage, rows):
    os.makedirs("manifests", exist_ok=True)
    plan = os.path.join("manifests", f"{stage}.plan")
    with open(plan, "w") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    tot = sum(r[5] for r in rows)
    print(f"wrote {plan}: {len(rows)} array jobs, {tot} tasks")


# ------------------------------------------------------------- kappa stage
KAPPA_FACTOR = 2.0   # kappa_tilde evaluation / gradient evaluation (measured 1.1-1.2)


def family_sbatch_options(family):
    """sbatch options that place a job on nodes of a CPU family.  Slurm has no
    negated constraints, so the AVX-512 nodes are excluded by name."""
    import subprocess
    if family in ("any", "unknown"):
        return ""
    vendor = family.split("_")[0]
    if family.endswith("_avx512"):
        return f"--constraint={vendor}&avx512"
    out = subprocess.run(["sinfo", "-N", "-h", "-o", "%N %f"], capture_output=True,
                         text=True, check=True).stdout
    excl = sorted({ln.split()[0] for ln in out.splitlines()
                   if vendor in ln.split()[1].split(",") and "avx512" in ln.split()[1]})
    opts = f"--constraint={vendor}" + (f" --exclude={','.join(excl)}" if excl else "")
    if vendor == "intel":
        # Longleaf routes general jobs to hov as well, whose Intel nodes are all
        # AVX-512; a job that one of its partitions cannot run is rejected.
        opts = "--partition=general,spill " + opts
    return opts


def kappa_buffer_gb(d, G):
    """Upper bound on the orbit-row buffer of kappa.ResidualCumulant."""
    me, mo = (G + 1) // 2, G // 2
    norb = math.comb(me + d - 1, d)
    tot = 0
    for k in range(d + 1):
        N = mo**k * me**(d - k)
        tot += min(N, norb * math.comb(d, k)) * N
    return 8.0 * tot / 1024**3


def stage_kappa(args):
    from .fit_hosts import fit_host_table
    from .kappa_task import kappa_path, mismatch_path
    from .run_task import fit_path
    laws = args.laws.split(",")
    variants = args.variants.split(",")
    hosts = fit_host_table()
    hpath = os.path.join(args.out, "kappa", "fit_hosts.csv")
    os.makedirs(os.path.dirname(hpath), exist_ok=True)
    hosts.to_csv(hpath, index=False)
    fam = {(r.d, r.nu0, r.G, r.rep, r.law, r.variant): (r.family, int(r.cpus or 0))
           for r in hosts.itertuples()}
    rows, nmis, nfit, nunk = [], 0, 0, 0
    for d, G in _settings(config.GRIDS, args.dims, args.only):
        n = G**d
        b = _bench(args.out, d, G)
        groups = {}
        for nu0 in config.NU0:
            for rep in range(args.rep_start, args.rep_start + args.nrep):
                for v in variants:
                    for law in laws:
                        if not os.path.exists(fit_path(args.out, d, nu0, G, v, law, rep)):
                            continue
                        nfit += 1
                        if os.path.exists(mismatch_path(args.out, d, nu0, G, v, law, rep)):
                            nmis += 1
                            continue
                        if args.skip_done and os.path.exists(
                                kappa_path(args.out, d, nu0, G, v, law, rep)):
                            continue
                        f, c = fam.get((d, nu0, G, rep, law, v), ("unknown", 0))
                        if f == "unknown" and not args.any_family:
                            nunk += 1        # its log will name the node next round
                            continue
                        groups.setdefault((f, c), {}).setdefault(nu0, {}).setdefault(
                            rep, 0)
                        groups[(f, c)][nu0][rep] += 1
        for (family, cpus), by_nu in sorted(groups.items()):
            ncpu = cpus or cpus_for(n)
            if b is not None:
                tg = scaled_t_grad(b, ncpu)
                setup = b["t_symmetry"] + b["t_eigen"] * max(
                    1.0, int(b.get("threads") or 1) / ncpu) + 60
                maxrss = b["maxrss_gb"]
            else:
                tg, setup, maxrss = 60.0, 600.0, 8.0
            t_fit = KAPPA_FACTOR * tg + 0.5
            lines, tmax = [], 0.0
            for nu0, reps in sorted(by_nu.items()):
                rs = sorted(reps)
                # chunks of consecutive pending reps, each <= target hours of work
                chunk, work = [], 0.0
                for r in rs + [None]:
                    w = 0.0 if r is None else reps[r] * t_fit + 0.3 * tg
                    if chunk and (r is None or setup + work + w > args.target_hours * 3600):
                        lines.append(
                            f"--d {d} --nu0 {nu0:g} --G {G} --reps {chunk[0]}:{chunk[-1] + 1} "
                            f"--laws {','.join(laws)} --variants {','.join(variants)} "
                            f"--family {family} --cpus {cpus} --out {args.out}")
                        tmax = max(tmax, setup + work)
                        chunk, work = [], 0.0
                    if r is not None:
                        chunk.append(r)
                        work += w
            if not lines:
                continue
            path = os.path.join("manifests", f"kappa_{args.name}_d{d}_G{G}_{family}_c{cpus}.tsv")
            with open(path, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            tmin = _hm(min(args.safety * tmax / 60 + 20, 7 * 24 * 60))
            mem = max(4, math.ceil(1.4 * maxrss + kappa_buffer_gb(d, G) + 2))
            opts = family_sbatch_options(family)
            rows.append((path, f"kappa_d{d}G{G}", ncpu, mem, tmin, len(lines),
                         "matern_sim.kappa_task", opts))
            print(f"d={d} G={G:3d} {family:12s} cpus={ncpu}: {len(lines):3d} tasks, "
                  f"{sum(sum(r.values()) for r in by_nu.values()):5d} fits, "
                  f"walltime {tmin / 60:4.1f} h, mem {mem} GB  {opts.split(' ')[0]}")
    print(f"{nfit} fits on disk; {nmis} with a data-mismatch marker "
          f"(results/kappa_mismatch, need attention); {nunk} not yet in the logs")
    write_plan(f"kappa_{args.name}", rows)


def stage_bench(args):
    rows = []
    gs = {d: sorted(set(config.GRIDS.get(d, ())) | set(config.THEORY_GRIDS.get(d, ())))
          for d in config.GRIDS}
    for d, G in _settings(gs, args.dims):
        n = G**d
        path = os.path.join("manifests", f"bench_d{d}_G{G}.tsv")
        with open(path, "w") as fh:
            fh.write(f"--d {d} --nu0 0.5 --G {G} --bench --out {args.out}\n")
        mem = 4 if n < 20_000 else (16 if n < 60_000 else 40)
        rows.append((path, f"bench_d{d}G{G}", bench_cpus_for(n), mem, 60, 1))
    write_plan("bench", rows)


def _bench(out, d, G):
    p = os.path.join(out, "bench", f"d{d}_nu0.5_G{G}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def stage_theory(args):
    rows = []
    for d, G in _settings(config.THEORY_GRIDS, args.dims):
        n = G**d
        b = _bench(args.out, d, G)
        lines = [f"--d {d} --nu0 {nu0:g} --G {G} --theory-only --out {args.out}"
                 for nu0 in config.NU0]
        path = os.path.join("manifests", f"theory_d{d}_G{G}.tsv")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        if b:
            mem = max(4, math.ceil(1.5 * b["maxrss_gb"] + 2))
            tmin = _hm(2 * (b["t_symmetry"] + 4 * b["t_fisher"]) / 60 + 15)
        else:
            mem, tmin = (8 if n < 50_000 else 48), 240
        rows.append((path, f"theory_d{d}G{G}", cpus_for(n), mem, tmin, len(lines)))
    write_plan("theory", rows)


def stage_fit(args):
    laws = args.laws.split(",")
    variants = args.variants.split(",")
    rows = []
    missing = []
    for d, G in _settings(config.GRIDS, args.dims, args.only):
        n = G**d
        b = _bench(args.out, d, G)
        if b is None:
            missing.append((d, G))
            continue
        cpus = cpus_for(n)
        tg = scaled_t_grad(b, cpus)
        tf = b["t_fisher"] * max(1.0, int(b.get("threads") or 1) / cpus)
        t_fit = {v: evals_per_fit(n, v) * tg + tf for v in variants}
        t_rep = len(laws) * sum(t_fit.values())
        setup = b["t_symmetry"] + b["t_eigen"] + 4 * b["t_fisher"] + 60
        per = max(1, int((args.target_hours * 3600 - setup) // t_rep))
        per = min(per, args.nrep)
        lines = []
        for nu0 in config.NU0:
            for r0 in range(args.rep_start, args.rep_start + args.nrep, per):
                r1 = min(r0 + per, args.rep_start + args.nrep)
                if args.skip_done and _range_done(args.out, d, nu0, G, r0, r1, laws, variants):
                    continue
                lines.append(f"--d {d} --nu0 {nu0:g} --G {G} --reps {r0}:{r1} "
                             f"--laws {','.join(laws)} --variants {','.join(variants)} "
                             f"--out {args.out}")
        if not lines:
            print(f"d={d} G={G:3d}: all fits done")
            continue
        path = os.path.join("manifests", f"fit_{args.name}_d{d}_G{G}.tsv")
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        est_min = (setup + per * t_rep) / 60
        tmin = _hm(min(args.safety * est_min + 20, 7 * 24 * 60))
        mem = max(4, math.ceil(1.4 * b["maxrss_gb"] + 2))
        rows.append((path, f"fit_d{d}G{G}", cpus, mem, tmin, len(lines)))
        print(f"d={d} G={G:3d} n={n:7d}: t_grad={tg:7.2f}s  "
              f"t_rep={t_rep / 60:6.1f} min  reps/task={per:4d}  tasks={len(lines):4d}  "
              f"walltime={tmin / 60:5.1f} h  mem={mem} GB  cpus={cpus}  "
              f"core-h={len(lines) * per * t_rep / 3600 * cpus:8.0f}")
    if missing:
        print("no bench results for", missing, "- run the bench stage first")
    write_plan(f"fit_{args.name}", rows)


# ------------------------------------------------------------- hess0 stage
def stage_hess0(args):
    """Observed Hessian at theta0 for stored fits (matern_sim.hess0_task), one
    task per (setting, CPU family, threads) and chunk of <= HESS0_BATCH fits."""
    from .fit_hosts import fit_host_table
    from .hess0_task import hess0_mismatch_path, hess0_path
    from .run_task import fit_path
    laws = args.laws.split(",")
    variants = args.variants.split(",")
    hosts = fit_host_table()
    fam = {(r.d, r.nu0, r.G, r.rep, r.law, r.variant): (r.family, int(r.cpus or 0))
           for r in hosts.itertuples()}
    rows, nfit, nmis = [], 0, 0
    for d, G in _settings(config.GRIDS, args.dims, args.only):
        n = G**d
        b = _bench(args.out, d, G)
        groups = {}
        for nu0 in config.NU0:
            for rep in range(args.rep_start, args.rep_start + args.nrep):
                for v in variants:
                    for law in laws:
                        if not os.path.exists(fit_path(args.out, d, nu0, G, v, law, rep)):
                            continue
                        nfit += 1
                        if os.path.exists(hess0_mismatch_path(args.out, d, nu0, G, v, law,
                                                              rep)):
                            nmis += 1
                            continue
                        if args.skip_done and os.path.exists(
                                hess0_path(args.out, d, nu0, G, v, law, rep)):
                            continue
                        f, c = fam.get((d, nu0, G, rep, law, v), ("unknown", 0))
                        if f == "unknown" and not args.any_family:
                            continue
                        groups.setdefault((f, c), {}).setdefault(nu0, {}).setdefault(rep, 0)
                        groups[(f, c)][nu0][rep] += 1
        batch = 1600 if n <= 3_000 else 800
        for (family, cpus), by_nu in sorted(groups.items()):
            ncpu = cpus or cpus_for(n)
            if b is not None:
                tg = scaled_t_grad(b, ncpu)
                setup = b["t_symmetry"] + b["t_eigen"] * max(
                    1.0, int(b.get("threads") or 1) / ncpu) + 60
                maxrss = b["maxrss_gb"]
            else:
                tg, setup, maxrss = 60.0, 600.0, 8.0
            lines, tmax, fmax = [], 0.0, 0
            for nu0, reps in sorted(by_nu.items()):
                rs = sorted(reps)
                chunk, cnt = [], 0
                for r in rs + [None]:
                    w = 0 if r is None else reps[r]
                    if chunk and (r is None or cnt + w > batch):
                        lines.append(
                            f"--d {d} --nu0 {nu0:g} --G {G} --reps {chunk[0]}:{chunk[-1] + 1} "
                            f"--laws {','.join(laws)} --variants {','.join(variants)} "
                            f"--family {family} --cpus {cpus} --batch {batch} --out {args.out}")
                        # 8 batched gradients (~3 gradient evaluations each) + data
                        tmax = max(tmax, setup + 24 * tg + 0.5 * cnt)
                        fmax = max(fmax, cnt)
                        chunk, cnt = [], 0
                    if r is not None:
                        chunk.append(r)
                        cnt += w
            if not lines:
                continue
            path = os.path.join("manifests", f"hess0_{args.name}_d{d}_G{G}_{family}_c{cpus}.tsv")
            with open(path, "w") as fh:
                fh.write("\n".join(lines) + "\n")
            tmin = _hm(min(args.safety * tmax / 60 + 20, 7 * 24 * 60))
            mem = max(4, math.ceil(1.4 * maxrss + 3 * 8.0 * n * fmax / 1024**3 + 2))
            opts = family_sbatch_options(family)
            rows.append((path, f"hess0_d{d}G{G}", ncpu, mem, tmin, len(lines),
                         "matern_sim.hess0_task", opts))
            print(f"d={d} G={G:3d} {family:12s} cpus={ncpu}: {len(lines):3d} tasks, "
                  f"{sum(sum(r.values()) for r in by_nu.values()):5d} fits, "
                  f"walltime {tmin / 60:4.1f} h, mem {mem} GB  {opts.split(' ')[0]}")
    print(f"{nfit} fits on disk; {nmis} with a data-mismatch marker (results/hess0_mismatch)")
    write_plan(f"hess0_{args.name}", rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["bench", "theory", "fit", "kappa", "hess0"])
    p.add_argument("--out", default="results")
    p.add_argument("--dims", type=lambda s: [int(x) for x in s.split(",")], default=None)
    p.add_argument("--laws", default=",".join(config.LAWS))
    p.add_argument("--variants", default="joint")
    p.add_argument("--nrep", type=int, default=config.NREP)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--target-hours", type=float, default=6.0)
    p.add_argument("--safety", type=float, default=2.0)
    p.add_argument("--name", default="main", help="label for this fit campaign")
    p.add_argument("--only", type=lambda s: set(s.split(",")), default=None,
                   help="restrict to settings, e.g. d3G22,d4G8")
    p.add_argument("--skip-done", action="store_true",
                   help="leave out replicate ranges whose fits all exist (for reruns)")
    p.add_argument("--any-family", action="store_true",
                   help="kappa: also plan fits whose CPU family is unknown (run anywhere)")
    args = p.parse_args(argv)
    {"bench": stage_bench, "theory": stage_theory, "fit": stage_fit,
     "kappa": stage_kappa, "hess0": stage_hess0}[args.stage](args)


if __name__ == "__main__":
    main()
