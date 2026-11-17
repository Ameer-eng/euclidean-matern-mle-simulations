"""Collect fits and theory files into tidy CSV summaries.

  python -m matern_sim.aggregate --out results

Writes results/summary/
  replicates.csv  one row per fit: estimates, errors, standard errors, coverage
  summary.csv     per (d, nu0, G, law, variant, parameter): RMSE, IQR/1.349,
                  bias, benchmark SDs, Wald coverage by method, boundary rate
  joint.csv       joint chi-square coverage per vector and method: nfit, n_valid
                  (finite, symmetric, positive-definite covariance; see joint_wald),
                  n_cover_valid, coverage = coverage_given_valid = n_cover_valid/n_valid,
                  prob_valid_and_cover = n_cover_valid/nfit, invalid counts by reason
  nugget.csv      nugget checks: sqrt(n) SD(log tau2) vs sqrt(E xi^4 - 1),
                  nugget-signal correlations, distance to the oracle fit, kappa_hat
  theory.csv      benchmark SDs from I_n(theta0) and the kurtosis matrix for every
                  theory grid (incl. n beyond the Monte Carlo range)

Notation: theta = (eta, nu, log a, log t).  Feasible covariance (Cor. 3.5):
  Sigma_hat = [A_hat^{-1}]_{zeta zeta} + kappa_hat / (n t_hat^2) e_t e_t^T
with A_hat = expected information or observed negative Hessian at theta_hat.
Covariance methods <curvature>_<inversion>_<kappa>.  curvature = fisher, hess
(at theta_hat; delta method and t at theta_hat) or fisher0, hess0 (at theta0:
I_n(theta0) from results/theory, -d2 ell(theta0) from results/hess0, see
hess0_task.py; delta method and t at theta0).  inversion (d <= 3) = full
(identifiable block of the inverse of the 4x4 matrix) or block (inverse of the
identifiable 3x3 block).  kappa = khat (residual
estimator kappa_tilde of eq. A.54, from results/kappa/, see kappa_task.py), none
(uncorrected), ktrue (true kappa4) and kpair (the former adjacent-pair estimator,
kept as a diagnostic).  A fit without a kappa_tilde file gets no khat coverage for
log tau2 (counted in n_khat_missing).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import digamma
from scipy.stats import chi2

from . import noise
from .kernel import log_A

PARAMS = ["log_nu", "log_c", "log_sigma2", "log_ell", "log_tau2"]
IDENT = [0, 1, 3]  # identifiable spectral coordinates for d <= 3 (eta, nu, log t)
Z975 = 1.959963984540054


def gvals(theta, d):
    """Parameters on the log scale, columns ordered as PARAMS (theta: (m, 4))."""
    eta, nu, la, lt = theta.T
    lA = np.array([log_A(d, v) for v in nu])
    return np.stack([np.log(nu), eta - lA, eta - lA - nu * la, -0.5 * la, lt], axis=1)


def gjac(theta, d):
    """Jacobians d g / d theta, shape (m, 5, 4)."""
    eta, nu, la, lt = theta.T
    B = digamma(nu + 0.5 * d) - digamma(nu)
    m = len(nu)
    J = np.zeros((m, 5, 4))
    J[:, 0, 1] = 1.0 / nu
    J[:, 1, 0] = 1.0
    J[:, 1, 1] = -B
    J[:, 2, 0] = 1.0
    J[:, 2, 1] = -B - la
    J[:, 2, 2] = -nu
    J[:, 3, 2] = -0.5
    J[:, 4, 3] = 1.0
    return J


def free_coords(variant):
    return [0, 2, 3] if variant == "knownnu" else [0, 1, 2, 3]


def params_for(d, variant):
    ps = ["log_nu", "log_c", "log_tau2"] if d <= 3 else ["log_nu", "log_sigma2", "log_ell", "log_tau2"]
    if variant == "knownnu":
        ps = [p for p in ps if p != "log_nu"]
    return ps


def sandwich_cov(I, Dk, kappa4, coords):
    """First-order covariance of theta_hat in `coords`: I^{-1} (I + k/4 Dk) I^{-1}."""
    Ic = I[np.ix_(coords, coords)]
    Kc = Ic + 0.25 * kappa4 * Dk[np.ix_(coords, coords)]
    Ii = np.linalg.inv(Ic)
    return Ii @ Kc @ Ii


def theory_sds(th, law, variant):
    """Benchmark SDs of each parameter at theta0: 'gauss_*' uses I^{-1};
    'law_*' uses the exact finite-n sandwich for the error law. Versions:
    full (all estimated coordinates) and, for d <= 3, block (range at a0)."""
    d = int(th["d"])
    theta0 = th["theta0"][None, :]
    J = gjac(theta0, d)[0]
    k4 = noise.kappa4(law, float(th["tau2"]))
    free = free_coords(variant)
    versions = {"full": free}
    if d <= 3:
        versions["block"] = [c for c in free if c in IDENT]
    out = {}
    for vname, coords in versions.items():
        for lname, kk in (("gauss", 0.0), ("law", k4)):
            V = sandwich_cov(th["I0"], th["Dk0"], kk, coords)
            Jc = J[:, coords]
            sd = np.sqrt(np.einsum("pi,ij,pj->p", Jc, V, Jc))
            for p, s in zip(PARAMS, sd):
                out[(p, f"sd_{lname}_{vname}")] = s
    return out


# ------------------------------------------------------------------ loading
def load_kappa(fit_file):
    """kappa_tilde for a fit (NaN if results/kappa/... does not exist yet)."""
    kf = fit_file.replace(os.sep + "fits" + os.sep, os.sep + "kappa" + os.sep)
    if not os.path.exists(kf):
        return math.nan
    with np.load(kf) as k:
        return float(k["kappa_resid"])


def load_hess0(fit_file):
    """-d2 ell(theta0) for a fit (NaN matrix if results/hess0/... is missing)."""
    hf = fit_file.replace(os.sep + "fits" + os.sep, os.sep + "hess0" + os.sep)
    if not os.path.exists(hf):
        return np.full((4, 4), np.nan)
    with np.load(hf) as h:
        return np.array(h["negH0"], float)


def load_fits(out):
    rows = []
    for f in glob.glob(os.path.join(out, "fits", "*", "*", "*", "rep*.npz")):
        z = np.load(f)
        rows.append(dict(kappa_resid=load_kappa(f), negH0=load_hess0(f),
            d=int(z["d"]), nu0=float(z["nu0"]), G=int(z["G"]), n=int(z["n"]),
            law=str(z["law"]), variant=str(z["variant"]), rep=int(z["rep"]),
            theta_hat=z["theta_hat"], theta0=z["theta0"], grad=z["grad"],
            negH=z["negH"], fisher_hat=z["fisher_hat"], success=bool(z["success"]),
            nfev=int(z["nfev"]), at_bound=z["at_bound"], kappa_pair=float(z["kappa_pair"]),
            t_pair=float(z["t_pair"]), logt_oracle=float(z["logt_oracle"]),
            kappa4_true=float(z["kappa4_true"]), m4_true=float(z["m4_true"]),
            seconds=float(z["seconds_total"])))
    return rows


def load_theory(out):
    th = {}
    for f in glob.glob(os.path.join(out, "theory", "*.npz")):
        z = dict(np.load(f))
        th[(int(z["d"]), float(z["nu0"]), int(z["G"]))] = z
    return th


# ------------------------------------------------------------- per group
AT0 = ("fisher0", "hess0")      # curvature sources evaluated at theta0


def feasible_covs(grp, d, variant, th=None):
    """Dict method -> (m, 4, 4) covariance of theta_hat (zeros outside coords)."""
    free = free_coords(variant)
    m = len(grp)
    n = grp[0]["n"]
    t_hat = np.array([math.exp(r["theta_hat"][3]) for r in grp])
    t_0 = np.full(m, math.exp(grp[0]["theta0"][3]))
    k_hat = np.array([r["kappa_resid"] for r in grp])
    k_pair = np.array([r["kappa_pair"] for r in grp])
    k_true = np.array([r["kappa4_true"] for r in grp])
    out = {}
    sources = [("fisher", np.stack([r["fisher_hat"] for r in grp])),
               ("hess", np.stack([r["negH"] for r in grp])),
               ("hess0", np.stack([r["negH0"] for r in grp]))]
    if th is not None:
        sources.append(("fisher0", np.repeat(np.asarray(th["I0"], float)[None], m, 0)))
    for src, A in sources:
        tt = t_0 if src in AT0 else t_hat
        modes = {"full": None}
        if d <= 3:
            modes["block"] = [c for c in free if c in IDENT]
        for mode, sub in modes.items():
            V = np.zeros((m, 4, 4))   # coordinates not estimated / not selected: 0
            for i in range(m):
                Af = A[i][np.ix_(free, free)]
                if not np.all(np.isfinite(A[i])):
                    V[i] = np.nan
                    continue
                try:
                    if sub is None:
                        Vi = np.linalg.inv(Af)
                        V[i][np.ix_(free, free)] = Vi
                    else:
                        loc = [free.index(c) for c in sub]
                        V[i][np.ix_(sub, sub)] = np.linalg.inv(Af[np.ix_(loc, loc)])
                except np.linalg.LinAlgError:
                    V[i] = np.nan
            for kname, kk in (("khat", k_hat), ("none", 0.0 * k_true), ("ktrue", k_true),
                              ("kpair", k_pair)):
                if kname in ("ktrue", "kpair") and not (src == "fisher" and mode == "full"):
                    continue
                Vk = V.copy()
                ok = np.isfinite(kk)
                Vk[:, 3, 3] += np.where(ok, kk, 0.0) / (n * tt**2)
                Vk[~ok, 3, 3] = np.nan       # only the nugget variance is unknown
                out[f"{src}_{mode}_{kname}"] = Vk
    return out


def quad_se(Jrow, V):
    """sqrt(J V J^T) per replicate.  A marginal SE is valid when its own corrected
    variance J V J^T is finite and strictly positive (other entries of V may be
    indefinite); otherwise NaN.  NaN also where a coordinate that J uses is NaN."""
    used = np.abs(Jrow) > 0                                  # (m, 4)
    Vz = np.where(np.isnan(V), 0.0, V)
    var = np.einsum("mi,mij,mj->m", Jrow, Vz, Jrow)
    bad = np.any(np.isnan(np.diagonal(V, axis1=1, axis2=2)) & used, axis=1)
    bad |= ~(np.isfinite(var) & (var > 0))
    return np.where(bad, np.nan, np.sqrt(np.where(bad, 1.0, var)))


# Joint Wald regions (FOLLOW-UP Task 2, 2026-11-10).  A joint region is valid only
# when its covariance is finite, symmetric up to floating-point rounding, and
# positive definite; positive definiteness is decided by a Cholesky factorisation
# and the statistic is computed from the Cholesky factor.  An invalid covariance is
# never repaired (no eigenvalue clipping, no jitter) and never counts as covering.
# Before this change the statistic was e' solve(S, e) for every finite S, so an
# indefinite S could give a negative statistic that was counted as covered.
SYM_RTOL = 1e-8      # max |S - S'| / max |S| tolerated before symmetrising


def joint_wald(e, S):
    """(status, stat, rel_asym) for the region {e' S^{-1} e <= crit}.
    status: 'ok', 'nonfinite', 'asymmetric' or 'not_pd'; stat is NaN unless 'ok'."""
    e = np.asarray(e, float)
    S = np.asarray(S, float)
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(e))):
        return "nonfinite", math.nan, math.nan
    scale = float(np.max(np.abs(S)))
    asym = float(np.max(np.abs(S - S.T))) / scale if scale > 0 else 0.0
    if asym > SYM_RTOL:
        return "asymmetric", math.nan, asym
    try:
        L = np.linalg.cholesky(0.5 * (S + S.T))
    except np.linalg.LinAlgError:
        return "not_pd", math.nan, asym
    z = solve_triangular(L, e, lower=True)
    return "ok", float(z @ z), asym


def summarise_group(grp, th):
    d, nu0, G, n = grp[0]["d"], grp[0]["nu0"], grp[0]["G"], grp[0]["n"]
    law, variant = grp[0]["law"], grp[0]["variant"]
    theta_hat = np.stack([r["theta_hat"] for r in grp])
    theta0 = grp[0]["theta0"]
    g_hat = gvals(theta_hat, d)
    g0 = gvals(theta0[None, :], d)[0]
    err = g_hat - g0
    J = gjac(theta_hat, d)
    J0 = np.repeat(gjac(theta0[None, :], d), len(grp), axis=0)
    covs = feasible_covs(grp, d, variant, th)
    params = params_for(d, variant)

    rep_rows = pd.DataFrame(dict(d=d, nu0=nu0, G=G, n=n, law=law, variant=variant,
                                 rep=[r["rep"] for r in grp],
                                 success=[r["success"] for r in grp],
                                 nfev=[r["nfev"] for r in grp],
                                 any_bound=[bool(np.any(r["at_bound"])) for r in grp],
                                 kappa_hat=[r["kappa_resid"] for r in grp],
                                 kappa_pair=[r["kappa_pair"] for r in grp]))
    for j, p in enumerate(PARAMS):
        rep_rows[f"{p}_hat"] = g_hat[:, j]
        rep_rows[f"{p}_err"] = err[:, j]

    sds = theory_sds(th, law, variant) if th is not None else {}
    summ = []
    rng = np.random.default_rng(0)
    for p in params:
        j = PARAMS.index(p)
        e = err[:, j]
        boot = rng.integers(0, len(e), size=(1000, len(e)))
        rmse_b = np.sqrt(np.mean(e[boot] ** 2, axis=1))
        q75, q25 = np.percentile(e, [75, 25])
        row = dict(d=d, nu0=nu0, G=G, n=n, law=law, variant=variant, param=p,
                   nfit=len(e), rmse=float(np.sqrt(np.mean(e**2))),
                   rmse_lo=float(np.quantile(rmse_b, 0.025)),
                   rmse_hi=float(np.quantile(rmse_b, 0.975)),
                   iqr_sd=float((q75 - q25) / 1.349), bias=float(np.mean(e)),
                   median_err=float(np.median(e)),
                   bound_rate=float(rep_rows["any_bound"].mean()),
                   fail_rate=float(1 - rep_rows["success"].mean()),
                   n_khat_missing=int(rep_rows["kappa_hat"].isna().sum()),
                   n_hess0_missing=int(sum(not np.all(np.isfinite(r["negH0"]))
                                           for r in grp)))
        for (pp, k), v in sds.items():
            if pp == p:
                row[k] = float(v)
        for meth, V in covs.items():
            Jm = J0 if meth.split("_")[0] in AT0 else J
            se = quad_se(Jm[:, j], V)
            cover = np.where(np.isnan(se), np.nan, np.abs(e) <= Z975 * se)
            row[f"cov_{meth}"] = float(np.nanmean(cover)) if np.any(np.isfinite(cover)) \
                else math.nan
            row[f"se_ratio_{meth}"] = float(np.nanmedian(se) / row["rmse"])
            # accuracy of the SE as an estimate of the sampling SD: RMS of the
            # standardized errors (1 if accurate) and the spread of the SEs
            ok = np.isfinite(se) & (se > 0)
            row[f"rmsz_{meth}"] = float(np.sqrt(np.mean((e[ok] / se[ok]) ** 2))) \
                if ok.any() else math.nan
            row[f"madz_{meth}"] = float(np.median(np.abs(e[ok] / se[ok])) / 0.6744897501960817) \
                if ok.any() else math.nan
            row[f"se_rms_ratio_{meth}"] = float(np.sqrt(np.mean(se[ok] ** 2)) / row["rmse"]) \
                if ok.any() else math.nan
            row[f"se_cv_{meth}"] = float(np.std(se[ok]) / np.mean(se[ok])) \
                if ok.sum() > 1 else math.nan
            row[f"n_se_{meth}"] = int(ok.sum())
            rep_rows[f"{p}_se_{meth}"] = se
        summ.append(row)

    # joint chi-square: log identifiable vector, and (d >= 4) raw public scale
    joint = []
    vecs = {"log_vector": params}
    if d >= 4:
        vecs["raw_public"] = params
    for vname, ps in vecs.items():
        idx = [PARAMS.index(p) for p in ps]
        ev = err[:, idx]
        if vname == "raw_public":
            ev = np.exp(g_hat[:, idx]) - np.exp(g0[idx])
        crit = chi2.ppf(0.95, len(idx))
        for meth, V in covs.items():
            at0 = meth.split("_")[0] in AT0
            Jv = (J0 if at0 else J)[:, idx, :]
            if vname == "raw_public":  # (sigma2, ell, nu, tau2): d exp(g) = exp(g) dg
                sc = np.exp(np.repeat(g0[None, idx], len(ev), 0) if at0 else g_hat[:, idx])
                Jv = Jv * sc[:, :, None]
            # NaN entries of V mark an unavailable covariance (failed inversion, or
            # missing kappa_tilde for the nugget); they propagate into S -> invalid.
            S = np.einsum("mpi,mij,mqj->mpq", Jv, V, Jv)
            stat = np.full(len(ev), np.nan)
            status = []
            asym = []
            for i in range(len(ev)):
                st, q, a_ = joint_wald(ev[i], S[i])
                status.append(st)
                stat[i] = q
                if np.isfinite(a_):
                    asym.append(a_)
            status = np.array(status)
            valid = status == "ok"
            nv = int(valid.sum())
            ncov = int(np.sum(valid & (np.where(valid, stat, np.inf) <= crit)))
            m_ = len(ev)
            joint.append(dict(d=d, nu0=nu0, G=G, n=n, law=law, variant=variant,
                              vector=vname, k=len(idx), method=meth,
                              # coverage = coverage conditional on a valid region
                              coverage=ncov / nv if nv else math.nan,
                              mean_stat=float(np.mean(stat[valid])) if nv else math.nan,
                              nfit=m_, n_valid=nv, n_cover_valid=ncov,
                              coverage_given_valid=ncov / nv if nv else math.nan,
                              prob_valid_and_cover=ncov / m_ if m_ else math.nan,
                              n_invalid_nonfinite=int(np.sum(status == "nonfinite")),
                              n_invalid_asymmetric=int(np.sum(status == "asymmetric")),
                              n_invalid_not_pd=int(np.sum(status == "not_pd")),
                              max_rel_asym=float(max(asym)) if asym else math.nan))

    # nugget checks (Theorem 3.1, eq. 3.19 and the oracle comparison)
    nug = {}
    if variant == "joint":
        e_t = err[:, PARAMS.index("log_tau2")]
        u_err = (theta_hat[:, 0] - theta0[0]) - 2 * (math.log(n) / (2 * nu0 + d)) * (
            theta_hat[:, 1] - theta0[1])
        lt_or = np.array([r["logt_oracle"] for r in grp])
        m4 = grp[0]["m4_true"]
        nug = dict(d=d, nu0=nu0, G=G, n=n, law=law, nfit=len(e_t), m4=m4,
                   sqrt_n_sd_logtau2=float(math.sqrt(n) * np.std(e_t, ddof=1)),
                   sqrt_n_rmse_logtau2=float(math.sqrt(n) * np.sqrt(np.mean(e_t**2))),
                   limit_sd=math.sqrt(m4 - 1.0),
                   sqrt_n_sd_oracle=float(math.sqrt(n) * np.std(lt_or - theta0[3], ddof=1)),
                   sqrt_n_rmse_joint_minus_oracle=float(
                       math.sqrt(n) * np.sqrt(np.mean((theta_hat[:, 3] - lt_or) ** 2))),
                   corr_tau_u=float(np.corrcoef(e_t, u_err)[0, 1]),
                   corr_tau_nu=float(np.corrcoef(e_t, theta_hat[:, 1])[0, 1]),
                   corr_tau_loga=float(np.corrcoef(e_t, theta_hat[:, 2])[0, 1]),
                   kappa_true=grp[0]["kappa4_true"],
                   kappa_hat_mean=float(rep_rows["kappa_hat"].mean()),
                   kappa_hat_sd=float(rep_rows["kappa_hat"].std(ddof=1)),
                   kappa_hat_n=int(rep_rows["kappa_hat"].notna().sum()),
                   kappa_pair_mean=float(rep_rows["kappa_pair"].mean()),
                   kappa_pair_sd=float(rep_rows["kappa_pair"].std(ddof=1)))
        if "sd_law_full" in {k for (_, k) in sds}:
            nug["sqrt_n_sd_theory_full"] = math.sqrt(n) * sds[("log_tau2", "sd_law_full")]
            if d <= 3:
                nug["sqrt_n_sd_theory_block"] = math.sqrt(n) * sds[("log_tau2", "sd_law_block")]
    return rep_rows, summ, joint, nug


def theory_table(th_all):
    rows = []
    for (d, nu0, G), th in sorted(th_all.items()):
        for law in noise.LAWS:
            for variant in ("joint", "knownnu"):
                sds = theory_sds(th, law, variant)
                for p in params_for(d, variant):
                    row = dict(d=d, nu0=nu0, G=G, n=int(th["n"]), law=law,
                               variant=variant, param=p)
                    row.update({k: float(v) for (pp, k), v in sds.items() if pp == p})
                    rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    np.seterr(invalid="ignore", divide="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN groups
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    sdir = os.path.join(args.out, "summary")
    os.makedirs(sdir, exist_ok=True)
    th_all = load_theory(args.out)
    theory_table(th_all).to_csv(os.path.join(sdir, "theory.csv"), index=False)
    fits = load_fits(args.out)
    print(f"{len(fits)} fits, {len(th_all)} theory files")
    groups = {}
    for r in fits:
        groups.setdefault((r["d"], r["nu0"], r["G"], r["law"], r["variant"]), []).append(r)
    reps, summ, joint, nug = [], [], [], []
    for key in sorted(groups):
        grp = sorted(groups[key], key=lambda r: r["rep"])
        a, b, c, e = summarise_group(grp, th_all.get(key[:3]))
        reps.append(a)
        summ += b
        joint += c
        if e:
            nug.append(e)
    if reps:
        pd.concat(reps).to_csv(os.path.join(sdir, "replicates.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(sdir, "summary.csv"), index=False)
    pd.DataFrame(joint).to_csv(os.path.join(sdir, "joint.csv"), index=False)
    pd.DataFrame(nug).to_csv(os.path.join(sdir, "nugget.csv"), index=False)
    print(f"wrote {sdir}/{{replicates,summary,joint,nugget,theory}}.csv")


if __name__ == "__main__":
    main()
