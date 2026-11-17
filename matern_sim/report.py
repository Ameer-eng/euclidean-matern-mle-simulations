"""Numbers quoted in Section 4 and Appendix D of the paper.

  python -m matern_sim.report --out results > results/summary/report.txt

Reads results/summary/*.csv (matern_sim.aggregate) and prints, section by
section, every Monte Carlo number used in the text and tables.
"""
from __future__ import annotations

import argparse
import math
import os
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import kstest, norm, skew

from . import noise

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 400)

CURV = ["fisher", "hess", "fisher0", "hess0"]
TAU2_LO = math.log(1e-3)
NU_HI = math.log(4.0)


SETTINGS = [(3, 0.5), (3, 1.5), (4, 0.5), (4, 1.5), (5, 0.5), (5, 1.5)]
QQ_TARGET = (1e3, 1e4, 1e5)


def params_for(d):
    return ["log_nu", "log_c", "log_tau2"] if d <= 3 else \
        ["log_nu", "log_sigma2", "log_ell", "log_tau2"]


# ------------------------------------------------------------------ rates
def rate(d, nu0, p, n):
    """Theoretical rate of Theorem 3.1 at n, its label, and the limiting slope.
    For sigma2 and l in d = 4 the slope is in log log n."""
    n = np.asarray(n, float)
    den = int(round(4 * nu0 + 2 * d))
    a = Fraction(d, den)
    if p == "log_nu":
        return n ** (-float(a)), rf"$n^{{-{a}}}$", -float(a)
    if p == "log_c":
        return np.log(n) * n ** (-float(a)), rf"$(\log n)\,n^{{-{a}}}$", -float(a)
    if p == "log_tau2":
        return n ** -0.5, r"$n^{-1/2}$", -0.5
    if d == 4:
        return np.log(n) ** -0.5, r"$(\log n)^{-1/2}$", -0.5
    b = Fraction(d - 4, den)
    return n ** (-float(b)), rf"$n^{{-{b}}}$", -float(b)


def slope_x(d, p, n):
    n = np.asarray(n, float)
    return np.log(np.log(n)) if (d == 4 and p in ("log_sigma2", "log_ell")) else np.log(n)


def ls_slope(x, y):
    ok = np.isfinite(y) & (y > 0)
    return float(np.polyfit(x[ok], np.log(y[ok]), 1)[0])


def qq_ns(ns):
    ns = np.asarray(sorted(ns))
    return [int(ns[np.argmin(np.abs(np.log(ns) - math.log(t)))]) for t in QQ_TARGET]


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


def methods(d):
    return [f"{c}_{i}" for i in (("full", "block") if d <= 3 else ("full",)) for c in CURV]


class R:
    def __init__(self, out):
        sd = os.path.join(out, "summary")
        self.rep = pd.read_csv(os.path.join(sd, "replicates.csv"))
        self.s = pd.read_csv(os.path.join(sd, "summary.csv"))
        self.j = pd.read_csv(os.path.join(sd, "joint.csv"))
        self.nug = pd.read_csv(os.path.join(sd, "nugget.csv"))
        self.th = pd.read_csv(os.path.join(sd, "theory.csv"))
        self.th = self.th[self.th.variant == "joint"]

    def S(self, d, nu0, law="gauss", p=None):
        s = self.s[(self.s.d == d) & (self.s.nu0 == nu0) & (self.s.law == law)]
        return (s if p is None else s[s.param == p]).sort_values("n")

    def reps(self, d, nu0, law="gauss", n=None):
        r = self.rep[(self.rep.d == d) & (self.rep.nu0 == nu0) & (self.rep.law == law)]
        return r if n is None else r[r.n == n]

    def nmax(self, d):
        return int(self.s[self.s.d == d].n.max())

    def sd(self, d, nu0, p, n, col="sd_gauss_full", law="gauss"):
        t = self.th
        return float(t[(t.d == d) & (t.nu0 == nu0) & (t.param == p) & (t.n == n) &
                       (t.law == law)][col].iloc[0])


def boundary(r):
    hdr("Boundary solutions (% of fits with a coordinate on the boundary of Theta)")
    b = r.s[r.s.param == "log_nu"].pivot_table(index=["d", "nu0", "n"], columns="law",
                                               values="bound_rate") * 100
    print(b.round(2).to_string())
    for law in b.columns:
        bb = b[law]
        print(law, "largest n with boundary>0 per (d,nu0):",
              {k[:2]: k[2] for k in bb[bb > 0].index})
    rep = r.rep
    lo = rep.log_tau2_hat <= TAU2_LO + 1e-6
    hi = rep.log_nu_hat >= NU_HI - 1e-6
    t = rep.assign(tau_lo=lo, nu_hi=hi, other=rep.any_bound & ~lo & ~hi)
    g = t.groupby(["d", "nu0", "n"])[["any_bound", "tau_lo", "nu_hi", "other"]].sum()
    print("counts over all laws (1600 fits per row):")
    print(g[g.any_bound > 0].to_string())


def slopes(r):
    hdr("Table 3: least-squares slopes (Gaussian errors, all simulated n)")
    rows = []
    for d, nu0 in SETTINGS:
        for p in params_for(d):
            s = r.S(d, nu0, p=p)
            ns = s.n.to_numpy()
            x = slope_x(d, p, ns)
            fit = lambda y: float(np.polyfit(x, np.log(np.asarray(y, float)), 1)[0])
            sdb = [r.sd(d, nu0, p, n, "sd_gauss_block" if d <= 3 else "sd_gauss_full")
                   for n in ns]
            sdf = [r.sd(d, nu0, p, n) for n in ns]
            rows.append(dict(d=d, nu0=nu0, p=p, rmse=fit(s.rmse), iqr=fit(s.iqr_sd),
                             asy=fit(sdb), asy_full=fit(sdf) if d <= 3 else np.nan,
                             theory=rate(d, nu0, p, ns)[2]))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    hdr("Local slopes of the asymptotic SD of log nu and log tau2 at the end of the theory grid")
    for d, nu0 in SETTINGS:
        for p in ("log_nu", "log_tau2"):
            t = r.th[(r.th.d == d) & (r.th.nu0 == nu0) & (r.th.param == p) &
                     (r.th.law == "gauss")].sort_values("n")
            col = "sd_gauss_block" if d <= 3 else "sd_gauss_full"
            x, y = np.log(t.n.to_numpy()[-2:]), np.log(t[col].to_numpy()[-2:])
            print(d, nu0, p, "n", t.n.to_numpy()[-2:], "local slope", round((y[1] - y[0]) / (x[1] - x[0]), 3))


def benchmark(r):
    hdr("RMSE / asymptotic SD (full information) by n, Gaussian errors")
    rows = []
    for d, nu0 in SETTINGS:
        for p in params_for(d):
            s = r.S(d, nu0, p=p)
            for _, x in s.iterrows():
                rows.append(dict(d=d, nu0=nu0, p=p, n=int(x.n),
                                 rmse_over_full=x.rmse / x.sd_gauss_full,
                                 rmse_over_block=x.rmse / x.sd_gauss_block if d <= 3 else np.nan,
                                 iqr_over_full=x.iqr_sd / x.sd_gauss_full))
    t = pd.DataFrame(rows)
    print(t.pivot_table(index=["d", "nu0", "n"], columns="p", values="rmse_over_full").round(3).to_string())
    print("\nIQR/1.349 over full SD:")
    print(t.pivot_table(index=["d", "nu0", "n"], columns="p", values="iqr_over_full").round(3).to_string())
    print("\nd=3 RMSE over block SD:")
    print(t[t.d == 3].pivot_table(index=["nu0", "n"], columns="p", values="rmse_over_block").round(3).to_string())
    for d in (4, 5):
        for nu0 in (0.5, 1.5):
            tt = t[(t.d == d) & (t.nu0 == nu0)]
            for nmin in sorted(tt.n.unique()):
                u = tt[tt.n >= nmin]
                dev = np.abs(u.rmse_over_full - 1).max()
                print(f"d={d} nu0={nu0} n>={nmin}: max|RMSE/SD-1| = {dev:.3f}")
            for nmin in sorted(tt.n.unique()):
                u = tt[(tt.n >= nmin) & (tt.p != "log_tau2")]
                print(f"   signal only n>={nmin}: {np.abs(u.rmse_over_full - 1).max():.3f}")


def identifiability(r):
    hdr("Identifiability: d=3 robust SD of log sigma2, log l; RMSE decrease factors")
    for nu0 in (0.5, 1.5):
        rr = r.reps(3, nu0)
        for n in sorted(rr.n.unique()):
            x = rr[rr.n == n]
            q = lambda v: (np.percentile(v, 75) - np.percentile(v, 25)) / 1.349
            print(f"d=3 nu0={nu0} n={n}: IQR-sd log s2={q(x.log_sigma2_hat):.3f} log l={q(x.log_ell_hat):.3f}"
                  f"  median log s2 err={np.median(x.log_sigma2_err):.3f} log l err={np.median(x.log_ell_err):.3f}")
    for d in (4, 5):
        for nu0 in (0.5, 1.5):
            out = []
            for p in params_for(d):
                s = r.S(d, nu0, p=p)
                ns = s.n.to_numpy()
                n1 = ns[np.argmin(np.abs(np.log(ns) - math.log(1e3)))]
                n2 = ns.max()
                f = float(s.loc[s.n == n1, "rmse"].iloc[0] / s.loc[s.n == n2, "rmse"].iloc[0])
                fs = r.sd(d, nu0, p, n1) / r.sd(d, nu0, p, n2)
                out.append(f"{p}: {f:.2f} (asy {fs:.2f})")
            print(f"d={d} nu0={nu0} RMSE factor n~1e3 -> nmax:", "; ".join(out))


def smoothness(r):
    hdr("Smoothness and amplitude at the largest n (Gaussian)")
    for d, nu0 in SETTINGS:
        n = r.nmax(d)
        s = r.S(d, nu0)
        s = s[s.n == n].set_index("param")
        msg = f"d={d} nu0={nu0} n={n}: RMSE log nu={s.loc['log_nu','rmse']:.4f}"
        if d == 3:
            msg += f", RMSE log c={s.loc['log_c','rmse']:.4f}, ratio={s.loc['log_c','rmse']/s.loc['log_nu','rmse']:.2f}"
            msg += f" (asy full ratio {s.loc['log_c','sd_gauss_full']/s.loc['log_nu','sd_gauss_full']:.2f})"
        print(msg)


def nugget(r):
    hdr("Nugget: sqrt(n) RMSE(log tau2) vs theory, by law, at all n")
    t = r.nug.copy()
    t["ratio_theory_limit"] = t.sqrt_n_sd_theory_full / t.limit_sd
    cols = ["d", "nu0", "n", "law", "sqrt_n_rmse_logtau2", "sqrt_n_sd_theory_full", "limit_sd",
            "ratio_theory_limit", "sqrt_n_rmse_joint_minus_oracle", "sqrt_n_sd_oracle",
            "corr_tau_u", "corr_tau_nu", "corr_tau_loga"]
    print(t[cols].sort_values(["d", "nu0", "law", "n"]).round(3).to_string(index=False))
    hdr("Ratio of sqrt(n) RMSE(log tau2) to the law-specific asymptotic SD (all laws)")
    t["mc_over_theory"] = t.sqrt_n_rmse_logtau2 / t.sqrt_n_sd_theory_full
    print(t.pivot_table(index=["d", "nu0", "n"], columns="law", values="mc_over_theory").round(3).to_string())
    hdr("Mixture vs Bernoulli: ratio of sqrt(n) RMSE(log tau2)")
    pv = t.pivot_table(index=["d", "nu0", "n"], columns="law", values="sqrt_n_rmse_logtau2")
    print((pv.mixnorm / pv.bern).round(3).to_string())
    print("mixture / uniform:")
    print((pv.mixnorm / pv.unif).round(3).to_string())


def laws_signal(r):
    hdr("Signal RMSE under non-Gaussian laws relative to Gaussian (same fields)")
    rows = []
    for d, nu0 in SETTINGS:
        for p in params_for(d):
            if p == "log_tau2":
                continue
            g = r.S(d, nu0, "gauss", p).set_index("n").rmse
            for law in ("unif", "mixnorm", "bern"):
                l = r.S(d, nu0, law, p).set_index("n").rmse
                for n in g.index:
                    rows.append(dict(d=d, nu0=nu0, p=p, law=law, n=n, ratio=l[n] / g[n]))
    t = pd.DataFrame(rows)
    t["dev"] = (t.ratio - 1).abs()
    for nmin in (0, 1000, 2500, 10000, 25000):
        u = t[t.n >= nmin]
        k = u.dev.idxmax()
        print(f"n>={nmin}: max |ratio-1| = {u.dev.max():.3f} at", t.loc[k, ["d", "nu0", "p", "law", "n"]].to_dict())
    big = t[t.n == t.groupby("d").n.transform("max")]
    print("at the largest n: max |ratio-1| =", round(big.dev.max(), 4))
    print(big.pivot_table(index=["d", "nu0", "p"], columns="law", values="ratio").round(3).to_string())
    print("\nper n, max over params and laws:")
    print(t.groupby(["d", "nu0", "n"]).dev.max().round(3).to_string())


def coverage(r):
    hdr("Table 4: coverage at the largest n")
    rows = []
    for d, nu0 in SETTINGS:
        n = r.nmax(d)
        for law in noise.LAWS:
            s = r.S(d, nu0, law)
            s = s[s.n == n].set_index("param")
            sig = [p for p in params_for(d) if p != "log_tau2"]
            cs = s.loc[sig, "cov_fisher_full_khat"] * 100
            ch = s.loc[sig, "cov_hess_full_khat"] * 100
            sG = r.sd(d, nu0, "log_tau2", n, "sd_gauss_full", law)
            sP = r.sd(d, nu0, "log_tau2", n, "sd_law_full", law)
            rows.append(dict(d=d, nu0=nu0, law=law, signal=f"{cs.min():.1f}-{cs.max():.1f}",
                             signal_hess=f"{ch.min():.1f}-{ch.max():.1f}",
                             theory=100 * (2 * norm.cdf(1.96 * sG / sP) - 1),
                             uncorr=100 * s.loc["log_tau2", "cov_fisher_full_none"],
                             fisher=100 * s.loc["log_tau2", "cov_fisher_full_khat"],
                             hess=100 * s.loc["log_tau2", "cov_hess_full_khat"],
                             ktrue=100 * s.loc["log_tau2", "cov_fisher_full_ktrue"],
                             n_khat=int(s.loc["log_tau2", "n_se_fisher_full_khat"])))
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    hdr("Coverage by n (Fisher full khat at theta_hat): signal (min over params) and nugget")
    t = r.s[r.s.param != "log_tau2"].groupby(["d", "nu0", "n", "law"]).cov_fisher_full_khat.min()
    u = r.s[r.s.param == "log_tau2"].set_index(["d", "nu0", "n", "law"]).cov_fisher_full_khat
    v = r.s[r.s.param == "log_tau2"].set_index(["d", "nu0", "n", "law"]).cov_fisher_full_none
    print(pd.DataFrame(dict(signal_min=t, tau_corr=u, tau_uncorr=v)).unstack("law").round(3).to_string())


def joint(r):
    hdr("Joint chi-square coverage (log identifiable vector) at the largest n")
    j = r.j[(r.j.vector == "log_vector")]
    j = j[j.n == j.groupby("d").n.transform("max")]
    m = ["fisher_full_khat", "hess_full_khat", "fisher_full_ktrue", "fisher_full_none",
         "fisher_block_khat", "hess_block_khat", "fisher0_full_khat", "hess0_full_khat"]
    print(j[j.method.isin(m)].pivot_table(index=["d", "nu0", "law"], columns="method",
                                          values="coverage").mul(100).round(1).to_string())
    hdr("Joint coverage by n (fisher_full_khat), all laws")
    jj = r.j[(r.j.vector == "log_vector") & (r.j.method == "fisher_full_khat")]
    print(jj.pivot_table(index=["d", "nu0", "n"], columns="law", values="coverage").mul(100).round(1).to_string())
    hdr("Raw public vector (d>=4) joint coverage, fisher_full_khat")
    jr = r.j[(r.j.vector == "raw_public") & (r.j.method == "fisher_full_khat")]
    print(jr.pivot_table(index=["d", "nu0", "n"], columns="law", values="coverage").mul(100).round(1).to_string())


def kappa(r):
    hdr("Fourth-cumulant estimator kappa_tilde: mean, SD vs truth, by law and n")
    t = r.nug[["d", "nu0", "n", "law", "kappa_true", "kappa_hat_mean", "kappa_hat_sd",
               "kappa_hat_n", "kappa_pair_mean", "kappa_pair_sd"]].copy()
    t["sqrt_n_sd"] = np.sqrt(t.n) * t.kappa_hat_sd
    print(t.sort_values(["law", "d", "nu0", "n"]).round(4).to_string(index=False))
    hdr("Nugget coverage: true kappa vs kappa_tilde vs uncorrected, largest n")
    s = r.s[(r.s.param == "log_tau2")]
    s = s[s.n == s.groupby("d").n.transform("max")]
    print(s.pivot_table(index=["d", "nu0", "law"],
                        values=["cov_fisher_full_khat", "cov_fisher_full_ktrue",
                                "cov_fisher_full_none", "cov_fisher_full_kpair"]).mul(100).round(1).to_string())


def range_nuisance(r):
    hdr("Appendix D.7: d=3 full vs block, Gaussian, all n")
    for nu0 in (0.5, 1.5):
        for p in params_for(3):
            s = r.S(3, nu0, p=p)
            t = pd.DataFrame(dict(n=s.n, full_over_block=s.sd_gauss_full / s.sd_gauss_block,
                                  rmse_full=s.rmse / s.sd_gauss_full,
                                  rmse_block=s.rmse / s.sd_gauss_block,
                                  cov_full=100 * s.cov_fisher_full_khat,
                                  cov_block=100 * s.cov_fisher_block_khat,
                                  covH_full=100 * s.cov_hess_full_khat,
                                  covH_block=100 * s.cov_hess_block_khat))
            print(f"nu0={nu0} {p}")
            print(t.round(3).to_string(index=False))


def nugget_tail(r):
    hdr("Appendix D.8: skewness of standardized nugget errors, by law and n")
    rows = []
    for d, nu0 in SETTINGS:
        for law in noise.LAWS:
            rr = r.reps(d, nu0, law)
            for n in sorted(rr.n.unique()):
                x = rr[rr.n == n]
                rows.append(dict(d=d, nu0=nu0, law=law, n=n, skew=skew(x.log_tau2_err),
                                 tau_lo=100 * np.mean(x.log_tau2_hat <= TAU2_LO + 1e-6)))
    t = pd.DataFrame(rows)
    print(t.pivot_table(index=["d", "nu0", "n"], columns="law", values="skew").round(2).to_string())
    print("\n% tau2 at lower bound:")
    print(t.pivot_table(index=["d", "nu0", "n"], columns="law", values="tau_lo").round(2).to_string())
    for law in noise.LAWS:
        x = r.reps(3, 0.5, law, 1000)
        sd = r.sd(3, 0.5, "log_tau2", 1000)
        z = x.log_tau2_err / sd
        k = np.argsort(z.to_numpy())[:5]
        nu = np.exp(x.log_nu_hat.to_numpy())
        print(f"d=3 nu0=1/2 n=1000 {law}: five smallest z={np.round(z.to_numpy()[k],2)}, "
              f"nu_hat={np.round(nu[k],3)}, tau2_hat={np.round(np.exp(x.log_tau2_hat.to_numpy()[k]),4)}; "
              f"median nu_hat={np.median(nu):.3f}")


def ks(r):
    hdr("KS distance of standardized errors (full information SD) at n ~ 1e3, 1e4, 1e5, Gaussian")
    for d, nu0 in SETTINGS:
        rr = r.reps(d, nu0)
        for n in qq_ns(rr.n.unique()):
            x = rr[rr.n == n]
            out = []
            for p in params_for(d):
                z = x[f"{p}_err"] / r.sd(d, nu0, p, n)
                out.append(f"{p}={kstest(z, 'norm').statistic:.3f}")
            print(f"d={d} nu0={nu0} n={n}:", " ".join(out))


def eight(r):
    hdr("THE EIGHT STANDARD ERRORS: largest n, Gaussian; and pooled over laws (signal)")
    for metric in ("cov", "se_rms_ratio", "se_ratio", "rmsz", "madz", "se_cv", "n_se"):
        rows = []
        for d, nu0 in SETTINGS:
            n = r.nmax(d)
            s = r.S(d, nu0)
            s = s[s.n == n]
            for _, x in s.iterrows():
                row = dict(d=d, nu0=nu0, p=x.param)
                for m in methods(3):
                    row[m] = x.get(f"{metric}_{m}_khat", np.nan)
                rows.append(row)
        print(f"\n--- {metric}")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
    hdr("Eight SEs across n (Gaussian): coverage and RMS SE / RMSE")
    for d, nu0 in SETTINGS:
        for p in params_for(d):
            s = r.S(d, nu0, p=p)
            t = pd.DataFrame({"n": s.n})
            for m in methods(d):
                t[m + ":cov"] = 100 * s[f"cov_{m}_khat"]
            for m in methods(d):
                t[m + ":ratio"] = s[f"se_rms_ratio_{m}_khat"]
            for m in ("hess_full", "hess0_full"):
                t[m + ":nse"] = s[f"n_se_{m}_khat"]
            print(f"\nd={d} nu0={nu0} {p}")
            print(t.round(3).to_string(index=False))
    hdr("Eight SEs, all laws, nugget, largest n")
    s = r.s[(r.s.param == "log_tau2")]
    s = s[s.n == s.groupby("d").n.transform("max")]
    cols = [f"cov_{m}_khat" for m in methods(3)]
    print(s.set_index(["d", "nu0", "law"])[cols].mul(100).round(1).to_string())
    hdr("Average |coverage - 95| and |log(RMS SE/RMSE)| over params, settings, laws; by n-band")
    rows = []
    for _, x in r.s.iterrows():
        for m in methods(int(x.d)):
            rows.append(dict(d=x.d, nu0=x.nu0, n=x.n, law=x.law, p=x.param, m=m,
                             cov=x[f"cov_{m}_khat"], ratio=x[f"se_rms_ratio_{m}_khat"],
                             rmsz=x[f"rmsz_{m}_khat"], cv=x[f"se_cv_{m}_khat"],
                             nse=x[f"n_se_{m}_khat"], nfit=x.nfit))
    t = pd.DataFrame(rows)
    t["band"] = pd.cut(t.n, [0, 1500, 15000, 2e5], labels=["n<1.5e3", "1.5e3-1.5e4", ">1.5e4"])
    t["abs_cov_err"] = (100 * t["cov"] - 95).abs()
    t["abs_log_ratio"] = np.log(t.ratio).abs()
    t["nan_rate"] = 1 - t.nse / t.nfit
    for dsel in ("d=3", "d>=4"):
        u = t[(t.d == 3) if dsel == "d=3" else (t.d >= 4)]
        print(f"\n{dsel}")
        print(u.groupby(["band", "m"], observed=True)[["abs_cov_err", "abs_log_ratio", "cv", "nan_rate"]]
              .mean().round(3).unstack("band").to_string())
    return t


def cost(out):
    hdr("Computing cost from the fit files is reported by `sacct`; see the paper text")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    r = R(a.out)
    secs = dict(boundary=boundary, slopes=slopes, benchmark=benchmark,
                identifiability=identifiability, smoothness=smoothness, nugget=nugget,
                laws_signal=laws_signal, coverage=coverage, joint=joint, kappa=kappa,
                range=range_nuisance, nugget_tail=nugget_tail, ks=ks, eight=eight)
    only = [s for s in a.only.split(",") if s] or list(secs)
    for k in only:
        secs[k](r)


if __name__ == "__main__":
    main()
