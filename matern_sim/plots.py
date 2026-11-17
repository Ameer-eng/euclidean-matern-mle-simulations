"""Figures for Section 4 and Appendix D of the paper.

  python -m matern_sim.plots --out results --fig paper/figures

Reads results/summary/{replicates,summary,theory}.csv (written by
matern_sim.aggregate) and writes
  matern_mle_log_d{d}_nu{nu0}.pdf   Figures 1-6 (log scale, Gaussian errors)
  matern_mle_d{d}_nu{nu0}.pdf       Figures D1-D6 (original scale)
  matern_mle_laws.pdf               Figure 7 (error laws)
  matern_mle_se.pdf                 Figure 8 (the eight standard errors)
Needs matplotlib (not used by the simulation itself).
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import beta, norm  # noqa: E402

from . import config, noise  # noqa: E402
from .report import SETTINGS, ls_slope, params_for, qq_ns, rate, slope_x  # noqa: E402

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.6,
    "pdf.fonttype": 42, "axes.linewidth": 0.6, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "lines.linewidth": 0.9,
})

PCOL = {"log_nu": "#2b6cb0", "log_c": "#7b3fbf", "log_sigma2": "#c2410c",
        "log_ell": "#d97706", "log_tau2": "#2f855a"}
BOXCOL = "#e08a4a"
LOGLAB = {"log_nu": r"\log\nu", "log_c": r"\log c", "log_sigma2": r"\log\sigma^2",
          "log_ell": r"\log l", "log_tau2": r"\log\tau^2"}
RAWLAB = {"log_nu": r"\nu", "log_c": r"c", "log_sigma2": r"\sigma^2",
          "log_ell": r"l", "log_tau2": r"\tau^2"}
LAWS = ("gauss", "unif", "mixnorm", "bern")
LAWNAME = {"gauss": "normal", "unif": "uniform", "mixnorm": "normal mixture",
           "bern": "Bernoulli"}
LAWMARK = {"gauss": "o", "unif": "^", "mixnorm": "s", "bern": "D"}
LAWCOL = {"gauss": "#1f2937", "unif": "#2b6cb0", "mixnorm": "#c2410c", "bern": "#2f855a"}
NUSTR = {0.5: "1/2", 1.5: "3/2"}


# ------------------------------------------------------------------ data
class Data:
    def __init__(self, out):
        sdir = os.path.join(out, "summary")
        self.rep = pd.read_csv(os.path.join(sdir, "replicates.csv"))
        self.rep = self.rep[self.rep.variant == "joint"]
        self.summ = pd.read_csv(os.path.join(sdir, "summary.csv"))
        self.summ = self.summ[self.summ.variant == "joint"]
        th = pd.read_csv(os.path.join(sdir, "theory.csv"))
        self.th = th[th.variant == "joint"]

    def reps(self, d, nu0, law="gauss"):
        r = self.rep
        return r[(r.d == d) & (r.nu0 == nu0) & (r.law == law)]

    def theory(self, d, nu0, p, law="gauss"):
        t = self.th
        return t[(t.d == d) & (t.nu0 == nu0) & (t.law == law) & (t.param == p)].sort_values("n")


def g0_of(r, p):
    """True value of parameter p (log scale) from a replicate frame."""
    return float((r[f"{p}_hat"] - r[f"{p}_err"]).iloc[0])


def errors(r, p, raw):
    e = r[f"{p}_err"].to_numpy()
    if raw:
        g0 = g0_of(r, p)
        e = np.exp(g0 + e) - math.exp(g0)
    return e


def err_summaries(e, rng, B=1000):
    rmse = math.sqrt(np.mean(e**2))
    q = lambda x: (np.percentile(x, 75, axis=-1) - np.percentile(x, 25, axis=-1)) / 1.349
    iqr = float(q(e))
    idx = rng.integers(0, len(e), size=(B, len(e)))
    eb = e[idx]
    rb = np.sqrt(np.mean(eb**2, axis=1))
    ib = q(eb)
    return rmse, np.quantile(rb, [0.025, 0.975]), iqr, np.quantile(ib, [0.025, 0.975])


def shades(col, k=3):
    c = np.array(matplotlib.colors.to_rgb(col))
    return [tuple(1 - (1 - c) * f) for f in np.linspace(0.45, 1.0, k)]


# ---------------------------------------------------------- Figures 1-6, D1-D6
def fig_setting(D, d, nu0, raw, path):
    params = params_for(d)
    r = D.reps(d, nu0)
    ns = np.array(sorted(r.n.unique()))
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 6.6))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.075,
                        wspace=0.38, hspace=0.78)
    lab = RAWLAB if raw else LOGLAB
    scale = "original scale" if raw else "log scale"
    fig.suptitle(rf"$d={d}$, $\nu_0={NUSTR[nu0]}$: joint MLE of the Matérn parameters "
                 rf"({scale}; $\sigma_0^2=1$, $l_0=0.4$, $\tau_0^2=0.25$; Gaussian errors, "
                 rf"{config.NREP} replicates per $n$)", fontsize=8)
    letters = "abcdefgh"
    qn = qq_ns(ns)
    for j, p in enumerate(params):
        ax = axes[0, j]
        col = PCOL[p]
        rm, rci, iq, ici = [], [], [], []
        for n in ns:
            e = errors(r[r.n == n], p, raw)
            a, b, c, dd = err_summaries(e, rng)
            rm.append(a), rci.append(b), iq.append(c), ici.append(dd)
        rm, iq = np.array(rm), np.array(iq)
        rci, ici = np.array(rci), np.array(ici)
        th = D.theory(d, nu0, p)
        g0 = g0_of(r, p)
        fac = math.exp(g0) if raw else 1.0
        tn = th.n.to_numpy()
        main = th.sd_gauss_block if d <= 3 else th.sd_gauss_full
        main = main.to_numpy() * fac
        full = th.sd_gauss_full.to_numpy() * fac
        xs = slope_x(d, p, ns)
        on_mc = np.isin(tn, ns)
        sl = dict(rmse=ls_slope(xs, rm), iqr=ls_slope(xs, iq),
                  asy=ls_slope(slope_x(d, p, tn[on_mc]), main[on_mc]),
                  full=ls_slope(slope_x(d, p, tn[on_mc]), full[on_mc]))
        sword = r"slope in $\log\log n$" if (d == 4 and p in ("log_sigma2", "log_ell")) \
            else "fitted slope"
        ax.errorbar(ns / 1.06, rm, yerr=[rm - rci[:, 0], rci[:, 1] - rm], fmt="o", ms=3.2,
                    color=col, ecolor=col, elinewidth=0.7, capsize=1.5,
                    label=f"RMSE ({sword} = {sl['rmse']:.2f})")
        ax.errorbar(ns * 1.06, iq, yerr=[iq - ici[:, 0], ici[:, 1] - iq], fmt="s", ms=3.2,
                    mfc="white", color=col, ecolor=col, elinewidth=0.7, capsize=1.5,
                    label=f"IQR/1.349 ({sword} = {sl['iqr']:.2f})")
        ax.plot(tn, main, color="0.55", lw=1.6, zorder=1.5,
                label=(r"Asy. SD, identifiable block" if d <= 3 else "Asymptotic SD")
                + f" ({sl['asy']:.2f})")
        if d <= 3:
            # drawn above the markers: the Monte Carlo errors follow this line closely
            ax.plot(tn, full, color="black", ls=(0, (1, 1.3)), lw=1.8, zorder=4,
                    label=f"Asy. SD, full information ({sl['full']:.2f})")
        rv, rlab, rsl = rate(d, nu0, p, tn)
        rv = rv * main[-1] / rv[-1]
        ax.plot(tn, rv, "k--", lw=1.1, zorder=1.5, label=f"Theoretical rate {rlab} (slope = {rsl:.3g})")
        if tn.max() > ns.max() * 1.01:
            ax.axvspan(ns.max() * 1.25, tn.max() * 1.4, color="0.9", lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ns.min() / 1.6, tn.max() * 1.4)
        ax.set_xlabel(r"$n$ (log scale)")
        ax.set_ylabel(rf"error of ${lab[p]}$" if not raw else rf"error of $\widehat{{{lab[p]}}}$")
        ax.set_title(f"({letters[j]}) ${lab[p]}$", loc="left")
        hs, ls_ = ax.get_legend_handles_labels()
        order = sorted(range(len(ls_)), key=lambda i: (not ls_[i].startswith(("RMSE", "IQR")), i))
        ax.legend([hs[i] for i in order], [ls_[i] for i in order], loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), frameon=False, handlelength=2.2,
                  borderaxespad=0.0)

        # Q-Q plot
        ax = axes[1, j]
        R = None
        cols = shades(col)
        lim = 5.0
        for k, n in enumerate(qn):
            e = errors(r[r.n == n], p, raw)
            sd = float(th.loc[th.n == n, "sd_gauss_full"].iloc[0]) * fac
            z = np.sort(e / sd)
            R = len(z)
            q = norm.ppf((np.arange(1, R + 1) - 0.5) / R)
            inside = np.abs(z) <= lim
            ax.plot(q[inside], z[inside], ".", ms=2.2, color=cols[k], label=f"$n={n:,}$")
            hi, lo = z > lim, z < -lim
            ax.plot(q[hi], np.full(hi.sum(), lim * 0.97), "^", ms=2.6, color=cols[k])
            ax.plot(q[lo], np.full(lo.sum(), -lim * 0.97), "v", ms=2.6, color=cols[k])
        i = np.arange(1, R + 1)
        q = norm.ppf((i - 0.5) / R)
        band = norm.ppf(np.stack([beta.ppf(0.025, i, R - i + 1), beta.ppf(0.975, i, R - i + 1)]))
        ax.fill_between(q, band[0], band[1], color="0.82", lw=0, zorder=0,
                        label=r"95% band for exact $N(0,1)$")
        ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=0.6, ls="--", zorder=0.5)
        ax.set_xlim(-3.3, 3.3)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("standard normal quantiles")
        ax.set_ylabel("standardized error")
        ax.set_title(f"({letters[j + 4]}) normal Q–Q, ${lab[p]}$", loc="left")
        ax.legend(loc="upper left", frameon=False, markerscale=2.0, handlelength=1.2)

    if d <= 3:   # box plots of the non-consistent sigma2 and l
        for row, p in ((0, "log_sigma2"), (1, "log_ell")):
            ax = axes[row, 3]
            vals = [r.loc[r.n == n, f"{p}_hat"].to_numpy() for n in ns]
            g0 = g0_of(r, p)
            if raw:
                vals = [np.exp(v) for v in vals]
                ax.set_yscale("log")
            bp = ax.boxplot(vals, positions=np.log10(ns), widths=0.16, patch_artist=True,
                            whis=1.5, showfliers=True,
                            flierprops=dict(marker="o", ms=1.4, mec=BOXCOL, alpha=0.6),
                            medianprops=dict(color="#7c2d12", lw=0.9),
                            boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0.6))
            for b in bp["boxes"]:
                b.set_facecolor(BOXCOL)
                b.set_alpha(0.75)
            ax.axhline(math.exp(g0) if raw else g0, color="k", ls="--", lw=0.7,
                       label="true value")
            ax.set_xticks(np.log10(ns))
            ax.set_xticklabels([f"{x:.1f}" for x in np.log10(ns)], fontsize=5.5, rotation=90)
            ax.set_xlim(np.log10(ns).min() - 0.25, np.log10(ns).max() + 0.25)
            ax.set_xlabel(r"$\log_{10} n$")
            ax.set_ylabel(rf"estimate of ${lab[p]}$")
            ax.set_title(f"({'dh'[row]}) ${lab[p]}$ estimates (not consistent)", loc="left")
            ax.legend(loc="upper right", frameon=False)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 7
def paired_ratio_ci(a, b, rng, B=1000):
    idx = rng.integers(0, len(a), size=(B, len(a)))
    rb = np.sqrt(np.mean(a[idx] ** 2, 1)) / np.sqrt(np.mean(b[idx] ** 2, 1))
    return np.quantile(rb, [0.025, 0.975])


def fig_laws(D, path):
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(2, 6, figsize=(11.0, 4.6))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.2, wspace=0.42,
                        hspace=0.55)
    fig.suptitle(r"Effect of the error law (" + f"{config.NREP}" +
                 r" replicates per $n$ and law; the same fields for the four laws)",
                 fontsize=8)
    for k, (d, nu0) in enumerate(SETTINGS):
        ax = axes[0, k]
        for law in LAWS:
            r = D.reps(d, nu0, law)
            ns = np.array(sorted(r.n.unique()))
            v = [math.sqrt(n) * math.sqrt(np.mean(r.loc[r.n == n, "log_tau2_err"] ** 2))
                 for n in ns]
            ax.plot(ns, v, LAWMARK[law], ms=3, mfc="white" if law != "gauss" else LAWCOL[law],
                    color=LAWCOL[law], mew=0.8)
            th = D.theory(d, nu0, "log_tau2", law)
            ax.plot(th.n, np.sqrt(th.n) * th.sd_law_full, color=LAWCOL[law], lw=0.8)
            m4 = noise.MOMENTS[law]["m4"]
            ax.axhline(math.sqrt(m4 - 1), color=LAWCOL[law], ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$")
        if k == 0:
            ax.set_ylabel(r"$\sqrt{n}\,\mathrm{RMSE}(\log\widehat\tau^2)$")
        ax.set_title(f"({'abcdef'[k]}) $d={d}$, $\\nu_0={NUSTR[nu0]}$", loc="left")

        ax = axes[1, k]
        g = D.reps(d, nu0, "gauss").sort_values(["n", "rep"])
        ns = np.array(sorted(g.n.unique()))
        for j, law in enumerate(LAWS[1:]):
            r = D.reps(d, nu0, law).sort_values(["n", "rep"])
            rat, lo, hi = [], [], []
            for n in ns:
                a = r.loc[r.n == n, "log_nu_err"].to_numpy()
                b = g.loc[g.n == n, "log_nu_err"].to_numpy()
                rat.append(math.sqrt(np.mean(a**2) / np.mean(b**2)))
                ci = paired_ratio_ci(a, b, rng)
                lo.append(ci[0]), hi.append(ci[1])
            rat = np.array(rat)
            off = 1.07 ** (j - 1)
            ax.errorbar(ns * off, rat, yerr=[rat - np.array(lo), np.array(hi) - rat],
                        fmt=LAWMARK[law], ms=3, mfc="white", color=LAWCOL[law],
                        elinewidth=0.6, capsize=1.2, mew=0.8)
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel(r"$n$")
        if k == 0:
            ax.set_ylabel(r"RMSE$(\log\widehat\nu)$ relative to normal errors")
        ax.set_title(f"({'ghijkl'[k]}) $d={d}$, $\\nu_0={NUSTR[nu0]}$", loc="left")
    hs = [Line2D([], [], marker=LAWMARK[l], ls="", color=LAWCOL[l],
                 mfc=LAWCOL[l] if l == "gauss" else "white", ms=4) for l in LAWS]
    hs += [Line2D([], [], color="0.3", lw=0.8), Line2D([], [], color="0.3", lw=0.6, ls="--")]
    fig.legend(hs, [LAWNAME[l] for l in LAWS] +
               ["exact law-specific asymptotic SD", r"limit $\sqrt{E\xi^4-1}$"],
               loc="lower center", ncol=6, frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 8
METHODS = [("fisher", "full"), ("hess", "full"), ("fisher0", "full"), ("hess0", "full"),
           ("fisher", "block"), ("hess", "block"), ("fisher0", "block"), ("hess0", "block")]
MSTYLE = {"fisher": ("#2b6cb0", "o"), "hess": ("#c2410c", "s"),
          "fisher0": ("#2b6cb0", "^"), "hess0": ("#c2410c", "D")}
MLABEL = {"fisher": r"Fisher at $\widehat\theta_n$", "hess": r"Hessian at $\widehat\theta_n$",
          "fisher0": r"Fisher at $\theta_0$", "hess0": r"Hessian at $\theta_0$"}


def se_curves(S, d, nu0):
    """{(method, inverse): (coverage %, robust SD ratio, invalid %)}, each a Series over n,
    for the khat-corrected standard errors of Figure E1:
      coverage  arithmetic mean of cov_* over the (law x identifiable parameter) rows;
      ratio     geometric mean of madz_* (median |error / SE| / 0.6745) over the same rows;
      invalid   100 (1 - sum n_se / sum nfit), pooled over the same rows, where n_se counts
                the replicates with a finite, positive delta-mapped variance."""
    s = S[(S.d == d) & (S.nu0 == nu0) & S.param.isin(params_for(d))]
    out = {}
    for m, inv in METHODS:
        if d >= 4 and inv == "block":
            continue
        key = f"{m}_{inv}_khat"
        g = s.groupby("n")
        cov = 100 * g[f"cov_{key}"].mean()
        rat = np.exp(g[f"madz_{key}"].apply(lambda v: np.mean(np.log(v))))
        neg = 100 * (1 - g[f"n_se_{key}"].sum() / g["nfit"].sum())
        out[(m, inv)] = (cov, rat, neg)
    return out


def fig_se(D, path):
    """Coverage, robust SD of error/SE, and % invalid variances (delta-mapped
    variance non-finite or <= 0; pooled 100 (1 - sum n_se / sum nfit)) of the eight
    standard errors vs n, averaged over the identifiable parameters and the four
    laws (geometric mean for the robust SD)."""
    S = D.summ
    fig, axes = plt.subplots(3, 6, figsize=(11.0, 6.4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.14, wspace=0.42,
                        hspace=0.5)
    fig.suptitle(r"Accuracy of the eight standard errors (averages over the identifiable "
                 r"log-parameters and the four error laws; " + f"{config.NREP}" +
                 r" replicates per $n$ and law)", fontsize=8)
    for k, (d, nu0) in enumerate(SETTINGS):
        for (m, inv), ys in se_curves(S, d, nu0).items():
            col, mk = MSTYLE[m]
            ls = "-" if inv == "full" else ":"
            mfc = col if m in ("fisher", "hess") else "white"
            for row, y in enumerate(ys):
                axes[row, k].plot(y.index, y.to_numpy(), ls=ls, marker=mk, ms=2.6, color=col,
                                  mfc=mfc, lw=0.7, mew=0.7)
        ax = axes[0, k]
        ax.axhline(95, color="k", lw=0.5, ls="--")
        ax.set_ylim(40, 100)
        ax.set_title(f"({'abcdef'[k]}) $d={d}$, $\\nu_0={NUSTR[nu0]}$", loc="left")
        ax = axes[1, k]
        ax.axhline(1, color="k", lw=0.5, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(0.15, 8)
        ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"({'ghijkl'[k]})", loc="left")
        ax = axes[2, k]
        ax.set_ylim(-2, 70)
        ax.set_xlabel(r"$n$")
        ax.set_title(f"({'mnopqr'[k]})", loc="left")
        for row in range(3):
            axes[row, k].set_xscale("log")
    axes[0, 0].set_ylabel("coverage (%)")
    axes[1, 0].set_ylabel(r"robust SD of error / SE")
    axes[2, 0].set_ylabel("invalid variance (%)")
    hs, ls_ = [], []
    for m in ("fisher", "hess", "fisher0", "hess0"):
        col, mk = MSTYLE[m]
        hs.append(Line2D([], [], color=col, marker=mk, ms=4,
                         mfc=col if m in ("fisher", "hess") else "white"))
        ls_.append(MLABEL[m])
    hs += [Line2D([], [], color="0.3", lw=0.8), Line2D([], [], color="0.3", lw=0.8, ls=":")]
    ls_ += ["block of the full inverse", "inverse of the identifiable block ($d=3$)"]
    fig.legend(hs, ls_, loc="lower center", ncol=6, frameon=False, fontsize=6.5)
    fig.savefig(path)
    plt.close(fig)


SE_DIM_RC = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
             "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
             "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5}


def fig_se_dim(D, d, path, png=None):
    """Figure E1 for one dimension d (3 rows x 2 columns: nu0 = 1/2, 3/2), sized for a
    6.5 in text width. Same statistics as fig_se (se_curves); only the layout differs.
    Returns {(d, nu0, row): [(x, y), ...]}, the plotted arrays in drawing order."""
    with plt.rc_context(SE_DIM_RC):
        fig, axes = plt.subplots(3, 2, figsize=(6.5, 5.8), sharex="col")
        fig.subplots_adjust(left=0.12, right=0.98, top=0.925, bottom=0.165, wspace=0.22,
                            hspace=0.36)
        fig.suptitle(rf"$d={d}$", fontsize=10.5)
        arr = {}
        for k, nu0 in enumerate((0.5, 1.5)):
            for (m, inv), ys in se_curves(D.summ, d, nu0).items():
                col, mk = MSTYLE[m]
                ls = "-" if inv == "full" else ":"
                mfc = col if m in ("fisher", "hess") else "white"
                for row, y in enumerate(ys):
                    axes[row, k].plot(y.index, y.to_numpy(), ls=ls, marker=mk, ms=4.2,
                                      color=col, mfc=mfc, lw=1.0, mew=0.9)
            ax = axes[0, k]
            ax.axhline(95, color="k", lw=0.6, ls="--")
            ax.set_ylim(40, 100)
            ax.set_title(rf"({'ab'[k]}) $\nu_0={NUSTR[nu0]}$", loc="left")
            ax = axes[1, k]
            ax.axhline(1, color="k", lw=0.6, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(0.15, 8)
            ax.set_yticks([0.2, 0.5, 1, 2, 5])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"({'cd'[k]})", loc="left")
            ax = axes[2, k]
            ax.set_ylim(-2, 70)
            ax.set_xlabel(r"$n$ (log scale)")
            ax.set_title(f"({'ef'[k]})", loc="left")
            for row in range(3):
                ax = axes[row, k]
                ax.set_xscale("log")
                ax.grid(True, which="major", color="0.9", lw=0.5)
                ax.set_axisbelow(True)
                arr[(d, nu0, row)] = [(np.asarray(ln.get_xdata(), float),
                                       np.asarray(ln.get_ydata(), float))
                                      for ln in ax.get_lines()
                                      if ln.get_marker() not in (None, "None", "")]
        axes[0, 0].set_ylabel("coverage (%)")
        axes[1, 0].set_ylabel("Median |error/SE| / 0.67449")
        axes[2, 0].set_ylabel("invalid variance (%)")
        hs, ls_ = [], []
        for m in ("fisher", "hess", "fisher0", "hess0"):
            col, mk = MSTYLE[m]
            hs.append(Line2D([], [], color=col, marker=mk, ms=5,
                             mfc=col if m in ("fisher", "hess") else "white"))
            ls_.append(MLABEL[m])
        hs.append(Line2D([], [], color="0.3", lw=1.0))
        ls_.append("block of the full inverse")
        if d == 3:
            hs.append(Line2D([], [], color="0.3", lw=1.0, ls=":"))
            ls_.append("inverse of the identifiable block")
        fig.legend(hs, ls_, loc="lower center", ncol=3, frameon=False, columnspacing=1.6,
                   handlelength=2.4)
        fig.savefig(path)
        if png:
            fig.savefig(png, dpi=160)
        plt.close(fig)
    return arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fig", default=os.path.join("paper", "figures"))
    ap.add_argument("--only", default="", help="comma list: setting,laws,se (se = matern_mle_se.pdf and matern_mle_se_d{3,4,5}.pdf)")
    args = ap.parse_args(argv)
    os.makedirs(args.fig, exist_ok=True)
    D = Data(args.out)
    only = set(filter(None, args.only.split(",")))
    if not only or "setting" in only:
        for d, nu0 in SETTINGS:
            for raw in (False, True):
                name = f"matern_mle_{'' if raw else 'log_'}d{d}_nu{nu0}.pdf"
                fig_setting(D, d, nu0, raw, os.path.join(args.fig, name))
                print("wrote", name)
    if not only or "laws" in only:
        fig_laws(D, os.path.join(args.fig, "matern_mle_laws.pdf"))
        print("wrote matern_mle_laws.pdf")
    if not only or "se" in only:
        fig_se(D, os.path.join(args.fig, "matern_mle_se.pdf"))
        print("wrote matern_mle_se.pdf")
        for d in (3, 4, 5):
            fig_se_dim(D, d, os.path.join(args.fig, f"matern_mle_se_d{d}.pdf"))
            print(f"wrote matern_mle_se_d{d}.pdf")


if __name__ == "__main__":
    main()
