"""FOLLOW-UP Task 1 (2026-11-10): decompose the joint log-parameter undercoverage.

For one setting (d, nu0, G) and the Gaussian law, uses the saved fits
(results/fits), the saved score at the truth (audit/out/score0{,_local}), the
Fisher information at the truth (results/theory) and kappa_tilde (results/kappa).
No data are regenerated and nothing is refitted.

Coordinates.  The code stores w = (eta, nu, log a, log t).  The paper uses
theta = (eta, nu, a, t).  With D0 = diag(1, 1, 1/a0, 1/t0) = d w / d theta at theta0,
  s_theta = D0 s_w,  I_theta = D0 I_w D0,
  h_hat = E_n^{-1}(theta_hat - theta0)      (a, t on the original scale),
  Delta = E_n^T s_theta,  J = E_n^T I_theta E_n,
  S = J^{-1/2} Delta,  T = J^{1/2} h_hat,  T - S = J^{1/2}(h_hat - J^{-1} Delta),
with symmetric square roots.  For Gaussian errors Cov(S) = I exactly.

Chain of joint statistics (all chi-square(4) regions at level 0.95):
  Q_S         ||S||^2                                score only (exact covariance)
  Q_T         ||T||^2 = dtheta' I_theta dtheta       linearisation in theta (paper)
  Q_Tw        dw' I_w dw                             linearisation in w (code chart)
  Q_or0       e' (J0 I_w^{-1} J0')^{-1} e            reported log vector, truth, kappa = 0
  Q_orK       as Q_or0 with + kappa_tilde/(n t0^2) on the log t variance (published oracle)
  Q_fit0      e' (J I_hat^{-1} J')^{-1} e at theta_hat, kappa = 0
  Q_fitK      as Q_fit0 with + kappa_tilde/(n t_hat^2) (published fitted region)
Here e = g(w_hat) - g(w0) for the reported log vector g (log nu, log sigma2, log l,
log tau2 for d >= 4; log nu, log c, log tau2 for d <= 3).  Q_or0 equals Q_Tw up to the
nonlinearity of g; Q_Tw equals Q_T up to the a -> log a, t -> log t change.

  python -m audit.joint_diagnostic --d 5 --nu0 0.5 --G 10 8 6 --out audit/out/followup
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, skew

from matern_sim import aggregate as A

CRIT4 = chi2.ppf(0.95, 4)
QS = (0.5, 0.9, 0.95, 0.99)


def score_file(tagg, law, rep):
    for s in ("audit/out/score0_local", "audit/out/score0"):
        p = os.path.join(s, "ok", tagg, "joint", law, f"rep{rep:04d}.npz")
        if os.path.exists(p):
            return p
    return None


def symsqrt(M, inv=False):
    lam, U = np.linalg.eigh(0.5 * (M + M.T))
    p = -0.5 if inv else 0.5
    return (U * lam**p) @ U.T


def En_matrix(n, nu0, d):
    R = n ** (1.0 / (2 * nu0 + d))
    L = math.log(R)
    r = R ** (d / 2)
    E = np.zeros((4, 4))
    E[0, 0] = 1 / r
    E[0, 1] = 2 * L / r
    E[1, 1] = 1 / r
    if d >= 4:
        b = math.sqrt(math.log(n)) if d == 4 else R ** ((d - 4) / 2)
        E[2, 2] = 1 / b
    else:
        E[2, 2] = 1.0       # range not normalised in d <= 3 (not identifiable)
    E[3, 3] = n ** -0.5
    return E, dict(R=R, L=L, r=r, b=1 / E[2, 2])


def wald(e, S):
    """chi-square statistic with Cholesky; NaN unless S is finite and PD."""
    S = 0.5 * (S + S.T)
    try:
        Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return math.nan
    z = np.linalg.solve(Lc, e)
    return float(z @ z)


def load(d, nu0, G, law="gauss", nrep=400):
    tagg = f"d{d}_nu{nu0:g}_G{G}"
    th = dict(np.load(f"results/theory/{tagg}.npz"))
    rows = []
    for rep in range(nrep):
        fp = f"results/fits/{tagg}/joint/{law}/rep{rep:04d}.npz"
        sp = score_file(tagg, law, rep)
        if not (os.path.exists(fp) and sp):
            continue
        with np.load(fp) as z:
            r = dict(rep=rep, theta_hat=np.array(z["theta_hat"]),
                     fisher_hat=np.array(z["fisher_hat"]), negH=np.array(z["negH"]),
                     refit="refit" in z.files, at_bound=np.array(z["at_bound"]))
        with np.load(sp) as z:
            r["score0"] = np.array(z["score0"])
            assert np.allclose(z["theta0"], th["theta0"])
        r["kappa"] = A.load_kappa(fp)
        rows.append(r)
    return th, rows


def analyse(d, nu0, G, law="gauss"):
    th, rows = load(d, nu0, G, law)
    m = len(rows)
    w0 = th["theta0"]
    n = int(th["n"])
    Iw = np.asarray(th["I0"], float)
    a0, t0 = math.exp(w0[2]), math.exp(w0[3])
    D0 = np.diag([1.0, 1.0, 1 / a0, 1 / t0])
    E, sc = En_matrix(n, nu0, d)
    It = D0 @ Iw @ D0
    J = E.T @ It @ E
    Jh, Jmh = symsqrt(J), symsqrt(J, inv=True)
    Jinv = np.linalg.inv(J)

    wh = np.stack([r["theta_hat"] for r in rows])
    sw = np.stack([r["score0"] for r in rows])
    khat = np.array([r["kappa"] for r in rows])
    dth = np.column_stack([wh[:, 0] - w0[0], wh[:, 1] - w0[1],
                           np.exp(wh[:, 2]) - a0, np.exp(wh[:, 3]) - t0])
    hhat = np.linalg.solve(E, dth.T).T
    Delta = (E.T @ (D0 @ sw.T)).T
    S = Delta @ Jmh.T
    T = hhat @ Jh.T
    hstar = Delta @ Jinv.T
    TS = T - S

    # reported log vector
    ps = A.params_for(d, "joint")
    idx = [A.PARAMS.index(p) for p in ps]
    g_hat = A.gvals(wh, d)[:, idx]
    g0 = A.gvals(w0[None], d)[0, idx]
    e = g_hat - g0
    J0 = A.gjac(w0[None], d)[0][idx]
    Jh_ = A.gjac(wh, d)[:, idx, :]
    V0 = np.linalg.inv(Iw)
    k = len(idx)
    crit = chi2.ppf(0.95, k)

    dw = wh - w0
    Q = dict(S=np.sum(S**2, 1), T=np.sum(T**2, 1),
             Tw=np.einsum("mi,ij,mj->m", dw, Iw, dw))
    Q["or0"] = np.array([wald(e[i], J0 @ V0 @ J0.T) for i in range(m)])
    orK = np.empty(m)
    fit0 = np.empty(m)
    fitK = np.empty(m)
    hesK = np.empty(m)
    for i, r in enumerate(rows):
        Vk = V0.copy()
        Vk[3, 3] += khat[i] / (n * t0**2)
        orK[i] = wald(e[i], J0 @ Vk @ J0.T)
        th_i = math.exp(wh[i, 3])
        Vf = np.linalg.inv(r["fisher_hat"])
        fit0[i] = wald(e[i], Jh_[i] @ Vf @ Jh_[i].T)
        Vf[3, 3] += khat[i] / (n * th_i**2)
        fitK[i] = wald(e[i], Jh_[i] @ Vf @ Jh_[i].T)
        Vh = np.linalg.inv(r["negH"])
        Vh[3, 3] += khat[i] / (n * th_i**2)
        hesK[i] = wald(e[i], Jh_[i] @ Vh @ Jh_[i].T)
    Q.update(orK=orK, fit0=fit0, fitK=fitK, hessK=hesK)

    # diagnostic recalibrations (NOT inference methods): empirical mean/covariance of T
    muT, CT = T.mean(0), np.cov(T.T)
    Q["T_recal_cov"] = np.einsum("mi,ij,mj->m", T, np.linalg.inv(CT), T)
    Q["T_recal_meancov"] = np.einsum("mi,ij,mj->m", T - muT, np.linalg.inv(CT), T - muT)
    CS = np.cov(S.T)
    Q["S_recal_cov"] = np.einsum("mi,ij,mj->m", S, np.linalg.inv(CS), S)

    cover = {kk: (v <= (CRIT4 if kk in ("S", "T", "Tw", "T_recal_cov", "T_recal_meancov",
                                         "S_recal_cov") else crit)).astype(float)
             for kk, v in Q.items()}
    for kk, v in Q.items():
        cover[kk][~np.isfinite(v)] = np.nan
    tagg = f"d{d}_nu{nu0:g}_G{G}"
    base = dict(d=d, nu0=nu0, G=G, n=n, law=law, nrep=m, n_refit=int(sum(r["refit"] for r in rows)),
                n_bound=int(sum(r["at_bound"].any() for r in rows)), **sc)

    # 1-2. moments and squared-norm quantiles
    vecs = []
    for name, X in (("S", S), ("T", T), ("T_minus_S", TS)):
        C = np.cov(X.T)
        ev = np.linalg.eigvalsh(C)
        vecs.append(dict(base, vector=name, mean=json.dumps(np.round(X.mean(0), 4).tolist()),
                         mean_se=json.dumps(np.round(X.std(0, ddof=1) / math.sqrt(m), 4).tolist()),
                         cov=json.dumps(np.round(C, 4).tolist()),
                         cov_eig=json.dumps(np.round(ev, 4).tolist()),
                         comp_skew=json.dumps(np.round(skew(X, axis=0), 3).tolist()),
                         comp_exkurt=json.dumps(np.round(kurtosis(X, axis=0), 3).tolist())))
    norms = []
    ref = chi2.ppf(QS, 4)
    norms.append(dict(base, statistic="chi2_4_reference", mean=4.0,
                      **{f"q{int(100*q)}": float(x) for q, x in zip(QS, ref)}, coverage=0.95))
    for kk in Q:
        v = Q[kk][np.isfinite(Q[kk])]
        norms.append(dict(base, statistic=kk, n_valid=len(v), mean=float(np.mean(v)),
                          **{f"q{int(100*q)}": float(np.quantile(v, q)) for q in QS},
                          coverage=float(np.nanmean(cover[kk]))))
    # 4. remainder
    rn = np.linalg.norm(TS, axis=1)
    rem = dict(base, rms_norm_T_minus_S=float(np.sqrt(np.mean(rn**2))),
               **{f"q{int(100*q)}_norm_T_minus_S": float(np.quantile(rn, q)) for q in QS},
               rms_norm_S=float(np.sqrt(np.mean(np.sum(S**2, 1)))),
               rms_norm_T=float(np.sqrt(np.mean(np.sum(T**2, 1)))),
               corr_QS_QT=float(np.corrcoef(Q["S"], Q["T"])[0, 1]),
               # h coordinates: remainder per coordinate, relative to sd of h*
               rms_h_minus_hstar=json.dumps(np.round(np.sqrt(np.mean((hhat - hstar)**2, 0)), 4).tolist()),
               sd_hstar=json.dumps(np.round(np.sqrt(np.diag(Jinv)), 4).tolist()),
               mean_h_minus_hstar=json.dumps(np.round(np.mean(hhat - hstar, 0), 4).tolist()))
    # standardized marginal score and estimator components in h coordinates
    zD = Delta / np.sqrt(np.diag(J))
    zh = hhat / np.sqrt(np.diag(Jinv))
    rem.update(score_comp_skew=json.dumps(np.round(skew(zD, axis=0), 3).tolist()),
               score_comp_exkurt=json.dumps(np.round(kurtosis(zD, axis=0), 3).tolist()),
               hhat_comp_skew=json.dumps(np.round(skew(zh, axis=0), 3).tolist()),
               hhat_comp_mean=json.dumps(np.round(zh.mean(0), 3).tolist()),
               hhat_comp_sd=json.dumps(np.round(zh.std(0, ddof=1), 3).tolist()))
    # 6. paired differences
    pairs = [("S", "T"), ("T", "Tw"), ("Tw", "or0"), ("or0", "orK"), ("orK", "fitK"),
             ("or0", "fit0"), ("fit0", "fitK"), ("S", "fitK"), ("S", "or0"), ("fitK", "hessK"),
             ("T", "T_recal_cov"), ("T", "T_recal_meancov")]
    pd_rows = []
    for a_, b_ in pairs:
        x = cover[b_] - cover[a_]
        ok = np.isfinite(x)
        x = x[ok]
        pd_rows.append(dict(base, from_=a_, to=b_, n_pairs=int(ok.sum()),
                            cov_from=float(np.nanmean(cover[a_])), cov_to=float(np.nanmean(cover[b_])),
                            diff=float(x.mean()), paired_se=float(x.std(ddof=1) / math.sqrt(len(x))),
                            discordant=int(np.sum(x != 0))))
    # direction of the remainder in the reported log coordinates.  Linear part
    # l = J0 I_w^{-1} s_w (so ||Sigma^{-1/2} l||^2 = ||S||^2), remainder r = e - l,
    # Sigma = J0 I_w^{-1} J0'.  Components along the eigenvectors of Sigma,
    # each standardized by its own SD (eigenvalue^{1/2}); ordered by SD.
    Sig = J0 @ V0 @ J0.T
    mu, U = np.linalg.eigh(Sig)
    lin = sw @ V0 @ J0.T
    remv = e - lin
    dirs = []
    for kk_ in range(len(mu)):
        u = U[:, kk_]
        u = u * np.sign(u[np.argmax(np.abs(u))])
        cl = lin @ u / math.sqrt(mu[kk_])
        cr = remv @ u / math.sqrt(mu[kk_])
        ce = e @ u / math.sqrt(mu[kk_])
        dirs.append(dict(base, eig_index=kk_, sd_direction=float(math.sqrt(mu[kk_])),
                         loading=json.dumps(dict(zip(ps, np.round(u, 4).tolist()))),
                         sd_linear_std=float(cl.std(ddof=1)), mean_rem_std=float(cr.mean()),
                         rms_rem_std=float(np.sqrt(np.mean(cr**2))),
                         share_of_rem_ms=float(np.mean(cr**2)),     # normalised below
                         skew_err_std=float(skew(ce)), mean_err_std=float(ce.mean()),
                         sd_err_std=float(ce.std(ddof=1)),
                         cover_1d=float(np.mean(np.abs(ce) <= 1.959963984540054))))
    tot = sum(r_["share_of_rem_ms"] for r_ in dirs)
    for r_ in dirs:
        r_["share_of_rem_ms"] /= tot
    # Is the remainder along each direction a quadratic function of the score?
    # Regress the standardized remainder on z = h* / sd(h*) (h* = J^{-1} Delta, the
    # first-order term; coordinates u, v, a, t) and on all products z_i z_j.
    zs = hstar / np.sqrt(np.diag(Jinv))
    names = ["u", "v", "a", "t"]
    quad = [(i, j) for i in range(4) for j in range(i, 4)]
    X1 = np.column_stack([np.ones(m), zs])
    X2 = np.column_stack([X1] + [zs[:, i] * zs[:, j] for i, j in quad])
    for kk_, r_ in enumerate(dirs):
        u = U[:, kk_] * np.sign(U[np.argmax(np.abs(U[:, kk_])), kk_])
        y = remv @ u / math.sqrt(mu[kk_])
        out = {}
        for lab, X in (("lin", X1), ("quad", X2)):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            out[lab] = 1 - res.var() / y.var()
        r_["R2_rem_on_linear_z"] = float(out["lin"])
        r_["R2_rem_on_linear_quadratic_z"] = float(out["quad"])
        beta, *_ = np.linalg.lstsq(X2, y, rcond=None)
        qb = sorted(((abs(b), f"z_{names[i]}*z_{names[j]}", float(b))
                     for b, (i, j) in zip(beta[5:], quad)), reverse=True)[:3]
        r_["top_quadratic_terms"] = json.dumps([(nm, round(b, 3)) for _, nm, b in qb])
    per_rep = pd.DataFrame(dict(rep=[r["rep"] for r in rows], **{f"Q_{kk}": v for kk, v in Q.items()},
                                norm_T_minus_S=rn, refit=[r["refit"] for r in rows],
                                any_bound=[r["at_bound"].any() for r in rows],
                                kappa_hat=khat))
    per_rep.insert(0, "setting", tagg)
    return vecs, norms, [rem], pd_rows, per_rep, dirs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--nu0", type=float, nargs="+", default=[0.5])
    ap.add_argument("--G", type=int, nargs="+", default=[10, 8, 6])
    ap.add_argument("--law", default="gauss")
    ap.add_argument("--out", default="audit/out/followup")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    V, N, R, P, PR, DI = [], [], [], [], [], []
    for nu0 in a.nu0:
        for G in a.G:
            v, nn, r, p, pr, di = analyse(a.d, nu0, G, a.law)
            V += v; N += nn; R += r; P += p; PR.append(pr); DI += di
    sfx = f"d{a.d}_{a.law}"
    pd.DataFrame(V).to_csv(os.path.join(a.out, f"task1_vectors_{sfx}.csv"), index=False)
    pd.DataFrame(N).to_csv(os.path.join(a.out, f"task1_norms_coverage_{sfx}.csv"), index=False)
    pd.DataFrame(R).to_csv(os.path.join(a.out, f"task1_remainder_{sfx}.csv"), index=False)
    pd.DataFrame(P).to_csv(os.path.join(a.out, f"task1_paired_{sfx}.csv"), index=False)
    pd.concat(PR).to_csv(os.path.join(a.out, f"task1_per_replicate_{sfx}.csv"), index=False)
    pd.DataFrame(DI).to_csv(os.path.join(a.out, f"task1_directions_{sfx}.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    nn = pd.DataFrame(N)
    print(nn[["nu0", "n", "statistic", "mean", "q50", "q90", "q95", "q99", "coverage"]].round(3).to_string(index=False))
    pp = pd.DataFrame(P)
    print(pp[["nu0", "n", "from_", "to", "cov_from", "cov_to", "diff", "paired_se", "discordant"]].round(4).to_string(index=False))
    print(pd.DataFrame(R).T.to_string())
    print(pd.DataFrame(DI)[["nu0", "n", "eig_index", "sd_direction", "loading", "sd_linear_std", "mean_rem_std", "rms_rem_std", "share_of_rem_ms", "mean_err_std", "sd_err_std", "skew_err_std", "cover_1d"]].round(3).to_string(index=False))
    print(pd.DataFrame(V)[["nu0", "n", "vector", "mean", "cov_eig", "comp_skew", "comp_exkurt"]].to_string(index=False))


if __name__ == "__main__":
    main()
