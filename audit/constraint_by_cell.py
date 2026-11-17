"""AUDIT (Q3): per (d, nu0, n, law) counts of original-constraint violations and of
boundary cases (tolerance 1e-8 as in fit.py), from audit/out/fits_audit.pkl."""
import math
import numpy as np, pandas as pd
f = pd.read_pickle("audit/out/fits_audit.pkl")
lt_lo, lt_hi = math.log(1e-3), math.log(10.0)
la_lo, la_hi = -2 * math.log(2.0), -2 * math.log(0.02)
ab = np.stack(f.at_bound.values)
f = f.assign(
    eta_viol=np.maximum(np.maximum(-15 - f.eta, f.eta - 40), 0),
    nu_viol=np.maximum(np.maximum(0.1 - f.nu, f.nu - 4), 0),
    loga_viol=np.maximum(np.maximum(la_lo - f.loga, f.loga - la_hi), 0),
    logt_viol=np.maximum(np.maximum(lt_lo - f.logt, f.logt - lt_hi), 0),
    eta_margin=np.minimum(f.eta + 15, 40 - f.eta),
    b_eta=ab[:, 0], b_nu=ab[:, 1], b_a=ab[:, 2], b_t=ab[:, 3], b_any=ab.any(1))
g = f.groupby(["d", "nu0", "n", "law"]).agg(
    nfit=("eta", "size"), eta_violations=("eta_viol", lambda x: int((x > 0).sum())),
    max_eta_viol=("eta_viol", "max"), min_eta_margin=("eta_margin", "min"),
    max_nu_viol=("nu_viol", "max"), max_loga_viol=("loga_viol", "max"), max_logt_viol=("logt_viol", "max"),
    at_eta=("b_eta", "sum"), at_nu=("b_nu", "sum"), at_a=("b_a", "sum"), at_t=("b_t", "sum"),
    any_bound_pct=("b_any", lambda x: 100 * x.mean()))
g.to_csv("audit/out/constraint_by_cell.csv")
print("cells:", len(g), " total eta violations:", int(g.eta_violations.sum()),
      " max violation sizes (eta, nu, log a, log t):", g.max_eta_viol.max(), g.max_nu_viol.max(),
      g.max_loga_viol.max(), g.max_logt_viol.max(), " min eta margin:", round(g.min_eta_margin.min(), 3))
