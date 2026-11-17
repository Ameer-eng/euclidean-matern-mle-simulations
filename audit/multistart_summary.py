"""AUDIT: summarise audit/out/ms_local/multistart_*.csv (gap of each start below the best)."""
import glob, json, sys
import numpy as np, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else "audit/out/ms_local"
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{src}/multistart_*.csv"))])
hi = {tuple(x) for x in json.load(open("audit/out/hi_pg_cases.json"))}
df["sub"] = df.group
isC2 = df.apply(lambda r: (r.d, r.nu0, r.G, r.law, r.rep) in hi, axis=1)
df.loc[(df.group == "C") & isC2, "sub"] = "C2"
df.loc[(df.group == "C") & ~isC2, "sub"] = "C1"
key = ["sub", "d", "nu0", "G", "law", "rep"]
df["gap"] = df.groupby(key).loglik.transform("max") - df.loglik
df["kind"] = df.start.map(lambda s: "truth" if s == "truth" else ("far" if s.startswith("S") else s))
print("data sets per subgroup:", df.drop_duplicates(key).groupby("sub").size().to_dict())
print("data_match:", df.groupby("sub").data_match.mean().round(3).to_dict())
print("all starts in Theta & xbox:", bool(df.start_in_Theta.all() and df.start_in_xbox.all()))
print("fitted values in Theta (tol 1e-12):",
      bool(((df.nu <= 4 + 1e-12) & (df.nu >= 0.1 - 1e-12) & (df.logt >= np.log(1e-3) - 1e-12)).all()))
tab = df.pivot_table(index="sub", columns="kind", values="gap", aggfunc="max")
print(tab.map(lambda v: f"{v:.1e}").to_string())
cnt = df[df.gap > 1e-6].groupby(["sub", "kind"]).size()
print("data sets with a start more than 1e-6 below the best:\n", cnt.to_string())
c2 = df[(df["sub"] == "C2") & (df.start == "truth")]
print("C2 production-start gaps:", np.sort(c2.gap.values).round(4))
print("max |truth restart - stored ll| (data matched):",
      (df[(df.start == "truth") & df.data_match].loglik - df[(df.start == "truth") & df.data_match].ll_stored).abs().max())
print("nfev by start (median):", df.groupby("kind").nfev.median().to_dict())
