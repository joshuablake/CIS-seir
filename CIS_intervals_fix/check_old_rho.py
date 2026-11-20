"""Scan the ORIGINAL runs' thinned params CSVs (read only) for very low log rho.

log rho (parameter 'theta', log scale) < -25.3 (rho < 1e-11) is where scipy's
betabinom.logpmf becomes unreliable; < -27.6 (rho < 1e-12) is clearly garbage.
CSVs are thinned every 5000 iterations, so brief excursions can be missed, but
a chain trapped there (as London chain 2 in the pilot) cannot.
"""
import os
from config import REGIONS, cutoff_dates
import numpy as np
import pandas as pd

OLD = "/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals_sd0.1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "old_rho_check.csv")
BURN = 1_000_000
rows = []
for cut in cutoff_dates:
    for region in REGIONS:
        f = os.path.join(OLD, f"params_{region}_{cut.isoformat()}.csv")
        if not os.path.exists(f):
            print("missing", f, flush=True); continue
        parts = []
        for ch in pd.read_csv(f, usecols=["chain", "iteration", "parameter", "value"], chunksize=2_000_000):
            parts.append(ch[ch.parameter == "theta"])
        t = pd.concat(parts)
        for chain, g in t.groupby("chain"):
            v = g.value.values; it = g.iteration.values; post = v[it >= BURN]
            rows.append(dict(cutoff=cut, region=region, chain=chain, n=len(v),
                             min_all=v.min(), iter_at_min=int(it[v.argmin()]),
                             q025_post=np.quantile(post, .025) if len(post) else np.nan,
                             median_post=np.median(post) if len(post) else np.nan,
                             frac_lt_m20=np.mean(v < -20), frac_lt_m25_3=np.mean(v < -25.3),
                             frac_lt_m27_6=np.mean(v < -27.6),
                             first_iter_lt_m25_3=int(it[v < -25.3].min()) if (v < -25.3).any() else -1))
        print(cut, region, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
pd.set_option("display.width", 220)
print(f"{len(df)} chains scanned")
bad = df[df.frac_lt_m25_3 > 0]
print(f"chains ever below -25.3: {len(bad)}")
print(bad.to_string(index=False))
print("chains ever below -20:", (df.frac_lt_m20 > 0).sum())
print(df[["min_all", "q025_post", "median_post"]].describe().round(2))
print("overall min by region:"); print(df.groupby("region").min_all.min().round(2))
