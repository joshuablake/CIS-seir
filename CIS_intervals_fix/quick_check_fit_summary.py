"""Summarise matched-fit (new_fit) vs old trajectories from quick_check output."""
import os, pandas as pd, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "quick_check")
t = pd.read_csv(os.path.join(D, "trajectories_draws.csv.gz"))
f = pd.read_csv(os.path.join(D, "matched_fit.csv"))[["region", "chain", "iteration"]]
t = t.merge(f, on=["region", "chain", "iteration"])
w = t.pivot_table(index=["region", "chain", "iteration", "day"], columns="variant",
                  values=["incidence", "prevalence"]).reset_index()
w.columns = ["_".join(c).strip("_") for c in w.columns]
for v in ["incidence", "prevalence"]:
    w[f"{v}_ratio"] = w[f"{v}_new_fit"] / w[f"{v}_old"]
    w[f"{v}_ratio_same"] = w[f"{v}_new_same"] / w[f"{v}_old"]
days = [0, 7, 14, 21, 28, 42, 56, 90]
r = w[w.day.isin(days)].groupby(["region", "day"])[["incidence_ratio", "prevalence_ratio"]].median().reset_index()
r.to_csv(os.path.join(D, "matched_fit_ratios_by_region.csv"), index=False)
s = r.groupby("day")[["incidence_ratio", "prevalence_ratio"]].agg(["min", "median", "max"]).round(3)
s.to_csv(os.path.join(D, "matched_fit_ratios_summary.csv"))
pd.set_option("display.width", 200)
print(s)
# cumulative September incidence (days 0-29) ratio
sep = w[w.day < 30].groupby(["region", "chain", "iteration"])[["incidence_new_fit", "incidence_old"]].sum()
sep = (sep.incidence_new_fit / sep.incidence_old).groupby("region").median().round(3)
sep.to_csv(os.path.join(D, "matched_fit_sept_cuminc_ratio.csv"))
print(sep)
