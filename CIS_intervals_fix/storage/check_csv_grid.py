"""Check new posterior_as_csv output against the original run's CSVs for one region/cut-off.

Usage: python check_csv_grid.py <old_dir> <new_dir> <region> <cutoff>
- params: new must equal old exactly (same pipeline, every PARAM_THIN-th iteration incl. burn-in).
- predictive / final_state: the (chain, iteration) draws in the new file must be exactly those the
  R filter (iteration %% 20e3 == 0 & iteration >= 1e6) selects from the old file, with the same
  number of rows per draw. Values differ because the model was fixed.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from incidence_utils import load_csv_with_burnin

old, new, region, cutoff = sys.argv[1:5]
ok = True
po = pd.read_csv(os.path.join(old, f"params_{region}_{cutoff}.csv"))
pn = pd.read_csv(os.path.join(new, f"params_{region}_{cutoff}.csv"))
same = po.shape == pn.shape and po.equals(pn)
print(f"params: old {po.shape}, new {pn.shape}, identical={same}"); ok &= same
for kind in ["predictive", "final_state"]:
    o = pd.read_csv(os.path.join(old, f"{kind}_{region}_{cutoff}.csv"), usecols=["chain", "iteration"])
    n = pd.read_csv(os.path.join(new, f"{kind}_{region}_{cutoff}.csv"), usecols=["chain", "iteration"])
    o_r = o[(o.iteration % 20000 == 0) & (o.iteration >= 1_000_000)]  # the R filter
    ko = o_r.value_counts().sort_index(); kn = n.value_counts().sort_index()
    m = ko.index.equals(kn.index) and (ko.values == kn.values).all()
    print(f"{kind}: old {len(o)} rows / {o.drop_duplicates().shape[0]} draws; R-filtered "
          f"{len(o_r)} rows / {len(ko)} draws; new {len(n)} rows / {len(kn)} draws; "
          f"draws per chain new {n.drop_duplicates().groupby('chain').size().to_dict()}; match={m}")
    ok &= m
# Python loader (incidence_utils) applies the same selection to the old predictive file
lo = load_csv_with_burnin(os.path.join(old, f"predictive_{region}_{cutoff}.csv"))
ln = load_csv_with_burnin(os.path.join(new, f"predictive_{region}_{cutoff}.csv"))
m = lo[["chain", "iteration"]].value_counts().sort_index().equals(ln[["chain", "iteration"]].value_counts().sort_index())
print(f"incidence_utils loader: old {len(lo)} rows, new {len(ln)} rows, same draws={m}"); ok &= m
print("ALL OK" if ok else "MISMATCH")
