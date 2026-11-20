"""Interim summary from thinned MCMC checkpoints (small files only).

Usage: python checkpoint_report.py [cutoff YYYY-MM-DD] [burn_in]
Prints per-chain progress / acceptance, and per-region posterior summaries
(psi, i+, early beta, child susceptibility) with R-hat / ESS (arviz), using
draws after burn-in (default 1e6, or half the completed iterations if fewer).
Writes reports/checkpoint_<cutoff>.csv. Light enough for a short interactive
or batch job; reads ~36 files of a few hundred kB each.
"""
import os, sys
from config import *
import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

cutoff = sys.argv[1] if len(sys.argv) > 1 else FULL_END_DATE.isoformat()
burn_in_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
ckdir = os.environ.get("CKPT_DIR", os.path.join(RESULTS_DIR, "checkpoints"))
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(out_dir, exist_ok=True)


def transform(names, samples):
    """Return dict of natural-scale quantities of interest."""
    names = list(names)
    q = {}
    q["psir"] = samples[:, names.index("psir")]
    q["i0"] = expit(samples[:, names.index("i0")])
    q["child_susc"] = np.exp(samples[:, names.index("matrix_modifiers")])
    b = [i for i, n in enumerate(names) if n.startswith("beta[")]
    # beta block: [log RW sd, non-centred innovations z_1..]; beta_t = exp(sd * cumsum(z))
    sd = np.exp(samples[:, b[0]])
    q["rw_sd"] = sd
    walk = np.exp(np.cumsum(samples[:, b[1:]] * sd[:, None], axis=1))
    q["beta_wk1"] = walk[:, 0]   # relative to week 0 (2020-08-31)
    q["beta_wk2"] = walk[:, 1]
    q["beta_wk4"] = walk[:, 3]
    return q


rows, chain_rows = [], []
for region in REGIONS:
    chains = []
    for c in range(NUM_CHAINS_PER_REGION):
        f = os.path.join(ckdir, f"ckpt_{region}_{cutoff}_{c}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        chains.append(d)
        chain_rows.append(dict(region=region, chain=c, last_iter=int(d["last_iteration"]),
                               acceptance=float(d["acceptance"]),
                               log_target_last=float(d["log_target"][-1]),
                               psir_last=float(d["samples"][-1, list(d["param_names"]).index("psir")])))
    if len(chains) < 2:
        continue
    n_common = min(len(d["iteration"]) for d in chains)
    last = min(int(d["iteration"][n_common - 1]) for d in chains)
    burn = burn_in_arg if last > 2 * burn_in_arg else last // 2
    keep = chains[0]["iteration"][:n_common] >= burn
    qs = [transform(d["param_names"], d["samples"][:n_common][keep]) for d in chains]
    lt = np.stack([d["log_target"][:n_common][keep] for d in chains])
    for name in list(qs[0]) + ["log_target"]:
        arr = lt if name == "log_target" else np.stack([q[name] for q in qs])
        flat = arr.ravel()
        rows.append(dict(region=region, quantity=name, n_chains=arr.shape[0],
                         draws_per_chain=arr.shape[1], burn_in=burn,
                         median=np.median(flat), lo=np.quantile(flat, 0.025),
                         hi=np.quantile(flat, 0.975),
                         rhat=float(az.rhat(arr)) if arr.shape[1] >= 4 else np.nan,
                         ess_bulk=float(az.ess(arr, method="bulk")) if arr.shape[1] >= 4 else np.nan))

summ = pd.DataFrame(rows)
chains_df = pd.DataFrame(chain_rows)
summ.to_csv(os.path.join(out_dir, f"checkpoint_{cutoff}.csv"), index=False)
chains_df.to_csv(os.path.join(out_dir, f"checkpoint_chains_{cutoff}.csv"), index=False)
pd.set_option("display.width", 200)
print(chains_df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(summ.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
