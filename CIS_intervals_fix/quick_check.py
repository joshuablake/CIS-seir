"""Quick check: effect of the omega_0 fix on the ORIGINAL final-cut-off posterior.

For posterior draws from the original runs (thinned params CSVs, burn-in 1e6,
every 20000th iteration as in the paper figures), simulate with
  old      : original (buggy) initial P state
  new_same : corrected initial state, same parameters
  new_resc : corrected initial state, i+ scaled by F = (psi+w0)/(psi+w1) so
             day-0 PCR prevalence matches the old model exactly
             (p_init is linear in i+, so this match is exact)
and, on a subsample, a matched fit:
  new_fit  : corrected initial state with (i+ factor, psi shift) chosen to
             best match the old model's log prevalence over days 0-27
             (betas held fixed), to estimate how far a refit must move.

Outputs CSVs in reports/quick_check/. Run as a Slurm job (SLURM_quick_check).
"""
import os, sys
from config import *
from export_summary_stats import load_poststrat_table, normalise_region_names
from covid19_seir.models import prevalence_model as pm
from covid19_seir.models.prevalence_model import P_TRANSITIONS, P_STATES, KAPPA_Y
from scipy.special import expit, logit
from scipy.optimize import minimize
import numpy as np
import pandas as pd

OLD_RESULTS_DIR = "/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals_sd0.1"  # read only
CUTOFF = FULL_END_DATE
BURN_IN, THIN = 1_000_000, 20_000
N_FIT_DRAWS = int(os.environ.get("N_FIT_DRAWS", 40))
FIT_DAYS = 28
OUT_DIR = os.environ.get("QC_OUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "quick_check"))
MAX_DRAWS = int(os.environ.get("QC_MAX_DRAWS", 0))  # smoke tests only
QC_REGIONS = os.environ.get("QC_REGIONS", ",".join(REGIONS)).split(",")
os.makedirs(OUT_DIR, exist_ok=True)
AGE_ORDER = ["0–11", "12–16", "17–25", "26–50", "51–70", "71+"]
W0, W1 = -P_TRANSITIONS[0, 0], -P_TRANSITIONS[1, 1]

Cls = pm.SingleRegionStratifiedSEIRCompartmental
new_init = Cls._calculate_initial_prev_state


def old_init(self, e1_init, sigma, psir):
    p_init = np.empty((P_STATES, self.num_strata))
    p_init[0] = (sigma + psir) / (psir + -P_TRANSITIONS[1, 1]) * e1_init
    P_OUT = -np.diag(P_TRANSITIONS)
    P_IN = P_TRANSITIONS @ np.ones(P_STATES) + P_OUT
    for i in range(1, P_STATES):
        p_init[i] = (P_IN[i] / (psir + P_OUT[i])) * p_init[i - 1]
    return p_init


config = setup_experiment_config(CUTOFF)
dates = pd.to_datetime(config["dates"])
pop_weights = load_poststrat_table()
rng = np.random.default_rng(20261120)

traj_rows, draw_rows, fit_rows = [], [], []

for region in QC_REGIONS:
    print(f"Region {region}", flush=True)
    priors = config["BASE_PRIORS"].copy()
    priors["pi"] = PI_PRIORS[region]
    lp = config["PROBABILISTIC_MODEL_CLASS"](config["forward_model"], None, None, priors)
    names = lp.get_param_names()
    n_par = len(names)
    i_i0, i_psi = names.index("i0"), names.index("psir")

    rname = normalise_region_names(region)
    w = pop_weights[pop_weights.region == rname].set_index("age").loc[AGE_ORDER, "pop"].values
    w = w / w.sum()

    df = pd.read_csv(os.path.join(OLD_RESULTS_DIR, f"params_{region}_{CUTOFF.isoformat()}.csv"))
    df = df[(df.iteration >= BURN_IN) & (df.iteration % THIN == 0)]
    draws = []
    for (chain, it), g in df.groupby(["chain", "iteration"], sort=True):
        assert list(g.parameter) == names
        draws.append((chain, it, g.value.values))
    if MAX_DRAWS:
        draws = draws[:: max(1, len(draws) // MAX_DRAWS)][:MAX_DRAWS]
    print(f"  {len(draws)} draws", flush=True)

    def sim(theta, init):
        Cls._calculate_initial_prev_state = init
        try:
            inc, prev, _ = lp.simulate(theta)[0]
        finally:
            Cls._calculate_initial_prev_state = new_init
        return inc @ w, prev @ w  # population-weighted regional proportions

    fit_idx = set(rng.choice(len(draws), size=min(N_FIT_DRAWS, len(draws)), replace=False))
    for k, (chain, it, theta) in enumerate(draws):
        psi = theta[i_psi]
        F = (psi + W0) / (psi + W1)
        res = {"old": sim(theta, old_init), "new_same": sim(theta, new_init)}
        th = theta.copy()
        th[i_i0] = logit(expit(theta[i_i0]) * F)
        res["new_resc"] = sim(th, new_init)

        if k in fit_idx:
            target = np.log(res["old"][1][:FIT_DAYS])

            def loss(x):
                t = theta.copy()
                t[i_i0] = logit(min(expit(theta[i_i0]) * np.exp(x[0]), 0.5))
                t[i_psi] = psi + x[1]
                try:
                    p = sim(t, new_init)[1][:FIT_DAYS]
                except (ValueError, AssertionError):
                    return 1e6
                return np.sum((np.log(p) - target) ** 2)

            opt = minimize(loss, x0=[np.log(F) * 0.7, -0.03], method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 400})
            t = theta.copy()
            t[i_i0] = logit(expit(theta[i_i0]) * np.exp(opt.x[0]))
            t[i_psi] = psi + opt.x[1]
            res["new_fit"] = sim(t, new_init)
            fit_rows.append(dict(region=region, chain=chain, iteration=it, psi_old=psi,
                                 F=F, i0_factor=np.exp(opt.x[0]), dpsi=opt.x[1],
                                 rms_log_err=np.sqrt(opt.fun / FIT_DAYS), converged=opt.success))

        # day-level rows for days 0-90 (to end of Nov), plus per-draw summaries
        for var, (inc, prev) in res.items():
            for d in range(91):
                traj_rows.append((region, var, chain, it, d, inc[d], prev[d]))
        o_inc, o_prev = res["old"]
        draw_rows.append(dict(
            region=region, chain=chain, iteration=it, psi=psi, F=F,
            prev_excess_d14=1 - res["new_same"][1][14] / o_prev[14],
            prev_excess_d28=1 - res["new_same"][1][28] / o_prev[28],
            inc_ratio_resc_d0=res["new_resc"][0][0] / o_inc[0],
            prev_ratio_resc_d14=res["new_resc"][1][14] / o_prev[14],
            prev_ratio_resc_d28=res["new_resc"][1][28] / o_prev[28],
        ))
    print(f"  done ({len(fit_rows)} fits so far)", flush=True)

traj = pd.DataFrame(traj_rows, columns=["region", "variant", "chain", "iteration", "day",
                                        "incidence", "prevalence"])
traj["date"] = dates[traj.day].date
traj.to_csv(os.path.join(OUT_DIR, "trajectories_draws.csv.gz"), index=False, compression="gzip")
pd.DataFrame(draw_rows).to_csv(os.path.join(OUT_DIR, "per_draw_summary.csv"), index=False)
pd.DataFrame(fit_rows).to_csv(os.path.join(OUT_DIR, "matched_fit.csv"), index=False)

# England aggregate (regional population weights) and summaries
reg_pop = pop_weights.groupby("region")["pop"].sum()
traj["rpop"] = traj.region.map(lambda r: reg_pop[normalise_region_names(r)])


def summarise(d, by):
    q = d.groupby(by)[["incidence", "prevalence"]].quantile([0.025, 0.5, 0.975]).unstack()
    q.columns = [f"{v}_{p}" for v, p in q.columns]
    return q.reset_index()


summ_reg = summarise(traj[traj.variant != "new_fit"], ["region", "variant", "date"])
eng = traj[traj.variant != "new_fit"].copy()
# England: for each (variant, day), combine regions draw-wise by index within region
eng["k"] = eng.groupby(["region", "variant", "day"]).cumcount()
for v in ["incidence", "prevalence"]:
    eng[v] = eng[v] * eng.rpop
eng = eng.groupby(["variant", "date", "k"])[["incidence", "prevalence", "rpop"]].sum().reset_index()
eng["incidence"] /= eng.rpop
eng["prevalence"] /= eng.rpop
summ_eng = summarise(eng, ["variant", "date"])
summ_eng.insert(0, "region", "England")
pd.concat([summ_eng, summ_reg]).to_csv(os.path.join(OUT_DIR, "trajectory_summary.csv"), index=False)
print("Quick check complete", flush=True)
