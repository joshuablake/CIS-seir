from config import *
from covid19_seir.models import logPDF_prev
from scipy import stats
import json
import numpy as np
import pandas as pd
rng = np.random.default_rng(1)

def simulate_betabinom_from_mean_theta(n, mean, theta, random_state=None):
    alpha = mean / theta
    beta = (1 - mean) / theta
    return stats.betabinom.rvs(n=n, a=alpha, b=beta, random_state=random_state)


# ODE SIMULATION
true_params = []
nni_out = np.empty((NUM_SIMS, N_DAYS, N_STRATA))
prev_out = np.empty((NUM_SIMS, N_DAYS, N_STRATA))
probabilistic_model = PROBABILISTIC_MODEL_CLASS(model, None, None, PRIORS)
theta = np.empty(NUM_SIMS)
beta_increments = list(1/50 * stats.norm.rvs(size=20, random_state=rng))
for i in range(NUM_SIMS):
    param_vals = {
        "pi": list(PRIORS["pi"].value_to_sample(
            susc_beta_params[:, 0] / (susc_beta_params[:, 0] + susc_beta_params[:, 1])
        )),
        "i0": PRIORS["i0"].value_to_sample(8.6 / (8.6 + 27400)),
        "matrix_modifiers": -0.4325,
        "psir": PRIORS["psir"].value_to_sample(0.048),
        "beta": list(np.insert(
            beta_increments,
            0,
            PRIORS["beta"].sd_param.value_to_sample(1/50)
        )),
        "theta": PRIORS["theta"].value_to_sample(2e-5),
    }
    true_params.append(param_vals)
    simulate_out = probabilistic_model.simulate(
        parameters=probabilistic_model.param_value_dict_to_array(param_vals),
        return_theta=True
    )
    result = simulate_out[0]
    nni_out[i] = result[0]
    prev_out[i] = result[1]
    theta[i] = simulate_out[1]

# SIMULATE OBSERVATIONS
tests_by_strata_per_day = NUM_DAILY_TESTS * pop["population"] / sum(pop["population"])
daily_tests_by_age = pd.Series(
    # Total tests multiplied by population proportions
    data=tests_by_strata_per_day.astype(int),
    name="num_tests",
)
daily_tests_by_age.index = pd.Index(STRATA_NAMES, name="age")
true_results = pd.DataFrame(
    data={"incidence": nni_out.flatten(), "prevalence": prev_out.flatten()},
    index=pd.MultiIndex.from_product([range(NUM_SIMS), dates, STRATA_NAMES], names=["sim", "date", "age"])
)\
    .merge(daily_tests_by_age, left_index=True, right_index=True)\
    .reset_index()\
    .merge(
        pd.DataFrame(
            data={"theta": theta},
            index=pd.Index(range(NUM_SIMS), name="sim")
        ),
        left_on="sim",
        right_index=True
    )
true_results["obs_positives"] = simulate_betabinom_from_mean_theta(
    n=true_results["num_tests"],
    mean=true_results["prevalence"],
    theta=true_results["theta"],
    random_state=rng
)
true_results["obs_prevalence"] = true_results["obs_positives"] / true_results["num_tests"]

# SAVE OUTPUTS
true_results.to_csv(os.path.join(RESULTS_DIR, "sim_output.csv"))
with open(os.path.join(RESULTS_DIR, "sim_params.json"), 'w') as fp:
    json.dump(true_params, fp)