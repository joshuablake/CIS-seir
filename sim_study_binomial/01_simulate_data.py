from config import *
from covid19_seir.models import logPDF_prev
from scipy import stats
import json
import numpy as np
import pandas as pd
rng = np.random.default_rng(1)

# ODE SIMULATION
true_params = []
nni_out = np.empty((NUM_SIMS, N_DAYS, N_STRATA))
prev_out = np.empty((NUM_SIMS, N_DAYS, N_STRATA))
probabilistic_model = PROBABILISTIC_MODEL_CLASS(model, None, None, PRIORS)
for i in range(NUM_SIMS):
    param_vals = {k: v.draw_from_prior(random_state=rng).tolist() for k, v in PRIORS.items()}
    true_params.append(param_vals)
    result = probabilistic_model.simulate(
        parameters=probabilistic_model.param_value_dict_to_array(param_vals)
    )
    nni_out[i] = result[0]
    prev_out[i] = result[1]

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
    .reset_index()
true_results["obs_positives"] = stats.binom.rvs(
    n=true_results["num_tests"],
    p=true_results["prevalence"],
    random_state=rng
)
true_results["obs_prevalence"] = true_results["obs_positives"] / true_results["num_tests"]

# SAVE OUTPUTS
true_results.to_csv(os.path.join(RESULTS_DIR, "sim_output.csv"))
with open(os.path.join(RESULTS_DIR, "sim_params.json"), 'w') as fp:
    json.dump(true_params, fp)