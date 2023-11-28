from config import *
import numpy as np
import pandas as pd
import json

list_posteriors = []
failed_sims = []
posterior_regions = []
posterior_chains = []
for i in range(NUM_SIMS):
    f = os.path.join(RESULTS_DIR, f"posterior_{i}.npy")
    # Read matrix of iteration x parameter, entries are values
    list_posteriors.append(np.load(f, allow_pickle=True))
posteriors = np.stack(list_posteriors)
del list_posteriors

# Put all successful chains into one dataframe
# Columns are: region, chain, iteration, parameter, value
log_posterior = PROBABILISTIC_MODEL_CLASS(model, None, None, PRIORS)
thinned_params = range(0, posteriors.shape[1], 100)
df_posterior = pd.DataFrame(
    posteriors[:,thinned_params,:].flatten(),
    index=pd.MultiIndex.from_product(
        [
            range(NUM_SIMS),
            thinned_params,
            log_posterior.get_param_names()
        ],
        names=["sim", "iteration", "parameter"]),
    columns=["value"]
).reset_index()

# Save to csv
df_posterior.to_csv(os.path.join(RESULTS_DIR, "posteriors_combined.csv"), index=False)

# Posterior predictive incidence and prevalence
thinned_for_predictive = range(500_000, posteriors.shape[1], 100)
predicted_incidence = np.empty(
    (posteriors.shape[0], len(thinned_for_predictive), N_DAYS, N_STRATA)
)
predicted_prevalence = np.empty_like(predicted_incidence)
for c in range(posteriors.shape[0]):
    for i, s in enumerate(thinned_for_predictive):
        result = log_posterior.simulate(posteriors[c, s])[0]
        predicted_incidence[c, i] = result[0]
        predicted_prevalence[c, i] = result[1]
dates = pd.date_range(START_DATE, END_DATE, freq="1D")
pop = pd.read_csv(os.path.join(FILE_BASE, "data/contact_m/population.csv"))
STRATA_NAMES = list(pop["AgeGroup"])
posterior_predictive = pd.DataFrame(
    {
        "incidence": predicted_incidence.flatten(),
        "prevalence": predicted_prevalence.flatten(),
    },
    index=pd.MultiIndex.from_product(
        [
            range(NUM_SIMS),
            thinned_for_predictive,
            dates,
            STRATA_NAMES,
        ],
        names=["sim", "iteration", "day", "age"]
    )
).reset_index()
# Save to csv
posterior_predictive.to_csv(os.path.join(RESULTS_DIR, "posteriors_predictive.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "sim_params.json"), "r") as fp:
    true_vals =  np.array([
        log_posterior.param_value_dict_to_array(p)
        for p in json.load(fp)
    ]).flatten()
df_true_params = pd.DataFrame(
    true_vals,
    index=pd.MultiIndex.from_product(
        [range(NUM_SIMS), log_posterior.get_param_names()],
        names=["sim", "param"]
    ),
    columns=["true"]
).reset_index()
df_true_params.to_csv(os.path.join(RESULTS_DIR, "true_vals.csv"), index=False)