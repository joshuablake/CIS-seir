from config import *
from covid19_seir.models import prevalence_model as model
import numpy as np
import pandas as pd

list_posteriors = []
failed_sims = []
posterior_regions = []
posterior_chains = []
for r in REGIONS:
    for c in range(NUM_CHAINS_PER_REGION):
        f = os.path.join(RESULTS_DIR, f"posterior_{r}_{c}.npy")
        # Read matrix of iteration x parameter, entries are values
        list_posteriors.append(np.load(f, allow_pickle=True))
posteriors = np.stack(list_posteriors)
del list_posteriors

# Put all successful chains into one dataframe
# Columns are: region, chain, iteration, parameter, value
priors = BASE_PRIORS.copy()
priors["pi"] = PI_PRIORS[REGIONS[0]]
log_posterior = PROBABILISTIC_MODEL_CLASS(forward_model, None, None, priors)
thinned_params = range(0, posteriors.shape[1], 100)
df_posterior = pd.DataFrame(
    posteriors[:,thinned_params,:].flatten(),
    index=pd.MultiIndex.from_product(
        [
            REGIONS,
            range(NUM_CHAINS_PER_REGION),
            thinned_params,
            log_posterior.get_param_names()
        ],
        names=["region", "chain", "iteration", "parameter"]),
    columns=["value"]
).reset_index()

# Save to csv
df_posterior.to_csv(os.path.join(RESULTS_DIR, "params.csv"), index=False)

# Posterior predictive incidence and prevalence
thinned_for_predictive = range(500_000, posteriors.shape[1], 1000)
predicted_incidence = np.empty(
    (posteriors.shape[0], len(thinned_for_predictive), N_DAYS, N_STRATA)
)
predicted_prevalence = np.empty_like(predicted_incidence)
final_state = np.empty(
    (posteriors.shape[0], len(thinned_for_predictive), 6 + model.P_STATES, N_STRATA)
)
for c in range(posteriors.shape[0]):
    for i, s in enumerate(thinned_for_predictive):
        result = log_posterior.simulate(posteriors[c, s])[0]
        predicted_incidence[c, i] = result[0]
        predicted_prevalence[c, i] = result[1]
        final_state[c, i] = result[2]
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
            REGIONS,
            range(NUM_CHAINS_PER_REGION),
            thinned_for_predictive,
            dates,
            STRATA_NAMES,
        ],
        names=["region", "chain", "iteration", "day", "age"]
    )
).reset_index()
# Save to csv
posterior_predictive.to_csv(os.path.join(RESULTS_DIR, "predictive.csv"), index=False)

final_state = pd.DataFrame(
    {
        "occupancy": final_state.flatten()
    },
    index=pd.MultiIndex.from_product(
        [
            REGIONS,
            range(NUM_CHAINS_PER_REGION),
            thinned_for_predictive,
            model.SEIR_state._fields,
            STRATA_NAMES,
        ],
        names=["region", "chain", "iteration", "state", "age"]
    )
).reset_index()
# Save to csv
final_state.to_csv(os.path.join(RESULTS_DIR, "final_state.csv"), index=False)