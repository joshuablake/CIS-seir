from config import *
from covid19_seir.models import prevalence_model as model
import numpy as np
import pandas as pd
import sys

# --- EXPERIMENT DEFINITION (modified for region/cutoff combinations) ---
JOB_NUM = int(sys.argv[1])

# Map job number to (region, cutoff) combinations - all chains per combination
region_idx = JOB_NUM % num_regions  
cutoff_idx = (JOB_NUM // num_regions) % num_cutoffs

# Select the parameters for this specific job run
run_cutoff_date = cutoff_dates[cutoff_idx]
run_region = REGIONS[region_idx]

print(f"Processing job {JOB_NUM}:")
print(f"  Cut-off Date: {run_cutoff_date.isoformat()}")
print(f"  Region: {run_region}", flush=True)

# --- SETUP EXPERIMENT USING THE REFACTORED CONFIG ---
config = setup_experiment_config(run_cutoff_date)
dates = config["dates"]
N_DAYS = config["N_DAYS"]
N_STRATA = config["N_STRATA"]
forward_model = config["forward_model"]
PROBABILISTIC_MODEL_CLASS = config["PROBABILISTIC_MODEL_CLASS"]
BASE_PRIORS = config["BASE_PRIORS"]

# --- LOAD POSTERIOR FILES FOR THIS REGION/CUTOFF ---
print("Loading files...", flush=True)
list_posteriors = []
posterior_chains = []

for c in range(NUM_CHAINS_PER_REGION):
    f = os.path.join(RESULTS_DIR, f"posterior_{run_region}_{run_cutoff_date.isoformat()}_{c}.npy.npz")
    if os.path.exists(f):
        list_posteriors.append(np.load(f, allow_pickle=True)['arr_0'])
        posterior_chains.append(c)
        print(f"Loaded chain {c}", flush=True)
    else:
        print(f"Missing: {f}", flush=True)

if len(posterior_chains) < 2:
    raise ValueError(f"Fewer than two files found for {run_region}, {run_cutoff_date}")

print("Loaded all chains")

# --- CREATE PARAMETER DATAFRAME ---
print("Creating paramter dataframe...", flush=True)
priors = BASE_PRIORS.copy()
priors["pi"] = PI_PRIORS[run_region]
log_posterior = PROBABILISTIC_MODEL_CLASS(forward_model, None, None, priors)
param_names = log_posterior.get_param_names()

param_data = []
thinned_params = range(0, min(p.shape[0] for p in list_posteriors), 5000)

for chain_idx, posterior_array in enumerate(list_posteriors):
    chain = posterior_chains[chain_idx]
    max_iter = posterior_array.shape[0]
    valid_thinned = [s for s in thinned_params if s < max_iter]
    
    for iteration in valid_thinned:
        for p, param_name in enumerate(param_names):
            param_data.append({
                'region': run_region,
                'chain': chain,
                'cutoff_date': run_cutoff_date,
                'iteration': iteration,
                'parameter': param_name,
                'value': posterior_array[iteration, p]
            })

df_posterior = pd.DataFrame(param_data)

# Save region/cutoff-specific parameter file
params_filename = f"params_{run_region}_{run_cutoff_date.isoformat()}.csv"
df_posterior.to_csv(os.path.join(RESULTS_DIR, params_filename), index=False)
print(f"Saved parameters to {params_filename}")

# --- POSTERIOR PREDICTIVE SIMULATION ---
print("Starting posterior predictive simulation...")

thinned_for_predictive = thinned_params
print(f"Using {len(thinned_for_predictive)} samples per chain for prediction", flush=True)

pop = pd.read_csv(os.path.join(FILE_BASE, "data/contact_m/population.csv"))
STRATA_NAMES = list(pop["AgeGroup"])
dates = pd.date_range(START_DATE, run_cutoff_date, freq="1D")

predictive_data = []
final_state_data = []

for chain_idx, posterior_array in enumerate(list_posteriors):
    chain = posterior_chains[chain_idx]
    max_iter = posterior_array.shape[0]
    valid_thinned = [s for s in thinned_for_predictive if s < max_iter]
    
    print(f"Simulating chain {chain} with {len(valid_thinned)} samples", flush=True)
    
    for i, iteration in enumerate(valid_thinned):
        result = log_posterior.simulate(posterior_array[iteration], include_new_pcr_pos=True)[0]
        predicted_incidence = result[0]
        predicted_prevalence = result[1]
        final_state = result[2]
        predicted_new_pos = result[3]
        
        # Store predictive data
        for d, day in enumerate(dates):
            for a, age in enumerate(STRATA_NAMES):
                predictive_data.append({
                    'region': run_region,
                    'chain': chain,
                    'cutoff_date': run_cutoff_date,
                    'iteration': iteration,
                    'day': day,
                    'age': age,
                    'incidence': predicted_incidence[d, a],
                    'prevalence': predicted_prevalence[d, a],
                    'new_pcr_pos': predicted_new_pos[d, a]
                })
        
        # Store final state data
        for s, state in enumerate(model.SEIR_state._fields):
            for a, age in enumerate(STRATA_NAMES):
                final_state_data.append({
                    'region': run_region,
                    'chain': chain,
                    'cutoff_date': run_cutoff_date,
                    'iteration': iteration,
                    'state': state,
                    'age': age,
                    'occupancy': final_state[s, a]
                })

# Save region/cutoff-specific output files
posterior_predictive = pd.DataFrame(predictive_data)
predictive_filename = f"predictive_{run_region}_{run_cutoff_date.isoformat()}.csv"
posterior_predictive.to_csv(os.path.join(RESULTS_DIR, predictive_filename), index=False)
print(f"Saved predictive data to {predictive_filename}", flush=True)

final_state_df = pd.DataFrame(final_state_data)
final_state_filename = f"final_state_{run_region}_{run_cutoff_date.isoformat()}.csv"
final_state_df.to_csv(os.path.join(RESULTS_DIR, final_state_filename), index=False)
print(f"Saved final state data to {final_state_filename}", flush=True)
