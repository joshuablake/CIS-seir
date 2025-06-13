# file: run_MCMC.py
import os
from config import setup_experiment_config, START_DATE, REGIONS, FILE_BASE
from covid19_seir.mcmc.single_block_adapt_mcmc import SingleBlockAMGS
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import sys

# --- NEW: EXPERIMENT DEFINITION ---
JOB_NUM = int(sys.argv[1])
np.random.seed(JOB_NUM * 37)

# Define experiment parameters
FULL_END_DATE = date(2021, 1, 24)
NUM_CHAINS_PER_REGION = 4
RESULTS_DIR = "/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Generate two-weekly cut-off dates
cutoff_dates = pd.date_range(START_DATE + timedelta(days=14) - timedelta(days=1), FULL_END_DATE, freq="14D").date

# --- NEW: SLURM ID MAPPING ---
# Map the single SLURM_ARRAY_TASK_ID to the (cutoff, region, chain) tuple
num_regions = len(REGIONS)
num_cutoffs = len(cutoff_dates)

chain_idx = JOB_NUM % NUM_CHAINS_PER_REGION
region_idx = (JOB_NUM // NUM_CHAINS_PER_REGION) % num_regions
cutoff_idx = (JOB_NUM // (NUM_CHAINS_PER_REGION * num_regions))

# Select the parameters for this specific job run
run_cutoff_date = cutoff_dates[cutoff_idx]
run_region = REGIONS[region_idx]
run_chain = chain_idx

print(f"Running job {JOB_NUM}:")
print(f"  Cut-off Date: {run_cutoff_date.isoformat()}")
print(f"  Region: {run_region}")
print(f"  Chain: {run_chain}")

# --- SETUP EXPERIMENT USING THE REFACTORED CONFIG ---
config = setup_experiment_config(run_cutoff_date)
# Unpack config variables for local use
dates = config["dates"]
N_DAYS = config["N_DAYS"]
N_STRATA = config["N_STRATA"]
forward_model = config["forward_model"]
PROBABILISTIC_MODEL_CLASS = config["PROBABILISTIC_MODEL_CLASS"]
BASE_PRIORS = config["BASE_PRIORS"]
PI_PRIORS = config["PI_PRIORS"]

# --- DATA LOADING AND FILTERING ---
# Prevalence data - load once, then filter for this run's specific dates
all_data_in = pd.read_csv(
    os.path.join(
        FILE_BASE,
        "SRS_extracts/20230829_STATS18115/modelling_data.csv"
    ),
    parse_dates=["date"]
)
data_shape_for_model = (N_DAYS, N_STRATA)
data_in = all_data_in[(all_data_in["region"] == run_region) & (all_data_in["date"].dt.date.isin(dates))]
array_num_tested = data_in["num_tests"].values.reshape(data_shape_for_model)
array_num_positive = data_in["obs_positives"].values.reshape(data_shape_for_model)

# Setup priors
priors = BASE_PRIORS.copy()
priors["pi"] = PI_PRIORS[run_region]

# Setup inference
start_dict = {
    k: v.draw_from_prior() for k, v in priors.items()
}
log_posterior = PROBABILISTIC_MODEL_CLASS(forward_model, array_num_positive, array_num_tested, priors)
start_dict["psir"] = np.random.normal(loc=0.056, scale=0.01) # Using numpy directly
start = log_posterior.param_value_dict_to_array(start_dict)

# Do inference
mcmc = SingleBlockAMGS(log_posterior, start, iterations=10_000_000)
start_time = datetime.now()
print(f"Starting MCMC at {start_time}")
results = mcmc.run()
end_time = datetime.now()
print(f"Finished at {end_time} in {end_time - start_time}")

# --- SAVE RESULTS ---
# Modify filename to include the cut-off date to prevent overwriting
filename = f"posterior_{run_region}_{run_cutoff_date.isoformat()}_{run_chain}.npy"
np.save(os.path.join(RESULTS_DIR, filename), results)
print(f"Results saved to {filename}")