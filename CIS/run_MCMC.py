from config import *
from covid19_seir.mcmc.single_block_adapt_mcmc import SingleBlockAMGS
from covid19_seir.util.ess import effective_sample_size
from datetime import datetime
import numpy as np
import pandas as pd

JOB_NUM = int(sys.argv[1])
np.random.seed(JOB_NUM*34)

# Check what run this is
REGION = REGIONS[JOB_NUM // NUM_CHAINS_PER_REGION]
CHAIN = JOB_NUM % NUM_CHAINS_PER_REGION
print(f"Running {REGION} chain {CHAIN}")

# Prevalence data
all_data_in = pd.read_csv("SRS_extracts/20230829_STATS18115/modelling_data.csv", parse_dates=["date"])
data_shape_for_model = (N_DAYS, N_STRATA)
data_in = all_data_in[(all_data_in["region"] == REGION) & (all_data_in["date"].isin(dates))]
array_num_tested = data_in["num_tests"].values.reshape(data_shape_for_model)
array_num_positive = data_in["obs_positives"].values.reshape(data_shape_for_model)

# Setup priors
priors = BASE_PRIORS.copy()
priors["pi"] = PI_PRIORS[REGION]

# Setup inference
start_dict = {
    k: v.draw_from_prior() for k, v in priors.items()
}
log_posterior = PROBABILISTIC_MODEL_CLASS(forward_model, array_num_positive, array_num_tested, priors)
# Below line needed if using an uninformative prior on psir to give sensible start conditions
start_dict["psir"] = stats.norm.rvs(size=1, loc=0.056, scale=0.01)
start = log_posterior.param_value_dict_to_array(start_dict)

# Do inference
mcmc = SingleBlockAMGS(log_posterior, start, iterations=1_000_000)
start_time = datetime.now()
print(f"Starting MCMC at {start_time}")
results = mcmc.run()
end_time = datetime.now()
print(f"Finished at {end_time} in {end_time - start_time}")

# Save results
np.save(os.path.join(RESULTS_DIR, f"posterior_{REGION}_{CHAIN}.npy"), results)