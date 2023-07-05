from config import *
from covid19_seir.mcmc.single_block_adapt_mcmc import SingleBlockAMGS
from covid19_seir.util.ess import effective_sample_size
from datetime import datetime
from pandas import read_csv
import csv

# DECIDE WHICH SIMULATION'S DATA WE USE
try:
    SIM_NUM = int(os.environ["SLURM_ARRAY_TASK_ID"])
except KeyError:
    try:
        SIM_NUM = int(sys.argv[1])
    except:
        SIM_NUM = 1
        print(f"WARNING: running sim {SIM_NUM} as default")

# READ THE DATA
data = read_csv(
    os.path.join(RESULTS_DIR, "sim_output.csv"),
    dtype={"age": "string"},
    parse_dates=[2],
    index_col=0
)
data = data[data["sim"] == SIM_NUM]
data_shape = (N_DAYS, N_STRATA)
num_tested = data["num_tests"].values.reshape(data_shape)
num_positive = data["obs_positives"].values.reshape(data_shape)

# RUN THE MCMC
log_posterior = PROBABILISTIC_MODEL_CLASS(model, num_positive, num_tested, PRIORS)
start = log_posterior.param_value_dict_to_array({
    k: v.draw_from_prior() for k, v in PRIORS.items()
})
start_time = datetime.now()
mcmc = SingleBlockAMGS(log_posterior, start, iterations=500_000)
mcmc_out = mcmc.run()
end_time = datetime.now()
print("MCMC took {} seconds".format(end_time - start_time))

# SAVE RESULTS
np.save(os.path.join(RESULTS_DIR, f"posterior_{SIM_NUM}.npy"), mcmc_out)

# ESS
ess_out = effective_sample_size(mcmc_out[300_000:])
with open(os.path.join(RESULTS_DIR, f"ESS_{SIM_NUM}.npy"), 'w') as ess_file:
    wr = csv.writer(ess_file)
    wr.writerows(zip(log_posterior.get_param_names(), ess_out))