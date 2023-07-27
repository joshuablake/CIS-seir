# Use modules in this file's parent directory
import os, sys
FILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(FILE_BASE)

from scipy import stats
from covid19_seir.models import params, prevalence_model
import numpy as np
from pandas import date_range, read_csv
from datetime import date, timedelta
from math import ceil

from covid19_seir.models.logPDF_prev import LogPosteriorBetaBinomial

# CONSTANTS
START_DATE = date(2020, 8, 31)
END_DATE = date(2021, 1, 24)
NUM_SIMS = 100
N_STRATA = 6
NUM_DAILY_TESTS = 15e3
DATA_DIR = "/rds/user/jbb50/hpc-work/SEIR_model"
INFERENCE_DIR = "/rds/user/jbb50/hpc-work/SEIR_model/stop_adapt"

# TIME
dates = date_range(START_DATE, END_DATE, freq="1D").date
N_DAYS = (END_DATE - START_DATE).days + 1
N_WEEKS = N_DAYS / 7


# CONTACT MATRICES
n_matrices = ceil(N_WEEKS)
matrix_dates = date_range(START_DATE, END_DATE-timedelta(days=1), freq="7D")
assert len(matrix_dates) == n_matrices
matrices_list = []
for d in matrix_dates:
    file = os.path.join(FILE_BASE, f"data/contact_m/England/{d.date()}.csv")
    matrices_list.append(np.loadtxt(file, delimiter=","))
matrices = np.stack(matrices_list)
assert matrices.shape[1] == N_STRATA

# DEMOGRAPHICS
pop = read_csv(os.path.join(FILE_BASE, "data/contact_m/population.csv"))
STRATA_NAMES = list(pop["AgeGroup"])

# SETUP FORWARD MODEL
model = prevalence_model.SingleRegionStratifiedSEIRCompartmental(matrices, N_DAYS)
PROBABILISTIC_MODEL_CLASS = LogPosteriorBetaBinomial

# PARAMETERS
susc_beta_params = np.array([
    [  87.550464  ,    3.647936  ],
    [ 126.351416  ,    8.064984  ],
    [ 103.12896768,    8.96773632],
    [ 333.32631547,   24.19207796],
    [  55.39439784,    1.47870056],
    [1184.47150779,   40.42146821]
])
susc_betas = [stats.beta(*p) for p in susc_beta_params]
PRIORS = {
    "dL": params.FixedParam(3.5),
    "dI": params.FixedParam(4),
    "pi": params.VectorParamMultiplePriors(N_STRATA, susc_betas),
    "i0": params.SampleLogitScale(params.Param(stats.beta(8.60, 27400))),
    "matrix_modifiers": params.MatrixSusceptibleChildren(params.ExpAfterPrior(stats.norm(-0.4325, 0.1174)), N_STRATA, 1),
    "psir": params.Param(stats.norm(0.048, 0.0035)),
    "beta": params.LogGaussianRWNonCentred(stats.expon(scale=1/50), model.num_betas),
    "theta": params.SampleLogScale(params.Param(stats.expon(scale=2e-5))),
}