
import os, sys

# File location
FILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(FILE_BASE)

from covid19_seir.models import params, prevalence_model
from covid19_seir.models.logPDF_prev import LogPosteriorBetaBinomial
from datetime import date, timedelta
from math import ceil
from scipy import stats
import numpy as np
import pandas as pd

# CONSTANTS
START_DATE = date(2020, 8, 31)
END_DATE = date(2021, 1, 24)
DATA_FILE = "data/england_modelling_data.csv"
OUTPUT_DIR = "results/"
REGIONS = [
    'North_East_England',
    'North_West_England',
    'Yorkshire',
    'East_Midlands',
    'West_Midlands',
    'East_England',
    'London',
    'South_East_England',
    'South_West_England',
]
NUM_CHAINS_PER_REGION = 4
RESULTS_DIR = "/rds/user/jbb50/hpc-work/SEIR_model/CIS"

# TIME
dates = pd.date_range(START_DATE, END_DATE, freq="1D").date
N_DAYS = (END_DATE - START_DATE).days + 1
N_WEEKS = N_DAYS / 7


# CONTACT MATRICES
n_matrices = ceil(N_WEEKS)
matrix_dates = pd.date_range(START_DATE, END_DATE-timedelta(days=1), freq="7D")
assert len(matrix_dates) == n_matrices
matrices_list = []
for d in matrix_dates:
    file = os.path.join(FILE_BASE, f"data/contact_m/England/{d.date()}.csv")
    matrices_list.append(np.loadtxt(file, delimiter=","))
matrices = np.stack(matrices_list)
N_STRATA = matrices.shape[1]

# SETUP FORWARD MODEL
forward_model = prevalence_model.SingleRegionStratifiedSEIRCompartmental(matrices, N_DAYS)
PROBABILISTIC_MODEL_CLASS = LogPosteriorBetaBinomial

# PARAMETERS
BASE_PRIORS = {
    "dL": params.FixedParam(3.5),
    "dI": params.FixedParam(4),
    "i0": params.SampleLogitScale(params.Param(stats.beta(0.5, 1000))),
    "matrix_modifiers": params.MatrixSusceptibleChildren(
        params.ExpAfterPrior(stats.norm(-0.4325, 0.1174)),
        N_STRATA,
        1
    ),
    "psir": params.Param(stats.norm(0.06, 0.04)),
    "beta": params.LogGaussianRWNonCentred(
        stats.expon(scale=1/80),
        forward_model.num_betas,
        sample_log_scale=True
    ),
    "theta": params.SampleLogScale(params.Param(stats.expon(scale=2e-5))),
}

PI_PARAMS = {
    'North_East_England' : [
        [180, 6.1],
        [180, 6.1],
        [145, 12],
        [274, 14],
        [295, 14],
        [240, 4.2],
    ],
    'North_West_England': [
        [252, 12],
        [252, 12],
        [353, 38],
        [885, 72],
        [695, 44],
        [264, 6.3]
    ],
    'Yorkshire': [
        [314, 9.2],
        [314, 9.2],
        [242, 19],
        [589, 27],
        [474, 18],
        [316, 4.9],
    ],
    'East_Midlands': [
        [301, 8.5],
        [301, 8.5],
        [256, 18],
        [475, 19],
        [493, 20],
        [337, 5],
    ],
    'West_Midlands': [
        [167, 8],
        [167, 8],
        [151, 19],
        [475, 32],
        [504, 33],
        [241, 5.8],
    ],
    'East_England': [
        [236, 7.9],
        [236, 7.9],
        [248, 21],
        [677, 35],
        [613, 26],
        [332, 5.6],
    ],
    'London': [
        [131, 9.7],
        [131, 9.7],
        [328, 50],
        [1331, 154],
        [990, 106],
        [204, 7.6],
    ],
    'South_East_England': [
        [404, 8.9],
        [404, 8.9],
        [319, 17],
        [577, 20],
        [586, 19],
        [356, 4.4],
    ],
    'South_West_England': [
        [415, 6.9],
        [415, 6.9],
        [277, 11],
        [556, 12],
        [559, 11],
        [492, 4],
    ],
}
PI_PRIORS = {
    k: params.VectorParamMultiplePriors(
        N_STRATA, [stats.beta(*p) for p in v]
    )
    for k, v in PI_PARAMS.items()
}