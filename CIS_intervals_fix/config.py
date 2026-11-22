# file: experiment_setup.py
import os, sys

# File location
FILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.insert(0, FILE_BASE)  # fix copy: ensure this clone's covid19_seir is imported

from covid19_seir.models import params, prevalence_model
from covid19_seir.models.logPDF_prev import LogPosteriorBetaBinomial
from datetime import date, timedelta
from math import ceil
from scipy import stats
import numpy as np
import pandas as pd

# Original constants that don't change
START_DATE = date(2020, 8, 31)
FULL_END_DATE = date(2021, 1, 24)
NUM_CHAINS_PER_REGION = 4

# Posterior post-processing (iteration numbers are before thinning)
BURN_IN = 1_000_000        # drop iterations < BURN_IN
PARAM_THIN = 5000          # params CSVs: every 5000th iteration, burn-in included
PREDICTIVE_THIN = 20000    # predictive/final-state: post-burn-in draws on this grid only
# SEIR_RESULTS_DIR / SEIR_FOR_PAPER_DIR overrides are for read-only re-analysis of other runs (e.g. the original sd0.1 run)
RESULTS_DIR = os.environ.get("SEIR_RESULTS_DIR", "/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals_fix_omega0")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
# Outputs for the paper, inside this script directory
FOR_PAPER_DIR = os.environ.get("SEIR_FOR_PAPER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "for_paper"))
N_STRATA = 6
REGIONS = [
    'North_East_England', 'North_West_England', 'Yorkshire', 'East_Midlands',
    'West_Midlands', 'East_England', 'London', 'South_East_England', 'South_West_England'
]
PI_PARAMS = { # ... (copy the full PI_PARAMS dictionary from config.py here) ...
    'North_East_England' : [
        [180, 6.1], [180, 6.1], [145, 12], [274, 14], [295, 14], [240, 4.2],
    ],
    'North_West_England': [
        [252, 12], [252, 12], [353, 38], [885, 72], [695, 44], [264, 6.3]
    ],
    'Yorkshire': [
        [314, 9.2], [314, 9.2], [242, 19], [589, 27], [474, 18], [316, 4.9],
    ],
    'East_Midlands': [
        [301, 8.5], [301, 8.5], [256, 18], [475, 19], [493, 20], [337, 5],
    ],
    'West_Midlands': [
        [167, 8], [167, 8], [151, 19], [475, 32], [504, 33], [241, 5.8],
    ],
    'East_England': [
        [236, 7.9], [236, 7.9], [248, 21], [677, 35], [613, 26], [332, 5.6],
    ],
    'London': [
        [131, 9.7], [131, 9.7], [328, 50], [1331, 154], [990, 106], [204, 7.6],
    ],
    'South_East_England': [
        [404, 8.9], [404, 8.9], [319, 17], [577, 20], [586, 19], [356, 4.4],
    ],
    'South_West_England': [
        [415, 6.9], [415, 6.9], [277, 11], [556, 12], [559, 11], [492, 4],
    ],
}
# PI_PRIORS can also be defined here, as it doesn't depend on the end date
PI_PRIORS = {
    k: params.SampleLogitScale(
        params.VectorParamMultiplePriors(
            N_STRATA, [stats.beta(*p) for p in v] # N_STRATA is 17
        )
    )
    for k, v in PI_PARAMS.items()
}

# Generate two-weekly cut-off dates
cutoff_dates = pd.date_range(START_DATE + timedelta(days=14) - timedelta(days=1), FULL_END_DATE, freq="14D").date
# append the full end date if it's not already included
if cutoff_dates[-1] != FULL_END_DATE:
    cutoff_dates = np.append(cutoff_dates, FULL_END_DATE)

# --- NEW: SLURM ID MAPPING ---
# Map the single SLURM_ARRAY_TASK_ID to the (cutoff, region, chain) tuple
num_regions = len(REGIONS)
num_cutoffs = len(cutoff_dates)


def setup_experiment_config(end_date):
    """
    Generates the configuration for a single experimental run
    with a specific data cut-off date.
    """
    # TIME
    dates = pd.date_range(START_DATE, end_date, freq="1D").date
    N_DAYS = (end_date - START_DATE).days + 1
    N_WEEKS = N_DAYS / 7

    # CONTACT MATRICES
    n_matrices = ceil(N_WEEKS)
    matrix_dates = pd.date_range(START_DATE, end_date - timedelta(days=1), freq="7D")
    assert len(matrix_dates) == n_matrices
    matrices_list = []
    for d in matrix_dates:
        file = os.path.join(FILE_BASE, f"data/contact_m/England/{d.date()}.csv")
        matrices_list.append(np.loadtxt(file, delimiter=","))
    matrices = np.stack(matrices_list)
    assert N_STRATA == matrices.shape[1]

    # SETUP FORWARD MODEL
    forward_model = prevalence_model.SingleRegionStratifiedSEIRCompartmental(matrices, N_DAYS)
    PROBABILISTIC_MODEL_CLASS = LogPosteriorBetaBinomial

    # PARAMETERS
    BASE_PRIORS = {
        "dL": params.FixedParam(3.5),
        "dI": params.FixedParam(4),
        "i0": params.SampleLogitScale(
            params.Param(stats.beta(0.5, 1000))
        ),
        "matrix_modifiers": params.MatrixSusceptibleChildren(
            params.ExpAfterPrior(stats.norm(-0.4325, 0.1174)),
            N_STRATA,
            1
        ),
        "psir": params.Param(stats.norm(0.06, 0.04)),
        "beta": params.LogGaussianRWNonCentred(
            stats.expon(scale=1/10),
            forward_model.num_betas,
            sample_log_scale=True
        ),
        "theta": params.SampleLogScale(params.Param(stats.expon(scale=2e-5))),
    }
    
    # Return all necessary config variables as a dictionary
    config = {
        "dates": dates,
        "N_DAYS": N_DAYS,
        "N_STRATA": N_STRATA,
        "forward_model": forward_model,
        "PROBABILISTIC_MODEL_CLASS": PROBABILISTIC_MODEL_CLASS,
        "BASE_PRIORS": BASE_PRIORS,
        "PI_PRIORS": PI_PRIORS,
        "REGIONS": REGIONS
    }
    return config