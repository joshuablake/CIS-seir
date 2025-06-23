from config import NUM_CHAINS_PER_REGION, RESULTS_DIR, REGIONS, cutoff_dates
from covid19_seir.models import prevalence_model as model
import os
import pandas as pd

i = 0
for run_region in REGIONS:
    for run_chain in range(NUM_CHAINS_PER_REGION):
        for run_cutoff_date in cutoff_dates:
            f = os.path.join(RESULTS_DIR, f"posterior_{run_region}_{run_cutoff_date.isoformat()}_{run_chain}.npy")
            if not os.path.isfile(f):
                print(f, " ", i)
            i += 1