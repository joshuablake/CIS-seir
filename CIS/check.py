from config import *
from covid19_seir.models import prevalence_model as model
import numpy as np
import pandas as pd

for r in REGIONS:
    for c in range(NUM_CHAINS_PER_REGION):
        f = os.path.join(RESULTS_DIR, f"posterior_{r}_{c}.npy")
        # Read matrix of iteration x parameter, entries are values
        if not os.path.isfile(f):
            print(f)