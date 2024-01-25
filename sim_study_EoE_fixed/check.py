from config import *

for i in range(NUM_SIMS):
    f = os.path.join(RESULTS_DIR, f"posterior_{i}.npy")
    if not os.path.isfile(f):
        print(f)