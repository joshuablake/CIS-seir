# CIS_intervals: fitting the SEIR model to CIS data

These scripts produce the mechanistic-model results in Blake et al., "The added value of randomly sampled prevalence surveys to estimate infection incidence and transmission".

## Changes since v1.0.0 (November 2026)

v1.0.0 contained an error in the model's initial state. All results were refitted with these changes:

- **Model fixes**, from covid19-seir after joshuablake/covid19-seir#1:
  - The initial PCR-positive state uses ω₀, the exit rate of P0. v1.0.0 used ω₁, which inflated the initial P states by about 2.3–2.9×.
  - The growth-rate bound is ψ > −ω₃ (previously ψ > 0).
  - The beta-binomial likelihood is guarded against θ < 1e-11.
- **Checkpointed MCMC** (`checkpointed_mcmc.py`). Run length, burn-in, thinning and seeds are the same as before. `SEIR_RETRY=k` reruns a task that failed numerically, with a different recorded seed; 3 of the 396 chains needed this.
- **Shared thinning constants** in `config.py`. Parameter CSVs keep every 5000th iteration, burn-in included. Predictive and final-state outputs use only post-burn-in draws on a 20000-iteration grid, which is 1800 per region.
- **Compressed posteriors.** Files are saved with `np.savez_compressed`, under the same name and key, so `np.load(...)["arr_0"]` is unchanged. `storage/compress_npz.py` (`storage/SLURM_compress_sd01`) losslessly compressed the existing v1.0.0 posteriors in place.
- **Diagnostics** now include the final cut-off, plus screening for chains with very low ρ or stuck chains.

## Running it

Submit the jobs from this directory on CSD3. They use the Python 3.8 venv at the repository root (`../.venv`).

1. **MCMC:** `sbatch SLURM`. The array covers 0–395: 9 regions × 11 cut-offs × 4 chains, 10M iterations each.
2. **Posterior CSVs:** `sbatch SLURM_csv`. The array covers 0–98, one task per region and cut-off.
3. **Diagnostics and exports:** `sbatch SLURM_export`. This writes `for_paper/`, which the paper's R figure scripts read as `model-outputs/mechanistic/`.
4. **Optional diagnostic plots:** `sbatch SLURM_visualise`.

`tests/test_initial_state.py` checks that the corrected initial state is a steady state of the observation model.

## Comparing with v1.0.0

`compare_old_new.py` compares parameters, incidence, nowcast accuracy and MCMC diagnostics between the original run and this refit. To get its inputs for the v1.0.0 run, run `compute_diagnostics.py` and `export_summary_stats.py` on the v1.0.0 posteriors. Use `DIAG_IN_DIR`, `DIAG_OUT_DIR`, `SEIR_RESULTS_DIR` and `SEIR_FOR_PAPER_DIR` to point them at those results, writing to `reports/old_reanalysis/` (or set `OLD_DIR`). Then run `OLD_PARAMS=<v1.0.0 for_paper/params.csv.gz> python compare_old_new.py`.

**Result:**
- The paper's reported numbers change only at the third digit.
- ψ falls by about 0.02 per day, and i⁺ rises 1.5–1.9×.
- Incidence is higher at the start of September (×1.5 on 31 Aug) and unchanged from late September.
- The September decline in β(t) seen in v1.0.0 disappears.
- R-hat is ≤ 1.009 for every parameter at every cut-off.

The refit's results are in `/rds/user/jbb50/hpc-work/SEIR_model/CIS_intervals_fix_omega0` (`RESULTS_DIR` in `config.py`); the v1.0.0 results are in `CIS_intervals_sd0.1`. Generated reports, intermediate CSVs and one-off debugging scripts from the refit are kept outside the repository.
