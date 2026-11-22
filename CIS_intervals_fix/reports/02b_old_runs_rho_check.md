# 02b — Were the original (published) runs hit by the low-ρ artefact?

*2026-11-20. Slurm job 41503709 (`check_old_rho.py`). Read-only scan of `CIS_intervals_sd0.1/params_*.csv` (thinned every 5000), all 396 chains. Full table in `old_rho_check.csv`.*

## Result

Two of the 396 original chains are **trapped** in the region where scipy's beta-binomial is unreliable: log ρ ≈ −42, with a constant value. This is the same failure as pilot task 386.

| cut-off | region | chain | trapped from iteration | share of draws trapped | original diagnostics |
|---|---|---|---|---|---|
| 2020-09-13 | North East | 2 | 2.56M | 74% | R-hat up to 1.39 (median over params 1.26) |
| 2021-01-17 | South West | 0 | 115k | 99% | R-hat up to 1.59 (median over params 1.36) |

- **Every other region/cut-off:** max R-hat is 1.01 in the original `all_diagnostics.csv`.
- **Other low dips:** 14 more chains dipped below log ρ = −20 at some point. All stayed above −25.3, where the likelihood is still accurate (checked to 1e-5 at ρ = 1e-10).
- **Typical posterior:** median log ρ about −10.8; the 2.5% quantile is about −13.5.

## Implications

- **What the outputs contain:** `posterior_as_csv.py` pools all 4 chains. So the published outputs for **NE at 13 Sep** and **SW at 17 Jan** include about 25% of draws from a stuck, invalid chain: a single fixed parameter vector with a spurious likelihood.
- **Paper outputs that use them:**
  - NE 13 Sep feeds the early-cut-off nowcast results.
  - SW 17 Jan feeds the 17 Jan nowcast and the England aggregate at that cut-off.
  - The final-cut-off (24 Jan) "gold standard" fits are not affected.
- **Size of the effect:** not yet known. It could be measured by recomputing those two summaries without the trapped chain.
- **In the refit:** the guard (ρ < 1e-11 → −inf) prevents this. Refit diagnostics will show whether every region/cut-off then reaches R-hat ≤ 1.01.

## Posterior mass near the guard threshold (log ρ = −25.3)

| | share of draws with log ρ < −20 | lowest value seen |
|---|---|---|
| Prior, ρ ~ Exp(5e4) | 1.0e-4 (and 5e-7 below −25.3) | — |
| Original runs (394 healthy chains, all iterations) | 4e-5 | −24.8 |
| Pilot, iterations < 250k (early adaptation) | 3.4e-3 | −22.3 |
| Pilot, iterations ≥ 250k | 0 (none even below −15) | −13.9 |

- **Conclusion:** there is no meaningful posterior mass near the threshold, so the guard removes nothing of substance.
- **Why the tail is like this:** below roughly nρ ≪ 1 the beta-binomial converges to the binomial. The likelihood is then flat in the ρ direction (though not in π), so the posterior for log ρ there is just the prior, which falls off like e^(log ρ).
- **What went wrong in the trapped chains:** scipy's precision loss created a false peak on that plateau. The chains reached it by jumping during early adaptation, not by drifting through posterior mass.
- **Decision (Joshua, 20 Nov):** keep the −inf guard for this refit; no binomial fallback.

## Now part of the run diagnostics (20 Nov)

**`compute_diagnostics.py`** writes `chain_checks.csv` alongside `all_diagnostics.csv`. Per chain it records:
- min log ρ (all iterations and post-burn-in) and the 2.5% quantile;
- the share of post-burn-in draws below −20 and of all draws below the guard (−25.3);
- the share of repeated consecutive thinned draws (a stuck chain).

It sets two flags:
- `flag_invalid`: below the guard, or >1% repeated draws. Rerun the chain with `SEIR_RETRY`.
- `flag_low_rho`: >0.1% of post-burn-in draws below −20.

**`checkpoint_report.py`** applies the same flags during runs, and also flags log target > 0.

**Validation on the original results** (read only, output in `debug/diag_old/`): `flag_invalid` catches exactly the two trapped chains (NE 13 Sep chain 2, SW 17 Jan chain 0). On the pilot checkpoints it catches London chain 2.

**Bug in the original `compute_diagnostics.py`:** its loop stopped at job 89, so the final cut-off (24 Jan) was never diagnosed. Now computed for the original runs: max R-hat 1.006, min bulk ESS 1526. Excluding the two trapped combinations, all 99 combinations have R-hat ≤ 1.007 and bulk ESS ≥ 400.
