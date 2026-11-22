# 02c — Interim report 2 (pilot at 1–1.5M iterations; first full-refit checkpoints)

*2026-11-20 14:25. Report job 41511528, plus 41513064 for the two earliest cut-offs.*

## Status

- **Jobs:** all tasks have a node. The queue is no longer limiting progress.
- **New SVD crashes:** tasks 354 and 357, both at the 17 Jan cut-off (South East chain 2 and South West chain 1).
  - Both resubmitted with `SEIR_RETRY=1` as job 41511282.
  - Crashed logs kept as `slurm/crashed_*_{354,357}_seed0.txt`.
- **Checkpoints:** 195 so far.
  - Pilot: 1.5M iterations. The exceptions are 386 (guard rerun) at 500k and 374 (retry) at 1M.
  - Oct cut-offs: 0.5–1M iterations. Sep cut-offs: up to 2M.
- **Acceptance:** 0.230–0.233 in every chain.
- **Screening:** `flag_invalid` is clear on every chain. No draw is below the guard, there are no positive log targets and there are no repeated-draw runs.
  - One `flag_low_rho` hit: London chain 3 at the 11 Oct cut-off, a single thinned draw at log ρ −22.1. This is benign; it is the same kind of dip seen in the original runs.

## Pilot (cut-off 24 Jan), draws after 750k

| region | ψ median (95% CrI) | i⁺ | child susc. | β wk1 / wk2 / wk4 |
|---|---|---|---|---|
| North East | 0.074 (0.041–0.109) | 6.7e-4 | 0.75 | 1.02 / 0.94 / 0.58 |
| North West | 0.078 (0.053–0.105) | 8.2e-4 | 0.77 | 0.89 / 0.84 / 0.75 |
| Yorkshire | 0.074 (0.051–0.102) | 5.7e-4 | 0.70 | 0.95 / 0.89 / 0.73 |
| East Midlands | 0.059 (0.040–0.083) | 2.9e-4 | 0.75 | 0.98 / 0.98 / 0.89 |
| West Midlands | 0.084 (0.051–0.115) | 1.4e-4 | 0.81 | 1.01 / 0.97 / 0.84 |
| East of England | 0.046 (0.021–0.071) | 2.6e-4 | 0.83 | 0.94 / 0.92 / 0.90 |
| London | 0.041 (0.020–0.057) | 5.3e-4 | 0.89 | 0.94 / 0.93 / 1.01 |
| South East | 0.029 (0.009–0.047) | 2.4e-4 | 0.88 | 0.99 / 1.01 / 1.03 |
| South West | 0.063 (0.034–0.097) | 1.9e-4 | 0.70 | 0.95 / 0.88 / 0.72 |

- **ψ:** still 0.02–0.03 below the original fit and stable since checkpoint 1. The lowest 2.5% quantile is 0.009 (South East). The −0.0223 bound is not approached.
- **Convergence:** R-hat is ≤ 1.05 for every region and quantity except two:
  - London (rw_sd 1.67): the guarded rerun of chain 2 has only 500k iterations, so it is still in burn-in. This should resolve.
  - South East rw_sd (1.07): this is the random-walk sd, which has a heavy left tail.

## Full refit, early look

| cut-off | iterations | ψ median range | max R-hat |
|---|---|---|---|
| 11 Oct | 1M | 0.031–0.078 | 1.02 |
| 25 Oct | 0.5M | 0.030–0.077 | 1.05 |
| 8 Nov | 0.5M | 0.030–0.079 | 1.05 |

- ψ is consistent across cut-offs and matches the pilot.
- The two September cut-offs failed in the report script, which assumed at least 4 β steps. That is fixed and the report is being rerun.

## Next

- The September cut-offs (~3–6 h runs) should produce the first finished posteriors this afternoon.
- Next routine check is around 15:25.
