# 02 — Pilot interim report 1 (first checkpoints, 500k iterations)

*2026-11-20 11:45. Pilot job 41495931 (tasks 360–395, cut-off 2021-01-24). Report job 41501821.*

## Status

- **Checkpoints:** 33 of 36 chains have checkpointed at 500k iterations.
  - Not yet: 377 and 380 (slower nodes), and 374, which was rerun with a new seed as job 41500413.
- **Throughput:** about 500k iterations in ~75 min, so the forecast finish is about 24–25 h after each task started (**~11:00 Sat 21 Nov**).
- **Acceptance:** 0.228–0.231 in every chain (target 0.234). No stuck chains apart from the one below.

## Problem: London chain 2 (task 386) is in a spurious mode

- **What happened:** at iteration ~20k its log target jumped from −1853 to +7.9e6, and it is now +1.54e7. Since then it has been locked at log ρ (`theta`) ≈ −42, i.e. ρ ≈ 6e-19. It also has absurd π for age 71+ (logit −3.3).
- **Cause:** a pre-existing numerical bug in `LogPosteriorBetaBinomial._log_lik`. That code is unchanged by the fix.
  - It computes α = π/ρ and β = (1−π)/ρ.
  - scipy 1.10.1's `betabinom.logpmf` loses precision for ρ ≲ 1e-12.
  - Example with k=3, n=300, p=0.01: ρ=1e-12 gives −1.499 and ρ=1e-15 gives −4.69, against the correct −1.491. At ρ=6e-19 the result is pure noise and can be hugely positive.
  - Nothing in the prior keeps log ρ away from that region.
- **Not caused by the fix:** ψ = 0.012 here, which is inside the old ψ > 0 region.
- **Possible link:** this may also be behind the SVD crashes (a log target blowing up corrupts the adapted covariance). That isn't verified.
- **Effect on results:** this chain's output is unusable. It is the only reason London's R-hat is 1.5.

## Early posterior (draws 250k–500k, 50 per chain; very preliminary)

| region | ψ median (95% CrI) | old ψ median | child susceptibility |
|---|---|---|---|
| North East | 0.080 (0.048–0.111) | ~0.096 | 0.74 |
| North West | 0.078 (0.054–0.109) | 0.106 | 0.77 |
| Yorkshire | 0.072 (0.053–0.095) | ~0.099 | 0.70 |
| East Midlands | 0.057 (0.039–0.080) | ~0.078 | 0.76 |
| West Midlands | 0.077 (0.054–0.103) | ~0.098 | 0.82 |
| East of England | 0.043 (0.022–0.067) | ~0.066 | 0.83 |
| London* | 0.038 (0.012–0.056) | ~0.066 | 0.91 |
| South East | 0.028 (0.012–0.042) | 0.053 | 0.88 |
| South West | 0.063 (0.037–0.087) | ~0.085 | 0.70 |

\*London includes the broken chain. "Old" values marked ~ are quick-check medians of the original posterior.

- **ψ:** 0.020–0.029 lower than the original in every region. That's about two-thirds of the −0.036 the quick check predicted; the prior and β(t) absorb the rest, as expected.
- **ψ bound:** no region is near −0.0223. The lowest 2.5% quantile is 0.012.
- **Convergence:** R-hat (ψ, i⁺, log target) is ≤ 1.07 for every region except London. That is a good sign this early.
- **i⁺:** medians 2e-4 to 8e-4, about 2× the original as predicted. This needs a proper old-vs-new comparison at the next checkpoint.
- **Early β:** week-1 and week-2 multipliers are 0.84–1.03, with no pathology.

## Decision needed

How should London chain 2 (task 386), and any recurrence in the full refit, be handled?

- **Option A — Rerun (recommended for consistency):** cancel 386 and rerun it with `SEIR_RETRY=1`. The model is unchanged. Each checkpoint gets screened for log target > 0 and any affected chain is rerun the same way.
- **Option B — Guard:** add a guard to the likelihood, so that log ρ below about −25 (ρ < 1e-11) returns −inf or uses the binomial limit, then rerun 386 and run the full refit with the guard. The 35 healthy pilot chains needn't be rerun: they never accepted a state in that region, so they would be the same under the guard. That's a model-code change, though the prior mass there is negligible (P(ρ < 1e-11) ≈ 5e-7). It also differs slightly from the original runs, which could have hit the same artefact.

The full refit (tasks 0–359) is held until this is decided.

## Decision and actions (11:51)

- **Decision:** Joshua chose **Option B (guard)**.
- **Guard:** `LogPosteriorBetaBinomial._log_lik` returns −inf for ρ < 1e-11 (`BETABINOM_THETA_MIN`). covid19-seir commit on `omega0-fix`, local only.
  - Checked: ρ = 1e-10 gives −1.49089 (exact binomial −1.49090); ρ < 1e-11 gives −inf.
- **Task 386:** cancelled and rerun with the guard and the original seed. The stuck logs and checkpoint are kept as `slurm/stuck_*_386_noguard.txt` and `debug/stuck_ckpt_London_2021-01-24_2_noguard.npz`.
- **Other pilot chains:** the 35 healthy ones, including the 374 retry, were started before the guard. That is fine provided they never accept a state with ρ < 1e-11. To confirm, check min log ρ in the final chains (it must stay above −25.3) and screen checkpoints for log target > 0.
- **374 crash (debug job 41500325):** reproduced at iteration 27415.
  - The state and covariance were finite and the log target normal (−1308).
  - log ρ was falling fast (−13 → −15.4), so the crash fits the same low-ρ drift but is not the garbage region itself.
- **Full refit:** tasks 0–359 submitted with the guard.
