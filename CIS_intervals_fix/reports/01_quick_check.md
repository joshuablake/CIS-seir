# 01 — Quick check: ω₀ fix applied to the original final-cut-off posterior

*2026-11-20. Slurm jobs 41494693 (`quick_check.py`) and 41495449 (`quick_check_fit_summary.py`).*

## What was done

- **Draws:** original posterior (`CIS_intervals_sd0.1/params_<region>_2021-01-24.csv`), burn-in 1e6, every 20000th iteration. That gives 1,800 draws per region (4 chains × 450), 16,200 in total.
- **Forward model:** rerun for each draw in four variants:

  | variant | initial P state | parameters |
  |---|---|---|
  | `old` | buggy (ω₁ in p₀) | as fitted |
  | `new_same` | corrected | as fitted |
  | `new_resc` | corrected | i⁺ × F, where F = (ψ+ω₀)/(ψ+ω₁), so that day-0 PCR prevalence matches exactly (p_init is linear in i⁺) |
  | `new_fit` | corrected | (i⁺ factor, Δψ) chosen to best match the old log prevalence on days 0–27, with β fixed. 40 draws per region (360 fits, all converged) |

- **Summaries:** population-weighted regional values; England combines regions by population.
- **Outputs:** `reports/quick_check/`.

## Results

**Size of the bug** (medians over draws):

| | range across regions |
|---|---|
| F (day-0 inflation of every p_k) | 2.34 (NW) – 2.86 (SE) |
| share of old prevalence that is spurious, day 14 | 29–37% |
| share of old prevalence that is spurious, day 28 | 7–10% |

England prevalence with the same parameters (`new_same`/`old`) is 0.39 on 31 Aug, 0.75 on 14 Sep, 0.96 on 28 Sep and 0.99 on 12 Oct.

**Rescaling i⁺ alone does not work.** `new_resc` matches day 0 but then overshoots. England prevalence is 1.9× old on 14 Sep and 2.3× on 28 Sep. So a refit must also lower ψ.

**What a corrected model needs to reproduce the same early prevalence** (`new_fit`, 360 draws):

| | median (range of region medians) |
|---|---|
| i⁺ factor | 1.99 (1.90–2.12) |
| Δψ (per day) | −0.036 (−0.034 to −0.038) |
| RMS log error vs old prevalence, days 0–27 | 0.066 (fits closely) |

Under that matched fit, new/old ratios by day (median across regions; range in brackets):

| day | incidence | prevalence |
|---|---|---|
| 0 (31 Aug) | **1.59** (1.55–1.68) | 0.93 |
| 7 | 1.26 | 1.01 |
| 14 | 0.98 | 1.09 |
| 21 | 0.77 | 1.01 |
| 28 | 0.61 | 0.86 |

Cumulative infection incidence over September (days 0–29) is 0.82–0.87× old.

After about day 21 the matched fit drifts away from the data because β is held fixed. A real refit will let β(t) absorb this, so ratios after September aren't meaningful here.

## Verdict

- The handoff's analytic estimates are confirmed:
  - i⁺ is about 2× higher;
  - ψ is about 0.036/day lower;
  - incidence is about 1.6× higher on 31 Aug, about equal by 14 Sep and lower after.
- So the published fits probably overstate September growth. A corrected fit has a flatter early epidemic that starts higher.
- **ψ bound:** the old ψ medians (0.053–0.106) minus 0.036 give 0.017–0.070, well above the −0.0223 bound. The relaxed bound is not expected to bind. The N(0.06, 0.04) prior will resist part of the shift, so expect a smaller Δψ in the refit, with weekly β steps taking up the rest.
- **Next:** pilot refit at the final cut-off (array tasks 360–395, job 41495931), submitted 10:14.
