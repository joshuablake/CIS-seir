from config import *
import arviz as az
import numpy as np
import pandas as pd
import os
import sys

# Low-rho screening: scipy's betabinom.logpmf is unreliable for rho < ~1e-12;
# the likelihood guards rho < 1e-11 (log rho < -25.3) -> -inf.
# Overrides for validation against other result sets (outputs never go to the input dir)
IN_DIR = os.environ.get("DIAG_IN_DIR", RESULTS_DIR)
OUT_DIR = os.environ.get("DIAG_OUT_DIR", RESULTS_DIR)
LOG_RHO_GUARD = np.log(1e-11)   # -25.33
LOG_RHO_WARN = -20.0


def chain_checks(df, region, cutoff_date, burn_in):
    """Per-chain screening: low log rho (parameter 'theta') and stuck chains."""
    rows = []
    for chain, g in df.groupby("chain"):
        th = g[g.parameter == "theta"].sort_values("iteration")
        v, it = th.value.values, th.iteration.values
        post = v[it >= burn_in]
        # A chain with acceptance ~0.23 essentially never repeats a state over
        # PARAM_THIN iterations; repeated consecutive thinned draws mean it is stuck.
        frac_repeat = float(np.mean(np.diff(post) == 0)) if len(post) > 1 else np.nan
        rows.append({
            "region": region, "cutoff_date": cutoff_date, "chain": chain,
            "min_log_rho_all": v.min(), "min_log_rho_post": post.min() if len(post) else np.nan,
            "q025_log_rho_post": np.quantile(post, 0.025) if len(post) else np.nan,
            "frac_post_below_warn": float(np.mean(post < LOG_RHO_WARN)) if len(post) else np.nan,
            "frac_all_below_guard": float(np.mean(v < LOG_RHO_GUARD)),
            "frac_repeated_draws_post": frac_repeat,
        })
    out = pd.DataFrame(rows)
    # invalid: chain reached the unreliable region or is stuck -> rerun with a new seed
    out["flag_invalid"] = (out.frac_all_below_guard > 0) | (out.frac_repeated_draws_post > 0.01)
    # low_rho: >0.1% of post-burn-in draws below -20 (posterior tail approaching the guard)
    out["flag_low_rho"] = out.frac_post_below_warn > 0.001
    return out


def compute_rhat_ess(region, cutoff_date):
    """Compute Rhat and ESS for all parameters from a specific region/cutoff combination"""
    
    # Load parameter CSV file
    params_filename = f"params_{region}_{cutoff_date.isoformat()}.csv"
    params_path = os.path.join(IN_DIR, params_filename)
    
    if not os.path.exists(params_path):
        print(f"Missing parameter file: {params_filename}")
        return None
    
    df = pd.read_csv(params_path)
    
    # Remove burn-in: ignore first 1,000,000 iterations (numbering before thinning)
    BURN_IN_ITERATIONS = BURN_IN
    df_post_burnin = df[df['iteration'] >= BURN_IN_ITERATIONS].copy()
    
    if len(df_post_burnin) == 0:
        print(f"Warning: No samples remain after burn-in removal for {params_filename}")
        return None

    checks = chain_checks(df, region, cutoff_date, BURN_IN_ITERATIONS)
    
    # Get unique parameters and chains
    parameters = df_post_burnin['parameter'].unique()
    chains = df_post_burnin['chain'].unique()
    
    diagnostics = []
    
    for param in parameters:
        param_data = df_post_burnin[df_post_burnin['parameter'] == param]
        
        # Create chain x iteration array for this parameter
        chain_arrays = []
        for chain in chains:
            chain_data = param_data[param_data['chain'] == chain]
            if len(chain_data) > 0:
                chain_arrays.append(chain_data['value'].values)
        
        if len(chain_arrays) < 2:
            print(f"Warning: Less than 2 chains for parameter {param}")
            continue
        
        # Convert to arviz format (chain, draw)
        param_array = np.array(chain_arrays)
        
        # Compute Rhat and ESS
        rhat = az.rhat(param_array)
        ess_bulk = az.ess(param_array, method="bulk")
        ess_tail = az.ess(param_array, method="tail")
        
        diagnostics.append({
            'region': region,
            'cutoff_date': cutoff_date,
            'parameter': param,
            'rhat': float(rhat),
            'ess_bulk': float(ess_bulk),
            'ess_tail': float(ess_tail),
            'n_chains': len(chain_arrays),
            'n_samples_per_chain': param_array.shape[1]
        })
    
    return pd.DataFrame(diagnostics), checks

def main():
    n_jobs = len(REGIONS) * len(cutoff_dates)
    print(f"Computing diagnostics for all jobs (0-{n_jobs - 1})...")
    
    num_regions = len(REGIONS)
    num_cutoffs = len(cutoff_dates)
    
    all_diagnostics = []
    all_checks = []
    
    # Loop through all jobs 0-89
    for JOB_NUM in range(n_jobs):
        # Use same job mapping as posterior_as_csv.py
        region_idx = JOB_NUM % num_regions  
        cutoff_idx = (JOB_NUM // num_regions) % num_cutoffs
        
        run_region = REGIONS[region_idx]
        run_cutoff_date = cutoff_dates[cutoff_idx]
        
        # Compute diagnostics
        result = compute_rhat_ess(run_region, run_cutoff_date)

        if result is not None:
            diagnostics_df, checks_df = result
            all_diagnostics.append(diagnostics_df)
            all_checks.append(checks_df)
        else:
            print(f"  Failed to compute diagnostics for job {JOB_NUM}")
        
        if (JOB_NUM + 1) % 10 == 0:
            print(f"Completed {JOB_NUM+1} of {n_jobs} jobs")
    
    if all_diagnostics:
        # Combine all diagnostics
        combined_diagnostics = pd.concat(all_diagnostics, ignore_index=True)
        
        # Save combined results
        output_path = os.path.join(OUT_DIR, "all_diagnostics.csv")
        combined_diagnostics.to_csv(output_path, index=False)
        print(f"\nSaved all diagnostics to all_diagnostics.csv")
        
        # Print overall summary
        print("\nOverall Diagnostic Summary:")
        print(f"Total parameters analyzed: {len(combined_diagnostics)}")
        print(f"Total combinations: {combined_diagnostics[['region', 'cutoff_date']].drop_duplicates().shape[0]}")
        print(f"Parameters with Rhat > 1.01: {(combined_diagnostics['rhat'] > 1.01).sum()}")
        print(f"Parameters with ESS bulk < 400: {(combined_diagnostics['ess_bulk'] < 400).sum()}")
        print(f"Parameters with ESS tail < 400: {(combined_diagnostics['ess_tail'] < 400).sum()}")
        
        # Print worst parameters
        print(f"\nWorst Rhat values:")
        worst_rhat = combined_diagnostics.nlargest(5, 'rhat')[['region', 'cutoff_date', 'parameter', 'rhat']]
        print(worst_rhat.to_string(index=False))
        
        print(f"\nLowest ESS bulk values:")
        worst_ess = combined_diagnostics.nsmallest(5, 'ess_bulk')[['region', 'cutoff_date', 'parameter', 'ess_bulk']]
        print(worst_ess.to_string(index=False))

        # Low-rho / stuck-chain screening
        checks = pd.concat(all_checks, ignore_index=True)
        checks.to_csv(os.path.join(OUT_DIR, "chain_checks.csv"), index=False)
        print(f"\nChain checks (saved to chain_checks.csv): {len(checks)} chains")
        print(f"Chains ever below log rho guard ({LOG_RHO_GUARD:.2f}): {(checks.frac_all_below_guard > 0).sum()}")
        print(f"Chains with post-burn-in draws below {LOG_RHO_WARN}: {(checks.frac_post_below_warn > 0).sum()}")
        print(f"Chains with >1% repeated thinned draws (stuck): {(checks.frac_repeated_draws_post > 0.01).sum()}")
        print(f"Lowest post-burn-in log rho: {checks.min_log_rho_post.min():.2f}")
        for col, label in [("flag_invalid", "INVALID CHAINS (rerun with SEIR_RETRY)"),
                           ("flag_low_rho", "LOW-RHO WARNING (>0.1% of draws below -20)")]:
            flagged = checks[checks[col]]
            print(f"{label}: {len(flagged)}")
            if len(flagged):
                print(flagged.to_string(index=False))
    else:
        print("No diagnostics computed successfully")

if __name__ == "__main__":
    main()