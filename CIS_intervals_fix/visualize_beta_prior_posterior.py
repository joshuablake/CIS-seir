from config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
from incidence_utils import save_plot_outputs

BURN_IN_ITERATIONS = 1000000

def load_beta0_posterior_data_by_region():
    """Load beta[0] posterior samples from gold standard run by region"""
    print("Loading beta[0] posterior data from gold standard run by region...")
    
    # Gold standard cutoff date
    gold_standard_date = max(cutoff_dates)
    print(f"Gold standard date: {gold_standard_date}")
    
    # Dictionary to store beta[0] samples by region
    region_beta0_samples = {}
    
    # Load data from all regions and chains
    for region in REGIONS:
        region_samples = []
        
        # Construct filename
        params_file = os.path.join(
            RESULTS_DIR,
            f"params_{region}_{gold_standard_date}.csv"
        )
        
        if not os.path.exists(params_file):
            raise ValueError(f"No parameter file {params_file} for {region}")
        # Load CSV file
        df = pd.read_csv(params_file)
        
        # Filter for beta[0] parameter
        region_samples = np.exp(df[
            (df['parameter'] == 'beta[0]') &
            (df['iteration'] >= BURN_IN_ITERATIONS)
        ]['value'].values)
        
        if len(region_samples) <= 0:
            raise ValueError(f"No samples for {region} in {params_file}")

        region_beta0_samples[region] = np.array(region_samples)
        print(f"{region}: {len(region_samples)} samples")
    
    return region_beta0_samples

def get_beta0_prior_distribution():
    """Get the prior distribution for beta[0] from config"""
    print("Getting beta[0] prior distribution...")
    
    # From config.py, beta prior is: LogGaussianRWNonCentred(stats.expon(scale=1/10), ...)
    # The first parameter (stats.expon(scale=1/10)) is the prior for beta[0]
    # This is an exponential distribution with rate=10 (scale=1/10)
    prior_dist = stats.expon(scale=1/10)
    
    print("Prior: Exponential(rate=10)")
    return prior_dist

def create_regional_prior_posterior_plot(region_posterior_dict, prior_dist):
    """Create grid visualization comparing prior and posterior distributions for beta[0] by region"""
    print("Creating regional prior vs posterior visualization...")
    
    # Set up the plot grid - 3 columns, 3 rows for 9 regions
    n_regions = len(region_posterior_dict)
    n_cols = 3
    n_rows = int(np.ceil(n_regions / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows), 
                            sharex=True, sharey=False)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Determine overall x-range for consistent scaling
    all_samples = np.concatenate(list(region_posterior_dict.values()))
    x_min = min(0, np.min(all_samples))
    x_max = max(0.5, np.max(all_samples))
    x = np.linspace(x_min, x_max, 1000)
    
    # Plot prior (same for all regions)
    prior_pdf = prior_dist.pdf(x)
    
    region_names = list(region_posterior_dict.keys())
    
    for idx, region in enumerate(region_names):
        ax = axes_flat[idx]
        posterior_samples = region_posterior_dict[region]
        
        # Plot prior distribution
        ax.plot(x, prior_pdf, 'b-', linewidth=2, label='Prior: Exp(rate=10)', alpha=0.8)
        ax.fill_between(x, prior_pdf, alpha=0.2, color='blue')
        
        # Plot posterior distribution (histogram + KDE)
        ax.hist(posterior_samples, bins=30, density=True, alpha=0.6, color='red', 
                label=f'Posterior (n={len(posterior_samples):,})')
        
        # Add KDE for smoother posterior curve
        if len(posterior_samples) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(posterior_samples)
            kde_x = np.linspace(np.min(posterior_samples), np.max(posterior_samples), 200)
            kde_y = kde(kde_x)
            ax.plot(kde_x, kde_y, 'r-', linewidth=2, alpha=0.8)
        
        # Add summary statistics
        post_mean = np.mean(posterior_samples)
        prior_mean = prior_dist.mean()
        
        # Add vertical lines for means
        ax.axvline(prior_mean, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(post_mean, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_title(f'{region.replace("_", " ")}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        
        # Add summary text
        post_median = np.median(posterior_samples)
        stats_text = f'Mean: {post_mean:.3f}\nMedian: {post_median:.3f}'
        ax.text(0.65, 0.85, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
                fontsize=8, verticalalignment='top')
        
        # Add legend to first subplot only
        if idx == 0:
            ax.legend(fontsize=9, loc='upper left')
    
    # Hide unused subplots
    for idx in range(n_regions, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Overall formatting
    fig.suptitle('Prior vs Posterior Distribution for β[0] Parameter by Region\n(Gold Standard Run)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add common labels
    fig.text(0.5, 0.02, 'β[0] (log transmission rate)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', 
             fontsize=12, fontweight='bold')
    
    # Add overall legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Prior: Exponential(rate=10)'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Posterior'),
        plt.Line2D([0], [0], color='blue', linestyle='--', alpha=0.7, label='Prior Mean'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Posterior Mean')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
              ncol=4, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.98, 
                       hspace=0.3, wspace=0.2)
    
    return fig

def main():
    """Main function to create beta[0] prior vs posterior visualization by region"""
    print("Starting beta[0] prior vs posterior analysis by region...")
    
    # Load posterior samples by region
    region_posterior_dict = load_beta0_posterior_data_by_region()
    
    if len(region_posterior_dict) == 0:
        print("Error: No posterior samples found!")
        return
    
    # Get prior distribution
    prior_dist = get_beta0_prior_distribution()
    
    # Create visualization
    fig = create_regional_prior_posterior_plot(region_posterior_dict, prior_dist)
    
    # Save outputs
    save_plot_outputs(fig, "beta0_prior_posterior_by_region", results_dir=".")
    
    plt.close()
    print("Beta[0] prior vs posterior by region visualization complete!")

if __name__ == "__main__":
    main()
