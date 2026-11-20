from config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_beta0_traceplots():
    """Create traceplots for beta[0] parameters across all region/cutoff combinations"""
    
    # Set up figure grid - optimized for 90 combinations (9 regions x 10 cutoffs)
    num_regions = len(REGIONS)
    num_cutoffs = len(cutoff_dates)
    
    # Create a very large figure to accommodate all combinations
    fig, axes = plt.subplots(num_cutoffs, num_regions, 
                            figsize=(2*num_regions, 1.5*num_cutoffs),
                            sharex=False, sharey=False)
    
    # Handle case where we only have one row or column
    if num_cutoffs == 1:
        axes = axes.reshape(1, -1)
    elif num_regions == 1:
        axes = axes.reshape(-1, 1)
    
    # Color palette for chains
    colors = plt.cm.Set1(np.linspace(0, 1, 10))  # Up to 10 different colors
    
    BURN_IN_ITERATIONS = 1000000
    
    for cutoff_idx, cutoff_date in enumerate(cutoff_dates):
        for region_idx, region in enumerate(REGIONS):
            ax = axes[cutoff_idx, region_idx]
            
            # Load parameter file for this combination
            params_filename = f"params_{region}_{cutoff_date.isoformat()}.csv"
            params_path = os.path.join(RESULTS_DIR, params_filename)
            
            if not os.path.exists(params_path):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_title(f"{region.replace('_', ' ')}\n{cutoff_date.strftime('%m-%d')}", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            df = pd.read_csv(params_path)
            
            # Filter for beta[0] parameter (include ALL data for burn-in visualization)
            beta0_data = df[df['parameter'] == 'beta[0]'].copy()
            
            if len(beta0_data) == 0:
                ax.text(0.5, 0.5, 'No beta[0]\ndata', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=6)
                ax.set_title(f"{region.replace('_', ' ')}\n{cutoff_date.strftime('%m-%d')}", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Get unique chains for this combination
            chains = sorted(beta0_data['chain'].unique())
            
            # Get iteration range for burn-in shading
            all_iterations = beta0_data['iteration'].values
            min_iter = all_iterations.min()
            max_iter = all_iterations.max()
            
            # Add burn-in shading (background only)
            if min_iter < BURN_IN_ITERATIONS:
                ax.axvspan(min_iter, min(BURN_IN_ITERATIONS, max_iter), 
                          alpha=0.2, color='gray', zorder=0)
            
            # Plot each chain with same color throughout
            for i, chain in enumerate(chains):
                chain_data = beta0_data[beta0_data['chain'] == chain]
                if len(chain_data) > 0:
                    color = colors[i % len(colors)]
                    ax.plot(chain_data['iteration'], chain_data['value'], 
                           color=color, alpha=0.8, linewidth=0.5)
            
            # Formatting - very compact for 90 panels
            ax.set_title(f"{region.replace('_', ' ')}\n{cutoff_date.strftime('%m-%d')}", fontsize=6)
            ax.tick_params(labelsize=4)
            ax.grid(True, alpha=0.2)
            
            # Reduce number of ticks
            ax.locator_params(axis='x', nbins=3)
            ax.locator_params(axis='y', nbins=3)
    
    # Overall formatting
    fig.suptitle('Traceplots for beta[0] Parameter (Gray shading = burn-in < 1M iterations)', 
                 fontsize=12, y=0.995)
    
    # Add common axis labels
    fig.text(0.5, 0.02, 'MCMC Iteration', ha='center', fontsize=10)
    fig.text(0.01, 0.5, 'beta[0] Value', va='center', rotation=90, fontsize=10)
    
    # Adjust layout for very tight spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.06, left=0.04, right=0.98, hspace=0.4, wspace=0.3)
    
    # Save plot with high DPI for readability
    output_path = os.path.join(RESULTS_DIR, "beta0_traceplots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved traceplots to {output_path}")
    
    # Also save as PDF for better quality when zooming
    output_path_pdf = os.path.join(RESULTS_DIR, "beta0_traceplots.pdf")
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Saved traceplots to {output_path_pdf}")
    
    # Don't show plot due to size - just save files
    plt.close()
    
    print(f"Generated traceplots grid: {num_cutoffs} rows x {num_regions} columns = {num_cutoffs * num_regions} panels")

def main():
    print("Creating traceplots for beta[0] parameters...")
    create_beta0_traceplots()
    print("Done!")

if __name__ == "__main__":
    main()