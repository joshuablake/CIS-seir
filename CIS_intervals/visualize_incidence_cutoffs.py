from config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import timedelta

def process_single_region(region):
    """Process all cutoff files for a single region"""
    print(f"Processing region: {region}")
    
    region_data = []
    BURN_IN_ITERATIONS = 1000000
    
    for cutoff_date in cutoff_dates:
        filename = f"predictive_{region}_{cutoff_date.isoformat()}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"  Loading {filename}...")
            df = pd.read_csv(filepath)
            
            # Apply burn-in filter
            df_filtered = df[df['iteration'] >= BURN_IN_ITERATIONS].copy()
            
            if len(df_filtered) > 0:
                df_filtered['day'] = pd.to_datetime(df_filtered['day'])
                region_data.append(df_filtered)
            else:
                print(f"  Warning: No post-burn-in data for {filename}")
        else:
            print(f"  Warning: Missing file {filename}")
    
    if not region_data:
        return None
    
    # Combine data for this region
    region_combined = pd.concat(region_data, ignore_index=True)
    
    # Compute summary statistics for this region
    grouped = region_combined.groupby(['region', 'cutoff_date', 'age', 'day'])['incidence']
    
    region_summary = grouped.agg([
        ('median', 'median'),
        ('q025', lambda x: np.percentile(x, 2.5)),
        ('q975', lambda x: np.percentile(x, 97.5))
    ]).reset_index()
    
    # Convert cutoff_date back to date objects
    region_summary['cutoff_date'] = pd.to_datetime(region_summary['cutoff_date']).dt.date
    
    print(f"  Region {region} processed: {len(region_summary)} summary rows")
    return region_summary

def load_and_process_data():
    """Load and process predictive CSV files one region at a time"""
    print("Processing predictive data region by region...")
    
    all_summaries = []
    
    # Process one region at a time to minimize memory usage
    for region in REGIONS:  # Limited for testing
        region_summary = process_single_region(region)
        
        if region_summary is not None:
            all_summaries.append(region_summary)
        
        # Force garbage collection after each region
        import gc
        gc.collect()
    
    if not all_summaries:
        raise ValueError("No data processed successfully")
    
    # Combine all regional summaries
    final_summary = pd.concat(all_summaries, ignore_index=True)
    print(f"Total processed data: {len(final_summary)} summary rows")
    
    return final_summary

def create_time_series_plot(summary_stats):
    """Create the 6x9 grid plot showing incidence time series"""
    print("Creating time series visualization...")
    
    # Get unique age groups and regions
    age_groups = sorted(summary_stats['age'].unique())
    regions = REGIONS
    
    # Set up the figure
    fig, axes = plt.subplots(len(age_groups), len(regions), 
                            figsize=(18, 12), 
                            sharex=False, sharey='row')
    
    # Color scheme for cutoffs (earliest = red, latest = black for gold standard)
    cutoff_list = sorted(cutoff_dates)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(cutoff_list)-1))  # Exclude gold standard
    gold_standard_date = max(cutoff_dates)
    
    for age_idx, age_group in enumerate(age_groups):
        for region_idx, region in enumerate(regions):
            ax = axes[age_idx, region_idx]
            
            # Filter data for this age group and region
            region_age_data = summary_stats[
                (summary_stats['age'] == age_group) & 
                (summary_stats['region'] == region)
            ].copy()
            
            if len(region_age_data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                continue
            
            # Plot gold standard (full timeline)
            gold_data = region_age_data[region_age_data['cutoff_date'] == gold_standard_date]
            if len(gold_data) > 0:
                gold_data = gold_data.sort_values('day')
                ax.fill_between(gold_data['day'], gold_data['q025'], gold_data['q975'], 
                              alpha=0.2, color='black', label='95% CI (Gold)')
                ax.plot(gold_data['day'], gold_data['median'], 
                       color='black', linewidth=2, label='Gold Standard')
            
            # Plot other cutoffs (final 2 weeks only)
            for cutoff_idx, cutoff_date in enumerate(cutoff_list[:-1]):  # Exclude gold standard
                cutoff_data = region_age_data[region_age_data['cutoff_date'] == cutoff_date]
                
                if len(cutoff_data) > 0:
                    # Filter to final 2 weeks before cutoff
                    cutoff_end = pd.Timestamp(cutoff_date)
                    cutoff_start = cutoff_end - timedelta(days=13)  # 14 days total
                    
                    final_weeks = cutoff_data[
                        (cutoff_data['day'] >= cutoff_start) & 
                        (cutoff_data['day'] <= cutoff_end)
                    ].sort_values('day')
                    
                    if len(final_weeks) > 0:
                        color = colors[cutoff_idx]
                        
                        # Plot confidence interval
                        ax.fill_between(final_weeks['day'], final_weeks['q025'], 
                                      final_weeks['q975'], alpha=0.3, color=color)
                        
                        # Plot median line
                        ax.plot(final_weeks['day'], final_weeks['median'], 
                               color=color, linewidth=1.5, 
                               label=f'Cutoff {cutoff_date.strftime("%m-%d")}')
            
            # Formatting
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            
            # Set title for top row (regions)
            if age_idx == 0:
                ax.set_title(region.replace('_', ' '), fontsize=8, fontweight='bold')
            
            # Set ylabel for leftmost column (age groups)
            if region_idx == 0:
                ax.set_ylabel(f'{age_group}\nIncidence/100k', fontsize=7, fontweight='bold')
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
            
            # Add legend only to first subplot
            if age_idx == 0 and region_idx == 0:
                ax.legend(fontsize=6, loc='upper left')
    
    # Overall formatting
    fig.suptitle('Daily Incidence Estimates: Gold Standard (Full Timeline) vs Recent Estimates (Final 2 Weeks)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Add common x-label
    fig.text(0.5, 0.02, 'Date', ha='center', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.98, hspace=0.3, wspace=0.2)
    
    return fig

def main():
    """Main function to create the visualization"""
    print("Starting incidence cutoff visualization...")
    
    # Load and process data (now includes summary statistics)
    summary_stats = load_and_process_data()
    
    # Create visualization
    fig = create_time_series_plot(summary_stats)
    
    # Save outputs
    output_png = os.path.join(RESULTS_DIR, "incidence_cutoff_comparison.png")
    output_pdf = os.path.join(RESULTS_DIR, "incidence_cutoff_comparison.pdf")
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    
    print(f"Saved visualization to:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")
    
    plt.close()
    print("Visualization complete!")

if __name__ == "__main__":
    main()