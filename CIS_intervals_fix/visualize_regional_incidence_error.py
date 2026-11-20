from config import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from incidence_utils import (
    load_predictive_files_by_cutoff, process_data_with_gc, 
    get_gold_standard_date, format_cutoff_date, save_plot_outputs, 
    convert_cutoff_dates_for_plotting
)

def normalise_region_names(in_names):
    """Normalize region names to match predictive data"""
    if isinstance(in_names, str):
        in_names = [in_names]
    
    out = []
    for name in in_names:
        # Replace underscores with spaces and remove " England"
        normalized = name.replace("_", " ").replace(" England", "")
        
        # Case matching
        region_mapping = {
            "1_NE": "North East", "North East": "North East",
            "2_NW": "North West", "North West": "North West", 
            "3_YH": "Yorkshire", "Yorkshire and the Humber": "Yorkshire", "Yorkshire": "Yorkshire",
            "4_EM": "East Midlands", "East Midlands": "East Midlands",
            "5_WM": "West Midlands", "West Midlands": "West Midlands",
            "6_EE": "East of England", "East of England": "East of England", "East": "East of England", "East of": "East of England",
            "7_LD": "London", "London": "London",
            "8_SE": "South East", "South East": "South East", 
            "9_SW": "South West", "South West": "South West",
            "0_Eng": "England", "England": "England"
        }
        
        if normalized in region_mapping:
            out.append(region_mapping[normalized])
        else:
            # Try original name as fallback
            out.append(normalized)
    
    return out[0] if len(out) == 1 else out

def normalise_age_groups(in_names):
    """Normalize age groups to match between datasets"""
    if isinstance(in_names, str):
        in_names = [in_names]
    
    out = []
    for name in in_names:
        # Handle predictive data format like "[0,11)", "[11,16)", etc.
        if name.startswith("[") and name.endswith(")"):
            age_mapping = {
                "[0,11)": "0–11",
                "[11,16)": "12–16", 
                "[16,25)": "17–25",
                "[25,50)": "26–50",
                "[50,70)": "51–70",
                "[70,+)": "71+"
            }
            
            if name in age_mapping:
                out.append(age_mapping[name])
            else:
                out.append(name)
        
        # Handle poststrat format like "(-Inf,1]", "(1,11]", etc.
        elif name.startswith("(") and name.endswith("]"):
            # Extract the numbers
            inner = name[1:-1]  # Remove outer parentheses
            parts = inner.split(",")
            
            if len(parts) == 2:
                start, end = parts
                start = start.strip()
                end = end.strip()
                
                # Map poststrat intervals to standard age groups
                poststrat_mapping = {
                    "(-Inf,1]": "0–11",
                    "(1,11]": "0–11",
                    "(11,16]": "12–16",
                    "(16,25]": "17–25",
                    "(25,50]": "26–50", 
                    "(50,70]": "51–70",
                    "(70,Inf]": "71+",
                    "(70, Inf]": "71+"  # Handle space after comma
                }
                
                if name in poststrat_mapping:
                    out.append(poststrat_mapping[name])
                else:
                    out.append(name)
            else:
                out.append(name)
        else:
            # Already in target format or unknown format
            out.append(name)
    
    return out[0] if len(out) == 1 else out

def load_poststrat_table():
    """Load and process post-stratification population weights"""
    print("Loading post-stratification table...")
    
    poststrat_path = os.path.join(FILE_BASE, "data", "poststrat.csv")
    df = pd.read_csv(poststrat_path)
    
    # Filter out Wales, Scotland, Northern Ireland
    england_regions = df[~df['Region_Name'].isin(['Wales', 'Scotland', 'Northern_Ireland'])].copy()
    
    # Apply normalization functions
    england_regions['region_normalized'] = england_regions['Region_Name'].apply(normalise_region_names)
    england_regions['age_normalized'] = england_regions['age_group'].apply(normalise_age_groups)
    
    # Aggregate population by normalized region and age group
    pop_weights = england_regions.groupby(['region_normalized', 'age_normalized'])['pop'].sum().reset_index()
    pop_weights.columns = ['region', 'age', 'pop']
    
    print(f"Loaded population weights for {len(pop_weights)} region-age combinations")
    print("Sample region names:", pop_weights['region'].unique()[:5])
    print("Sample age groups:", pop_weights['age'].unique())
    return pop_weights

def process_single_cutoff_summary(cutoff_data, cutoff_date, pop_weights):
    """Process loaded cutoff data with population weighting and summary computation"""
    # Apply normalization to predictive data to match population weights
    cutoff_data['region'] = cutoff_data['region'].apply(normalise_region_names)
    cutoff_data['age'] = cutoff_data['age'].apply(normalise_age_groups)
    
    # Apply population-weighted aggregation by region
    regional_data = poststratify_incidence_by_region(cutoff_data, pop_weights)
    
    # Compute summary statistics
    summary = compute_summary_statistics_single_cutoff_by_region(regional_data, cutoff_date)
    
    print(f"  Cutoff {cutoff_date} processed: {len(summary)} summary rows")
    return summary

def poststratify_incidence_by_region(data, pop_weights):
    """Apply population-weighted aggregation to get region-specific incidence"""
    print("Applying population-weighted aggregation by region...")
    
    # Debug: print unique values before merge
    print(f"Data regions: {sorted(data['region'].unique())}")
    print(f"Data age groups: {sorted(data['age'].unique())}")
    print(f"Pop weights regions: {sorted(pop_weights['region'].unique())}")
    print(f"Pop weights age groups: {sorted(pop_weights['age'].unique())}")
    
    # Check original data length before merge
    original_data_length = len(data)
    
    # Merge with population weights
    merged = data.merge(pop_weights, on=['region', 'age'], how='inner')
    
    if len(merged) == 0:
        # Additional debugging
        print("No matches found. Checking for partial matches...")
        region_matches = set(data['region'].unique()) & set(pop_weights['region'].unique())
        age_matches = set(data['age'].unique()) & set(pop_weights['age'].unique())
        print(f"Matching regions: {region_matches}")
        print(f"Matching age groups: {age_matches}")
        raise ValueError("No matches found between data and population weights")
    
    # Check if any rows were dropped during merge
    if len(merged) < original_data_length:
        dropped_rows = original_data_length - len(merged)
        print(f"ERROR: {dropped_rows} rows were dropped during merge!")
        
        # Find which region-age combinations from data are missing in pop_weights
        data_combinations = set(zip(data['region'], data['age']))
        pop_combinations = set(zip(pop_weights['region'], pop_weights['age']))
        missing_combinations = data_combinations - pop_combinations
        
        print(f"Missing region-age combinations in population weights:")
        for region, age in sorted(missing_combinations):
            print(f"  - {region}, {age}")
        
        raise ValueError(f"Merge dropped {dropped_rows} rows - some region-age combinations are missing from population weights")
    
    print(f"Successfully merged {len(merged)} rows")
    
    # Calculate weighted incidence (incidence * population) for each region-age combination
    merged['weighted_incidence'] = merged['incidence'] * merged['pop']
    
    # Group by day, iteration, cutoff_date, AND region to aggregate across age groups within each region
    # This is the key post-stratification step: sum weighted incidences and populations across ages
    aggregated = merged.groupby(['day', 'iteration', 'cutoff_date', 'region']).agg({
        'pop': 'sum',  # Total population for this region (sum across age groups)
        'weighted_incidence': 'sum'  # Total weighted incidence for this region (sum across age groups)
    }).reset_index()
    
    # Calculate region-specific incidence rate: total weighted incidence / total population
    aggregated['region_incidence'] = aggregated['weighted_incidence'] / aggregated['pop']
    
    print(f"Aggregated to {len(aggregated)} region-specific observations")
    return aggregated

def compute_summary_statistics_single_cutoff_by_region(aggregated_data, cutoff_date):
    """Compute summary statistics (median) for each region in a single cutoff"""
    
    # Group by day AND region to get median across iterations
    summary = aggregated_data.groupby(['day', 'region'])['region_incidence'].agg([
        ('median', 'median')
    ]).reset_index()
    
    # Add cutoff date
    summary['cutoff_date'] = cutoff_date
    
    return summary

def load_and_process_data(pop_weights):
    """Load and process data one cutoff at a time to minimize memory usage"""
    print("Processing data cutoff by cutoff...")
    
    def data_processor(cutoff_data, cutoff_date):
        return process_single_cutoff_summary(cutoff_data, cutoff_date, pop_weights)
    
    final_summary = process_data_with_gc(
        load_predictive_files_by_cutoff,
        data_processor,
        cutoff_dates
    )
    
    # Convert cutoff_date back to date objects for plotting
    return convert_cutoff_dates_for_plotting(final_summary)

def calculate_relative_absolute_error(summary_data):
    """Calculate relative absolute error compared to gold standard"""
    print("Calculating relative absolute error...")
    
    gold_standard_date = get_gold_standard_date()
    
    # Get gold standard data
    gold_data = summary_data[summary_data['cutoff_date'] == gold_standard_date].copy()
    gold_data = gold_data.rename(columns={'median': 'gold_median'})
    gold_data = gold_data[['day', 'region', 'gold_median']]
    
    # Get non-gold standard data
    other_data = summary_data[summary_data['cutoff_date'] != gold_standard_date].copy()
    
    # Merge to get gold standard reference for each estimate
    error_data = other_data.merge(gold_data, on=['day', 'region'], how='inner')
    
    # Calculate relative absolute error: |estimate - gold_standard| / |gold_standard|
    error_data['relative_absolute_error'] = np.abs(error_data['median'] - error_data['gold_median']) / np.abs(error_data['gold_median'])
    
    print(f"Calculated relative absolute error for {len(error_data)} observations")
    return error_data

def create_regional_error_plot(error_data):
    """Create grid plot with regions as columns and cutoff dates as rows showing relative absolute error"""
    print("Creating regional error visualization...")
    
    cutoff_list = sorted(error_data['cutoff_date'].unique())
    region_list = sorted(error_data['region'].unique())
    
    print(f"Creating plot for {len(region_list)} regions and {len(cutoff_list)} cutoffs")
    print(f"Regions: {region_list}")
    
    # Set up the figure - regions as columns, cutoffs as rows
    n_regions = len(region_list)
    n_cutoffs = len(cutoff_list)
    
    fig, axes = plt.subplots(n_cutoffs, n_regions, 
                            figsize=(3*n_regions, 2.5*n_cutoffs), 
                            sharex=True, sharey=True)
    
    # Handle single row/column cases
    if n_cutoffs == 1:
        axes = axes.reshape(1, -1)
    if n_regions == 1:
        axes = axes.reshape(-1, 1)
    if n_cutoffs == 1 and n_regions == 1:
        axes = np.array([[axes]])
    
    for cutoff_idx, cutoff_date in enumerate(cutoff_list):
        for region_idx, region in enumerate(region_list):
            ax = axes[cutoff_idx, region_idx]
            
            # Get data for this cutoff and region
            cutoff_region_data = error_data[
                (error_data['cutoff_date'] == cutoff_date) & 
                (error_data['region'] == region)
            ].copy()
            cutoff_region_data = cutoff_region_data.sort_values('day')
            
            if len(cutoff_region_data) > 0:
                # Plot relative absolute error
                ax.plot(cutoff_region_data['day'], 
                       cutoff_region_data['relative_absolute_error'], 
                       color='red', linewidth=1.5)
            
            # Add vertical line at cutoff date
            ax.axvline(pd.Timestamp(cutoff_date), color='gray', 
                      linestyle=':', alpha=0.5, linewidth=0.8)
            
            # Formatting
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_ylim(bottom=0)  # Start y-axis at 0
            
            # Add region name as column header (only for top row)
            if cutoff_idx == 0:
                ax.set_title(region, fontsize=10, fontweight='bold')
            
            # Add cutoff date as row label (only for first column)
            if region_idx == 0:
                ax.text(-0.15, 0.5, cutoff_date.strftime("%Y-%m-%d"), 
                       transform=ax.transAxes, rotation=90, 
                       va='center', ha='center', fontsize=9, fontweight='bold')
    
    # Overall formatting
    fig.suptitle('Relative Absolute Error in Daily Incidence Estimates by Region and Cutoff Date', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add common labels
    fig.text(0.5, 0.02, 'Date', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.5, 'Relative Absolute Error (%)', va='center', rotation='vertical', 
             fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Relative Absolute Error'),
        plt.Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='Cutoff Date')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
              ncol=3, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.98, 
                       hspace=0.15, wspace=0.1)
    
    return fig

def main():
    """Main function to create regional error visualization"""
    print("Starting regional incidence error analysis...")
    
    # Load population weights
    pop_weights = load_poststrat_table()
    
    # Load and process data one cutoff at a time
    summary_stats = load_and_process_data(pop_weights)
    
    # Calculate relative absolute error
    error_data = calculate_relative_absolute_error(summary_stats)
    
    # Create visualization
    fig = create_regional_error_plot(error_data)
    
    # Save outputs
    save_plot_outputs(fig, "regional_incidence_error")
    
    plt.close()
    print("Regional incidence error visualization complete!")

if __name__ == "__main__":
    main()