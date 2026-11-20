"""
Common utilities for incidence data processing and visualization
"""
from config import *
import pandas as pd
import numpy as np
import os


# Constants
BURN_IN_ITERATIONS = 1000000


def load_csv_with_burnin(filepath, cutoff_date=None):
    """Load CSV file and apply burn-in filter with proper date handling"""
    if not os.path.exists(filepath):
        return None
    
    df = pd.read_csv(filepath)
    
    # Apply burn-in filter
    df_filtered = df[df['iteration'] >= BURN_IN_ITERATIONS].copy()
    
    if len(df_filtered) == 0:
        return None
    
    # Convert day column to datetime
    df_filtered['day'] = pd.to_datetime(df_filtered['day'])
    
    # Add cutoff_date if provided
    if cutoff_date is not None:
        df_filtered['cutoff_date'] = cutoff_date
    
    return df_filtered


def compute_summary_statistics(data, group_columns, value_column='incidence'):
    """Compute median and 95% credible intervals for grouped data"""
    grouped = data.groupby(group_columns)[value_column]
    
    summary = grouped.agg([
        ('median', 'median'),
        ('q025', lambda x: np.percentile(x, 2.5)),
        ('q975', lambda x: np.percentile(x, 97.5))
    ]).reset_index()
    
    return summary


def load_predictive_files_by_region(region):
    """Load all cutoff files for a single region"""
    print(f"Processing region: {region}")
    
    region_data = []
    
    for cutoff_date in cutoff_dates:
        filename = f"predictive_{region}_{cutoff_date.isoformat()}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        df = load_csv_with_burnin(filepath, cutoff_date)
        if df is not None:
            print(f"  Loading {filename}...")
            region_data.append(df)
        else:
            print(f"  Warning: Missing or empty file {filename}")
    
    if not region_data:
        return None
    
    # Combine data for this region
    region_combined = pd.concat(region_data, ignore_index=True)
    print(f"  Region {region} loaded: {len(region_combined)} rows")
    
    return region_combined


def load_predictive_files_by_cutoff(cutoff_date):
    """Load all region files for a single cutoff date"""
    print(f"Processing cutoff date: {cutoff_date}")
    
    cutoff_data = []
    
    for region in REGIONS:
        filename = f"predictive_{region}_{cutoff_date.isoformat()}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        
        df = load_csv_with_burnin(filepath, cutoff_date)
        if df is not None:
            cutoff_data.append(df)
        else:
            print(f"  Warning: Missing or empty file {filepath}")
    
    if not cutoff_data:
        return None
    
    # Combine data for this cutoff
    cutoff_combined = pd.concat(cutoff_data, ignore_index=True)
    print(f"  Cutoff {cutoff_date} loaded: {len(cutoff_combined)} rows")
    
    return cutoff_combined


def save_plot_outputs(fig, base_filename, results_dir=None):
    """Save figure in both PNG and PDF formats"""
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    output_png = os.path.join(results_dir, f"{base_filename}.png")
    output_pdf = os.path.join(results_dir, f"{base_filename}.pdf")
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_pdf, bbox_inches='tight')
    
    print(f"Saved visualization to:")
    print(f"  {output_png}")
    print(f"  {output_pdf}")
    
    return output_png, output_pdf


def process_data_with_gc(data_loader, data_processor, items):
    """Generic function to process data items with garbage collection"""
    import gc
    
    all_results = []
    
    for item in items:
        # Load data
        data = data_loader(item)
        
        if data is not None:
            # Process data
            result = data_processor(data, item)
            if result is not None:
                all_results.append(result)
        
        # Force garbage collection
        gc.collect()
    
    if not all_results:
        raise ValueError("No data processed successfully")
    
    # Combine all results
    final_result = pd.concat(all_results, ignore_index=True)
    print(f"Total processed data: {len(final_result)} rows")
    
    return final_result


def get_gold_standard_date():
    """Get the latest cutoff date (gold standard)"""
    return max(cutoff_dates)


def setup_plot_colors(cutoff_list):
    """Setup color scheme for cutoff plots"""
    import matplotlib.pyplot as plt
    
    # Exclude gold standard from color mapping
    n_cutoffs = len(cutoff_list) - 1
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_cutoffs)) if n_cutoffs > 0 else []
    
    return colors


def format_cutoff_date(cutoff_date):
    """Format cutoff date for display"""
    return cutoff_date.strftime("%m-%d")


def convert_cutoff_dates_for_plotting(df):
    """Convert cutoff_date column to date objects for plotting"""
    df_copy = df.copy()
    df_copy['cutoff_date'] = pd.to_datetime(df_copy['cutoff_date']).dt.date
    return df_copy