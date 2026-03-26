from config import *
import pandas as pd
import numpy as np
import os
import gzip
from incidence_utils import (
    load_predictive_files_by_cutoff, process_data_with_gc, 
    compute_summary_statistics, convert_cutoff_dates_for_plotting
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
            out.append(normalized)
    
    return out[0] if len(out) == 1 else out

def normalise_age_groups(in_names):
    """Normalize age groups to match between datasets"""
    if isinstance(in_names, str):
        in_names = [in_names]
    
    out = []
    for name in in_names:
        if name.startswith("[") and name.endswith(")"):
            age_mapping = {
                "[0,11)": "0–11",
                "[11,16)": "12–16", 
                "[16,25)": "17–25",
                "[25,50)": "26–50",
                "[50,70)": "51–70",
                "[70,+)": "71+"
            }
            out.append(age_mapping.get(name, name))
        elif name.startswith("(") and name.endswith("]"):
            poststrat_mapping = {
                "(-Inf,1]": "0–11",
                "(1,11]": "0–11",
                "(11,16]": "12–16",
                "(16,25]": "17–25",
                "(25,50]": "26–50", 
                "(50,70]": "51–70",
                "(70,Inf]": "71+",
                "(70, Inf]": "71+"
            }
            out.append(poststrat_mapping.get(name, name))
        else:
            out.append(name)
    
    return out[0] if len(out) == 1 else out

def load_poststrat_table():
    """Load and process post-stratification population weights"""
    print("Loading post-stratification table...")
    poststrat_path = os.path.join(FILE_BASE, "data", "poststrat.csv")
    df = pd.read_csv(poststrat_path)
    england_regions = df[~df['Region_Name'].isin(['Wales', 'Scotland', 'Northern_Ireland'])].copy()
    england_regions['region_normalized'] = england_regions['Region_Name'].apply(normalise_region_names)
    england_regions['age_normalized'] = england_regions['age_group'].apply(normalise_age_groups)
    pop_weights = england_regions.groupby(['region_normalized', 'age_normalized'])['pop'].sum().reset_index()
    pop_weights.columns = ['region', 'age', 'pop']
    return pop_weights

def process_single_cutoff(cutoff_data, cutoff_date, pop_weights):
    """Process loaded cutoff data for both England and Regional summaries"""
    # Normalize
    cutoff_data['region'] = cutoff_data['region'].apply(normalise_region_names)
    cutoff_data['age'] = cutoff_data['age'].apply(normalise_age_groups)
    
    # Merge with weights
    merged = cutoff_data.merge(pop_weights, on=['region', 'age'], how='inner')
    merged['weighted_incidence'] = merged['incidence'] * merged['pop']
    
    # Regional Aggregation
    regional_agg = merged.groupby(['day', 'iteration', 'cutoff_date', 'region']).agg({
        'pop': 'sum',
        'weighted_incidence': 'sum'
    }).reset_index()
    regional_agg['incidence'] = regional_agg['weighted_incidence'] / regional_agg['pop']
    
    regional_summary = compute_summary_statistics(regional_agg, ['day', 'region', 'cutoff_date'])
    
    # England Aggregation
    england_agg = merged.groupby(['day', 'iteration', 'cutoff_date']).agg({
        'pop': 'sum',
        'weighted_incidence': 'sum'
    }).reset_index()
    england_agg['incidence'] = england_agg['weighted_incidence'] / england_agg['pop']
    england_agg['region'] = 'England'
    
    england_summary = compute_summary_statistics(england_agg, ['day', 'region', 'cutoff_date'])
    
    # Combine
    combined_summary = pd.concat([regional_summary, england_summary], ignore_index=True)
    print(f"  Cutoff {cutoff_date} processed: {len(combined_summary)} summary rows")
    return combined_summary

def main():
    print("Starting export of summary statistics...")
    pop_weights = load_poststrat_table()
    
    def data_processor(cutoff_data, cutoff_date):
        return process_single_cutoff(cutoff_data, cutoff_date, pop_weights)
    
    final_summary = process_data_with_gc(
        load_predictive_files_by_cutoff,
        data_processor,
        cutoff_dates
    )
    
    # Save as gzipped CSV
    # 1. Setup output directory
    output_dir = "for_paper"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    output_filename = "incidence_summary_statistics.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Writing output to {output_path}...")
    final_summary.to_csv(output_path, index=False)
    
    print("Export complete!")

if __name__ == "__main__":
    main()
