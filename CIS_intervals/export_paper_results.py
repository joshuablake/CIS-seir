from config import *
import pandas as pd
import os
from incidence_utils import load_csv_with_burnin, get_gold_standard_date

def main():
    print("Starting final results export for paper...")
    
    # 1. Setup output directory
    output_dir = "for_paper"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. Identify the final cut-off date (Gold Standard)
    final_cutoff = get_gold_standard_date()
    final_cutoff_str = final_cutoff.isoformat()
    print(f"Final cut-off date identified: {final_cutoff_str}")

    # 3. Define the result types to process
    result_types = ['predictive', 'final_state', 'params']
    
    for r_type in result_types:
        print(f"Combining {r_type} files...")
        combined_data = []
        
        for region in REGIONS:
            filename = f"{r_type}_{region}_{final_cutoff_str}.csv"
            filepath = os.path.join(RESULTS_DIR, filename)
            
            if os.path.exists(filepath):
                # Load and filter for burn-in (>= 1,000,000 iterations)
                df = pd.read_csv(filepath)
                if df is not None:
                    combined_data.append(df)
                    print(f"  Loaded {filename}")
            else:
                print(f"  Warning: File not found: {filepath}")
        
        if combined_data:
            # Combine all regions for this result type
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Save to gzipped CSV
            output_path = os.path.join(output_dir, f"{r_type}.csv.gz")
            print(f"  Writing gzipped CSV to {output_path}...")
            combined_df.to_csv(output_path, index=False, compression='gzip')
        else:
            print(f"  No data found for {r_type}")

    print("Export complete!")

if __name__ == "__main__":
    main()
