import pandas as pd
import numpy as np

# Patch for Sweetviz compatibility with Numpy 2.0+
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = UserWarning

import sweetviz as sv
from ydata_profiling import ProfileReport
import os

def generate_report(output_dir="plots"):
    """
    Generates automated EDA reports using Pandas Profiling (ydata-profiling).
    Sweetviz is currently disabled.
    """
    print("Loading data...")
    try:
        from data_loader import load_data
    except ImportError:
        import sys
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from data_loader import load_data
        
    df = load_data()
    if df is None:
        return
    
    # Create a copy for profiling
    df_vis = df.copy()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sweetviz analysis
    print("Generating Sweetviz report...")
    sweetviz_report = sv.analyze(df)
    sweetviz_path = os.path.join(output_dir, "sweetviz_report.html")
    sweetviz_report.show_html(sweetviz_path)
    print(f"Sweetviz report saved to {sweetviz_path}")
    
    # Pandas Profiling
    print("Generating Pandas Profiling report...")
    profile = ProfileReport(
        df_vis, 
        title="Heart Disease Profiling Report", 
        explorative=True
    )
    profiling_path = os.path.join(output_dir, "pandas_profiling_report.html")
    profile.to_file(profiling_path)
    print(f"Pandas Profiling report saved to {profiling_path}")

if __name__ == "__main__":
    generate_report()
