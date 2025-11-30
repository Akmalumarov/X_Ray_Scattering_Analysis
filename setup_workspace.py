# setup_workspace.py
"""
Automated SAXS/WAXS workspace setup
"""

import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import scipy as sp
    from scipy.optimize import curve_fit
    import pyFAI
    import h5py
    import fabio
    
    # Our tools
    from tools import *
    from tools.basic_tools import *
    from tools.saxs_tools import * 
    from tools.waxs_tools import *
    
    # Add scan_session to global namespace
    from tools.basic_tools.experiment_catalog import scan_session
    
    # Plot settings
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    
    # Useful variables
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    results_dir = current_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    print("✓ Workspace ready")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Run: pip install -e .")
except Exception as e:
    print(f"✗ Error: {e}")