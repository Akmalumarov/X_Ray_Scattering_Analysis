# setup_workspace.py
"""
Автоматическая настройка рабочего пространства SAXS/WAXS
"""

import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pyFAI
    import h5py
    from pathlib import Path
    
    # Наши инструменты
    from tools import *
    from tools.basic_tools import *
    from tools.saxs_tools import * 
    from tools.waxs_tools import *

    pi = np.pi
    
    print("✓ Рабочее пространство готово")
    
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Выполните: pip install -e .")
except Exception as e:
    print(f"✗ Ошибка: {e}")
