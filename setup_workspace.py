# setup_workspace.py
"""
Автоматическая настройка рабочего пространства SAXS/WAXS
"""

import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    
    # Наши инструменты
    from tools import *
    from tools.basic_tools import *
    from tools.saxs_tools import * 
    from tools.waxs_tools import *
    from tools.basic_tools import *  # ← это импортирует scan_session
    
    # Добавляем функцию scan_session в глобальное пространство
    from tools.basic_tools.experiment_catalog import scan_session
    
    # Настройка графиков
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12
    
    # Полезные переменные
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    results_dir = current_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    print("✓ Рабочее пространство готово")
    
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    print("Выполните: pip install -e .")
except Exception as e:
    print(f"✗ Ошибка: {e}")