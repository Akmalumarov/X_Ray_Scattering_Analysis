# tools/basic_tools/experiment_catalog.py
import os
import h5py
import pandas as pd
from pathlib import Path

def scan_session(session_path, catalog_path="experiment_catalog.csv"):
    """
    Scan session and update experiment catalog
    """
    session_path = Path(session_path)
    
    session_type = _detect_session_type(session_path)
    
    if session_type == "ID02":
        experiments = _scan_id02_session(session_path)
    elif session_type == "BM26":
        experiments = _scan_bm26_session(session_path)
    else:
        print(f"Unknown session type for {session_path}")
        return
    
    _update_catalog(experiments, catalog_path, session_path.name)

def _detect_session_type(session_path):
    h5_count = 0
    total_files = 0
    
    for root, dirs, files in os.walk(session_path):
        for file in files:
            total_files += 1
            if file.endswith('.h5'):
                h5_count += 1
    
    if total_files > 0 and h5_count / total_files > 0.5:
        return "ID02"
    else:
        return "BM26"

def _scan_id02_session(session_path):
    experiments = []
    
    for file_path in session_path.rglob("*_ave.h5"):
        if "eiger2" in file_path.name and "av_ave" not in file_path.name:
            try:
                with h5py.File(file_path, 'r') as file:
                    title = file['entry_0000']['PyFAI']['eiger2']['header']['Title'][()]
                    
                    if isinstance(title, bytes):
                        title = title.decode('utf-8')
                    
                    experiments.append({
                        'title': title,
                        'keys': title.split('_'),
                        'file_path': str(file_path)
                    })
                        
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return experiments

def _scan_bm26_session(session_path):
    experiments = []
    
    for pattern in ['SAXS', 'WAXS']:
        for saxswaxs_dir in session_path.rglob(pattern):
            exp_dir = saxswaxs_dir.parent
            if exp_dir != session_path:
                edf_files = list(saxswaxs_dir.glob("*.edf"))
                file_path = str(edf_files[0]) if edf_files else ""
                
                experiments.append({
                    'title': exp_dir.name,
                    'keys': exp_dir.name.split('_'),
                    'file_path': file_path
                })
    
    return experiments

def _update_catalog(experiments, catalog_path, session_name):
    new_data = []
    for exp in experiments:
        new_data.append({
            'session': session_name,
            'title': exp['title'],
            'keys': '|'.join(exp['keys']),
            'file_path': exp['file_path']
        })
    
    new_df = pd.DataFrame(new_data)
    
    if os.path.exists(catalog_path):
        existing_df = pd.read_csv(catalog_path)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=['session', 'title'], keep='last'
        )
    else:
        combined_df = new_df
    
    combined_df.to_csv(catalog_path, index=False)
    print(f"Catalog updated: {len(new_data)} new experiments added")

__all__ = ['scan_session']