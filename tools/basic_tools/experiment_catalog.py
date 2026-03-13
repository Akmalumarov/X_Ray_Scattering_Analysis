# tools/basic_tools/experiment_catalog.py
import os
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def scan_session(session_path, catalog_path="experiment_catalog.csv"):
    """
    Scan session and update experiment catalog
    
    Args:
        session_path: path to session folder
        catalog_path: path to output CSV catalog
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
    """Quick session type detection based on file extensions"""
    h5_count = 0
    total = 0
    
    for root, dirs, files in os.walk(session_path):
        for file in files[:100]:  # check first 100 files
            total += 1
            if file.endswith('.h5'):
                h5_count += 1
        if total > 500:  # limit search
            break
    
    return "ID02" if total > 0 and h5_count / total > 0.3 else "BM26"

def _find_title_in_h5(file_path):
    """Find title in HDF5 file - first try standard paths, then full search"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Standard ID02 paths
            standard_paths = [
                'entry_0000/PyFAI/parameters/Title',
                'entry_0000/PyFAI/waxs/header/Title',
                'entry_0000/title',
                'entry_0000/PyFAI/eiger2/header/Title',
                'entry_0000/configuration/Title'
            ]
            
            # Try standard paths first
            for path in standard_paths:
                try:
                    obj = f
                    for part in path.split('/'):
                        obj = obj[part]
                    val = obj[()]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8', errors='ignore')
                    return str(val)
                except (KeyError, TypeError):
                    continue
            
            # If not found, recursive search
            def search(obj):
                for key in obj.keys():
                    if 'title' in key.lower():
                        try:
                            val = obj[key][()]
                            if isinstance(val, bytes):
                                val = val.decode('utf-8', errors='ignore')
                            return str(val)
                        except:
                            pass
                    if isinstance(obj[key], h5py.Group):
                        val = search(obj[key])
                        if val:
                            return val
                return None
            
            return search(f)
            
    except Exception:
        return None

def _scan_id02_session(session_path):
    """Scan ID02 session for _ave.h5, _norm.h5, _azim.h5 files"""
    experiments = []
    
    # Target suffixes
    suffixes = ['_ave.h5', '_norm.h5', '_azim.h5']
    
    # Collect all target files
    target_files = []
    for suffix in suffixes:
        target_files.extend(session_path.rglob(f"*{suffix}"))
    
    if not target_files:
        print("No _ave, _norm, or _azim files found")
        return experiments
    
    print(f"Found {len(target_files)} target files")
    
    # Process with progress bar
    for file_path in tqdm(target_files, desc="Scanning HDF5 files"):
        # Skip tiny files
        if file_path.stat().st_size < 1000:
            continue
        
        title = _find_title_in_h5(file_path)
        if title:
            # Extract file type from suffix
            file_type = file_path.suffixes[-2][1:] if len(file_path.suffixes) >= 2 else "unknown"
            
            experiments.append({
                'title': title,
                'file_path': str(file_path),
                'file_type': file_type  # ave, norm, azim
            })
    
    print(f"Found {len(experiments)} experiments with titles")
    return experiments

def _scan_bm26_session(session_path):
    """Scan BM26 session for SAXS/WAXS folders"""
    experiments = []
    
    patterns = ['SAXS', 'WAXS']
    
    for pattern in patterns:
        dirs = list(session_path.rglob(pattern))
        
        for saxswaxs_dir in tqdm(dirs, desc=f"Scanning {pattern}"):
            exp_dir = saxswaxs_dir.parent
            
            # Skip root directory
            if exp_dir == session_path:
                continue
            
            # Look for data files
            edf_files = list(saxswaxs_dir.glob("*.edf"))
            dat_files = list(saxswaxs_dir.glob("*.dat"))
            
            if edf_files or dat_files:
                file_path = str(edf_files[0]) if edf_files else str(dat_files[0])
                
                experiments.append({
                    'title': exp_dir.name,
                    'file_path': file_path,
                    'file_type': 'edf/dat'
                })
    
    return experiments

def _update_catalog(experiments, catalog_path, session_name):
    """Update CSV catalog with new experiments"""
    if not experiments:
        print(f"No experiments found in session {session_name}")
        return
    
    # Prepare new data
    new_data = []
    seen = set()
    
    for exp in experiments:
        key = (exp['title'], exp['file_path'])
        if key not in seen:
            seen.add(key)
            new_data.append({
                'session': session_name,
                'title': exp['title'],
                'file_path': exp['file_path'],
                'file_type': exp.get('file_type', '')
            })
    
    new_df = pd.DataFrame(new_data)
    
    # Merge with existing catalog
    if os.path.exists(catalog_path):
        existing_df = pd.read_csv(catalog_path)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=['session', 'title', 'file_path'], 
            keep='last'
        )
    else:
        combined_df = new_df
    
    # Sort and save
    combined_df = combined_df.sort_values(['session', 'title'])
    combined_df.to_csv(catalog_path, index=False)
    
    print(f"Catalog updated: {len(new_data)} experiments added to {session_name}")
    print(f"Total entries in catalog: {len(combined_df)}")

# Export only scan_session function
__all__ = ['scan_session']