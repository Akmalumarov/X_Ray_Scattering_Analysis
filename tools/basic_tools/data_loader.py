# tools/basic_tools/data_loader.py
import os
import fabio
import h5py
import numpy as np
from typing import List, Tuple, Optional, Dict
import pandas as pd
from pathlib import Path

class BM26_experiment:
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.saxs_paths = None
        self.waxs_paths = None
        
    def _find_int_folder(self, main_dir: str) -> str:
        if not os.path.exists(main_dir):
            return ""
            
        for item in os.listdir(main_dir):
            item_path = os.path.join(main_dir, item)
            if os.path.isdir(item_path):
                dat_files = [f for f in os.listdir(item_path) if f.endswith('.dat')]
                if dat_files:
                    return item_path
        return ""
    
    def get_1d_SAXS_paths(self) -> List[str]:
        saxs_dir = os.path.join(self.base_path, 'SAXS')
        int_dir = self._find_int_folder(saxs_dir)
        
        dat_files = []
        if int_dir and os.path.exists(int_dir):
            dat_files = sorted([
                os.path.join(int_dir, f) 
                for f in os.listdir(int_dir) 
                if f.endswith('.dat') and not f.startswith('.')
            ])
        
        self.saxs_paths = {
            'main': saxs_dir,
            'int': int_dir,
            'edf': saxs_dir,
            'dat_files': dat_files
        }
        
        return dat_files
    
    def get_1d_WAXS_paths(self) -> List[str]:
        waxs_dir = os.path.join(self.base_path, 'WAXS')
        int_dir = self._find_int_folder(waxs_dir)
        
        dat_files = []
        if int_dir and os.path.exists(int_dir):
            dat_files = sorted([
                os.path.join(int_dir, f) 
                for f in os.listdir(int_dir) 
                if f.endswith('.dat') and not f.startswith('.')
            ])
        
        self.waxs_paths = {
            'main': waxs_dir,
            'int': int_dir,
            'edf': waxs_dir,
            'dat_files': dat_files
        }
        
        return dat_files

    def get_2d_SAXS_paths(self) -> List[str]:
        if self.saxs_paths is None:
            self.get_1d_SAXS_paths()
        
        edf_dir = self.saxs_paths['edf']
        edf_files = sorted([
            os.path.join(edf_dir, f) 
            for f in os.listdir(edf_dir) 
            if f.endswith('.edf') and not f.startswith('.')
        ])
        
        return edf_files

    def get_2d_WAXS_paths(self) -> List[str]:
        if self.waxs_paths is None:
            self.get_1d_WAXS_paths()
        
        edf_dir = self.waxs_paths['edf']
        edf_files = sorted([
            os.path.join(edf_dir, f) 
            for f in os.listdir(edf_dir) 
            if f.endswith('.edf') and not f.startswith('.')
        ])
        
        return edf_files
    
    def read_1d_dat(self, path_to_file: str) -> List[np.ndarray]:
        try:
            data = np.loadtxt(path_to_file)
            if data.ndim == 2 and data.shape[1] >= 2:
                q = data[:, 0]
                intensity = data[:, 1]
                return [q, intensity]
            else:
                print(f"Invalid data format in file {path_to_file}")
                return [np.array([]), np.array([])]
        except Exception as e:
            print(f"Error reading file {path_to_file}: {e}")
            return [np.array([]), np.array([])]
    
    def get_1d_SAXS(self, index: int = 0) -> List[np.ndarray]:
        if self.saxs_paths is None:
            self.get_1d_SAXS_paths()
        
        dat_files = self.saxs_paths['dat_files']
        
        if not dat_files:
            print("No available SAXS .dat files")
            return [np.array([]), np.array([])]
        
        if index < 0 or index >= len(dat_files):
            print(f"Index {index} out of range. Available files: {len(dat_files)}")
            return [np.array([]), np.array([])]
        
        file_path = dat_files[index]
        return self.read_1d_dat(file_path)
    
    def get_1d_WAXS(self, index: int = 0) -> List[np.ndarray]:
        if self.waxs_paths is None:
            self.get_1d_WAXS_paths()
        
        dat_files = self.waxs_paths['dat_files']
        
        if not dat_files:
            print("No available WAXS .dat files")
            return [np.array([]), np.array([])]
        
        if index < 0 or index >= len(dat_files):
            print(f"Index {index} out of range. Available files: {len(dat_files)}")
            return [np.array([]), np.array([])]
        
        file_path = dat_files[index]
        return self.read_1d_dat(file_path)
    
    def get_T_profile(self, detector: str = 'SAXS') -> List[float]:
        T = []
        
        if detector.upper() == 'SAXS':
            if self.saxs_paths is None:
                self.get_1d_SAXS_paths()
            edf_path = self.saxs_paths['edf']
        elif detector.upper() == 'WAXS':
            if self.waxs_paths is None:
                self.get_1d_WAXS_paths()
            edf_path = self.waxs_paths['edf']
        else:
            print(f"Unknown detector: {detector}")
            return T
        
        if not os.path.exists(edf_path):
            print(f"Directory {edf_path} does not exist")
            return T
        
        try:
            files_edf = sorted([f for f in os.listdir(edf_path) 
                              if not f.startswith('.') and f.endswith('.edf')])
            
            if not files_edf:
                print(f"No EDF files found in {edf_path}")
                return T
            
            print(f"Found {len(files_edf)} EDF files in {detector}")
            
            for file in files_edf:
                file_path = os.path.join(edf_path, file)
                try:
                    edf = fabio.open(file_path)
                    temp_str = edf.header.get('Tlinkam', '')
                    if temp_str:
                        T.append(float(temp_str))
                    else:
                        for key in ['Temperature', 'Temp', 'T', 'temperature']:
                            temp_str = edf.header.get(key, '')
                            if temp_str:
                                try:
                                    T.append(float(temp_str))
                                    break
                                except ValueError:
                                    continue
                        else:
                            print(f"Temperature not found in file {file}")
                            T.append(np.nan)
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
                    T.append(np.nan)
                    
        except Exception as e:
            print(f"Error accessing directory {edf_path}: {e}")
        
        return T

    def get_Tr_profile(self):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)

        Tr_array = []

        for frame in range(number_of_frames):
            edf = fabio.open(paths[frame])
            Photo = edf.header.get('Photo')
            Monitor = edf.header.get('Monitor')
            Tr_array.append(float(Photo)/float(Monitor))

        return Tr_array

    def integrate1d_SAXS(self, ai, npt, mask=None, unit='q_nm^-1'):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)
        results = []

        for frame in range(number_of_frames):
            img = fabio.open(paths[frame]).data
            res_1D = ai.integrate1d(img, npt, mask=mask, unit=unit)
            results.append(res_1D[1])
            
        return res_1D[0], results

    def integrate1d_WAXS(self, ai, npt, mask=None, unit='q_nm^-1'):
        paths = self.get_2d_WAXS_paths()
        number_of_frames = len(paths)
        results = []

        for frame in range(number_of_frames):
            img = fabio.open(paths[frame]).data
            res_1D = ai.integrate1d(img, npt, mask=mask, unit=unit)
            results.append(res_1D[1])
            
        return res_1D[0], results
        
    def integrate2d_SAXS(self, ai, npt, mask=None, unit='q_nm^-1', azim=True):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)
        results = []

        if azim == False:
            for frame in range(number_of_frames):
                img = fabio.open(paths[frame]).data
                res_2D = ai.integrate2d(img, npt, mask=mask, unit=("qx_nm^-1", "qy_nm^-1"))
                results.append(res_2D)
            return results
        else:  # azim == True
            for frame in range(number_of_frames):
                img = fabio.open(paths[frame]).data
                res_2D = ai.integrate2d(img, npt, mask=mask, unit=unit)
                results.append(res_2D)
            return results

    def integrate2d_WAXS(self, ai, npt, mask=None, unit='q_nm^-1', azim=True):
        paths = self.get_2d_WAXS_paths()
        number_of_frames = len(paths)
        results = []

        if azim == False:
            for frame in range(number_of_frames):
                img = fabio.open(paths[frame]).data
                res_2D = ai.integrate2d(img, npt, mask=mask, unit=("qx_nm^-1", "qy_nm^-1"))
                results.append(res_2D)
            return results
        else:  # azim == True
            for frame in range(number_of_frames):
                img = fabio.open(paths[frame]).data
                res_2D = ai.integrate2d(img, npt, mask=mask, unit=unit)
                results.append(res_2D)
            return results

    def subtract(self, I, Tr=None):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)
        Tr_array = self.get_Tr_profile()
        results = []

        for frame in range(number_of_frames):
            q, I_curr = self.get_1d_SAXS(frame)
            results.append(I_curr / Tr_array[frame] - I / Tr)
        return q, results

    def average_1d(self, frames):
        q, I = self.get_1d_SAXS(0)
        I *= 0
        for frame in frames:
            q, I_fr = self.get_1d_SAXS(frame)
            I += I_fr
        I /= len(frames)
        return q, I


class ID02_experiment:
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.saxs_files = []
        self.waxs_files = []
        self._scan_files()
    
    def _scan_files(self):
        for file_path in Path(self.base_path).rglob("*_ave.h5"):
            if "av_ave" not in file_path.name:
                if "eiger2" in file_path.name:
                    self.saxs_files.append(str(file_path))
                else:
                    self.waxs_files.append(str(file_path))
        
        self.saxs_files.sort()
        self.waxs_files.sort()
    
    def get_1d_data(self) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.base_path, 'r') as file:
            data = file['entry_0000']['PyFAI']['result_ave']['data'][:]
            q = file['entry_0000']['PyFAI']['result_ave']['q'][0:]
            return q, data
    
    # def get_1d_SAXS(self, frame: int = 0) -> List[np.ndarray]:
    #     if not self.saxs_files:
    #         print("No available SAXS files")
    #         return [np.array([]), np.array([])]
        
    #     if frame < 0 or frame >= len(self.saxs_files):
    #         print(f"Frame {frame} out of range. Available files: {len(self.saxs_files)}")
    #         return [np.array([]), np.array([])]
        
    #     file_path = self.saxs_files[frame]
    #     q, data = self.read_h5(file_path)
    #     return [q, data[frame]] if data.ndim > 1 else [q, data]
    
    # def get_1d_WAXS(self, frame: int = 0) -> List[np.ndarray]:
    #     if not self.waxs_files:
    #         print("No available WAXS files")
    #         return [np.array([]), np.array([])]
        
    #     if frame < 0 or frame >= len(self.waxs_files):
    #         print(f"Frame {frame} out of range. Available files: {len(self.waxs_files)}")
    #         return [np.array([]), np.array([])]
        
    #     file_path = self.waxs_files[frame]
    #     q, data = self.read_h5(file_path)
    #     return [q, data[frame]] if data.ndim > 1 else [q, data]


def list_all_files(folder_path):
    from pathlib import Path
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist")
        return []
    
    file_paths = [str(file) for file in folder.rglob('*') if file.is_file()]
    return sorted(file_paths)


def list_experiments(session_path):
    from pathlib import Path
    
    session = Path(session_path)
    if not session.exists():
        print(f"Session folder {session_path} does not exist")
        return []
    
    experiments = [str(item) for item in session.iterdir() if item.is_dir()]
    return sorted(experiments)  
    
    from pathlib import Path
    
    session = Path(session_path)
    if not session.exists():
        print(f"Session folder {session_path} does not exist")
        return []
    
    # Get only immediate subdirectories (top level)
    experiments = [str(item) for item in session.iterdir() if item.is_dir()]
    return sorted(experiments)