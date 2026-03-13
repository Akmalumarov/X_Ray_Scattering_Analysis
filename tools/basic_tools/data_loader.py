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
        
        return np.array(T)

    def get_Tr_profile(self):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)

        Tr_array = []

        for frame in range(number_of_frames):
            edf = fabio.open(paths[frame])
            Photo = edf.header.get('Photo')
            Monitor = edf.header.get('Monitor')
            Tr_array.append(float(Photo)/float(Monitor))

        return np.array(Tr_array)

    def get_Photo(self):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)

        Tr_array = []

        for frame in range(number_of_frames):
            edf = fabio.open(paths[frame])
            Photo = edf.header.get('Photo')
            Monitor = edf.header.get('Monitor')
            Tr_array.append(float(Photo))

        return np.array(Tr_array)

        

    def get_DSC_profile(self):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)

        DSC_array = []

        for frame in range(number_of_frames):
            edf = fabio.open(paths[frame])
            DSC = float(edf.header.get('DSClink'))
            DSC_array.append(DSC)

        return np.array(DSC_array)


    def get_Time_profile(self):
        paths = self.get_2d_SAXS_paths()
        number_of_frames = len(paths)

        edf0 = fabio.open(paths[0])
        hrs0 = int((edf0.header['time'].split()[3]).split(':')[0])*3600
        mts0 = int((edf0.header['time'].split()[3]).split(':')[1])*60
        sec0 = int((edf0.header['time'].split()[3]).split(':')[2])
        time0 = hrs0+mts0+sec0


        Time_array = []

        for frame in range(number_of_frames):
            edf = fabio.open(paths[frame])
            hrs = int((edf.header['time'].split()[3]).split(':')[0])*3600
            mts = int((edf.header['time'].split()[3]).split(':')[1])*60
            sec = int((edf.header['time'].split()[3]).split(':')[2])
            time = hrs + mts + sec - time0

            Time_array.append(time)

        return np.array(Time_array)

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
    
    def __init__(self, file_path: str):
        self.base_path = file_path
        path_obj = Path(file_path)
        self.directory = path_obj.parent
        self.filename_stem = path_obj.stem
        
        for suffix in ['_ave', '_norm', '_azim']:
            if self.filename_stem.endswith(suffix):
                self.file_basename = self.filename_stem[:-len(suffix)]
                break
        else:
            self.file_basename = self.filename_stem
        
        self.ave_path = os.path.join(self.directory, f"{self.file_basename}_ave.h5")
        self.norm_path = os.path.join(self.directory, f"{self.file_basename}_norm.h5")
        self.azim_path = os.path.join(self.directory, f"{self.file_basename}_azim.h5")
        
        self.has_ave = os.path.exists(self.ave_path)
        self.has_norm = os.path.exists(self.norm_path)
        self.has_azim = os.path.exists(self.azim_path)
    
    def show_tree(self, file_type='ave', max_depth=None):
        """Печатает дерево HDF5 файла"""
        path = getattr(self, f"{file_type}_path", None)
        if not path or not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        def _print(obj, indent=0, depth=0):
            if max_depth and depth > max_depth:
                print('  ' * indent + '├─ ...')
                return
            for key in obj.keys():
                print('  ' * indent + f'├─ {key}')
                if isinstance(obj[key], h5py.Group):
                    _print(obj[key], indent + 1, depth + 1)
        
        with h5py.File(path, 'r') as f:
            _print(f)
    
    def _find_title(self, obj):
        """Рекурсивно ищет первый ключ с title и возвращает значение"""
        for key in obj.keys():
            if 'title' in key.lower():
                try:
                    val = obj[key][()]
                    return val.decode('utf-8') if isinstance(val, bytes) else str(val)
                except:
                    pass
            if isinstance(obj[key], h5py.Group):
                val = self._find_title(obj[key])
                if val:
                    return val
        return None
    
    def get_Title(self):
        """Ищет title в _ave, _norm, _azim файлах"""
        for path in [self.ave_path, self.norm_path, self.azim_path]:
            if os.path.exists(path):
                try:
                    with h5py.File(path, 'r') as f:
                        title = self._find_title(f)
                        if title:
                            return title
                except:
                    continue
        return "Title not found"
    
    def get_1d_data(self, frame: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает q и intensity из _ave.h5"""
        if not self.has_ave:
            print(f"No _ave.h5 file found")
            return np.array([]), np.array([])
        
        try:
            with h5py.File(self.ave_path, 'r') as f:
                q = f['entry_0000']['PyFAI']['result_ave']['q'][:]
                data = f['entry_0000']['PyFAI']['result_ave']['data'][:]
                
                if data.ndim == 1:
                    return q, data
                elif data.ndim == 2:
                    if frame < 0 or frame >= data.shape[0]:
                        print(f"Frame {frame} out of range (0-{data.shape[0]-1})")
                        return np.array([]), np.array([])
                    return q, data[frame]
                else:
                    return np.array([]), np.array([])
        except Exception as e:
            print(f"Error reading 1D data: {e}")
            return np.array([]), np.array([])
    
    def get_2d_data(self, frame: int = 0) -> np.ndarray:
        """Возвращает 2D изображение из _norm.h5"""
        if not self.has_norm:
            print(f"No _norm.h5 file found")
            return np.array([])
        
        try:
            with h5py.File(self.norm_path, 'r') as f:
                data = f['entry_0000']['PyFAI']['result_ave']['data'][:]
                
                if data.ndim == 2:
                    return data
                elif data.ndim == 3:
                    if frame < 0 or frame >= data.shape[0]:
                        print(f"Frame {frame} out of range (0-{data.shape[0]-1})")
                        return np.array([])
                    return data[frame]
                else:
                    return np.array([])
        except Exception as e:
            print(f"Error reading 2D data: {e}")
            return np.array([])
    
    def get_azim_data(self, frame: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает chi, q и 2D данные из _azim.h5"""
        if not self.has_azim:
            print(f"No _azim.h5 file found")
            return np.array([]), np.array([]), np.array([])
        
        try:
            with h5py.File(self.azim_path, 'r') as f:
                chi = f['entry_0000']['PyFAI']['result_azim']['chi'][:]
                q = f['entry_0000']['PyFAI']['result_azim']['q'][:]
                data = f['entry_0000']['PyFAI']['result_azim']['data'][:]
                
                if data.ndim == 2:
                    return chi, q, data
                elif data.ndim == 3:
                    if frame < 0 or frame >= data.shape[0]:
                        print(f"Frame {frame} out of range (0-{data.shape[0]-1})")
                        return np.array([]), np.array([]), np.array([])
                    return chi, q, data[frame]
                else:
                    return np.array([]), np.array([]), np.array([])
        except Exception as e:
            print(f"Error reading azimuthal data: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def get_all_1d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает все 1D данные (матрица кадры x q)"""
        if not self.has_ave:
            return np.array([]), np.array([])
        
        try:
            with h5py.File(self.ave_path, 'r') as f:
                q = f['entry_0000']['PyFAI']['result_ave']['q'][:]
                data = f['entry_0000']['PyFAI']['result_ave']['data'][:]
                
                if data.ndim == 1:
                    return q, data.reshape(1, -1)
                return q, data
        except:
            return np.array([]), np.array([])
    
    def get_n_frames(self) -> dict:
        """Возвращает количество кадров в каждом типе файлов"""
        frames = {'ave': 0, 'norm': 0, 'azim': 0}
        
        if self.has_ave:
            try:
                with h5py.File(self.ave_path, 'r') as f:
                    data = f['entry_0000']['PyFAI']['result_ave']['data'][:]
                    frames['ave'] = data.shape[0] if data.ndim > 1 else 1
            except:
                pass
        
        if self.has_norm:
            try:
                with h5py.File(self.norm_path, 'r') as f:
                    data = f['entry_0000']['PyFAI']['result_ave']['data'][:]
                    frames['norm'] = data.shape[0] if data.ndim > 2 else 1
            except:
                pass
        
        if self.has_azim:
            try:
                with h5py.File(self.azim_path, 'r') as f:
                    data = f['entry_0000']['PyFAI']['result_azim']['data'][:]
                    frames['azim'] = data.shape[0] if data.ndim > 2 else 1
            except:
                pass
        
        return frames

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