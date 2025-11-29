import os
import fabio
import numpy as np
from typing import List, Tuple, Optional, Dict
import pandas as pd

class BM26_experiment:
    """Класс для работы с экспериментальными данными BM26"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.saxs_paths = None
        self.waxs_paths = None
        
    def _find_int_folder(self, main_dir: str) -> str:

        if not os.path.exists(main_dir):
            return ""
            
        # Ищем папки, которые могут содержать интегрированные данные
        for item in os.listdir(main_dir):
            item_path = os.path.join(main_dir, item)
            if os.path.isdir(item_path):
                # Проверяем, содержит ли папка .dat файлы
                dat_files = [f for f in os.listdir(item_path) if f.endswith('.dat')]
                if dat_files:
                    return item_path
        
        # Если не нашли папку с .dat файлами, возвращаем пустую строку
        return ""
    
    def get_SAXS_paths(self) -> List[str]:

        saxs_dir = os.path.join(self.base_path, 'SAXS')
        int_dir = self._find_int_folder(saxs_dir)
        
        dat_files = []
        if int_dir and os.path.exists(int_dir):
            # Получаем все .dat файлы и сортируем их
            dat_files = sorted([
                os.path.join(int_dir, f) 
                for f in os.listdir(int_dir) 
                if f.endswith('.dat') and not f.startswith('.')
            ])
        
        # Сохраняем информацию о путях для других методов
        self.saxs_paths = {
            'main': saxs_dir,
            'int': int_dir,
            'edf': saxs_dir,
            'dat_files': dat_files
        }
        
        return dat_files
    
    def get_WAXS_paths(self) -> List[str]:

        waxs_dir = os.path.join(self.base_path, 'WAXS')
        int_dir = self._find_int_folder(waxs_dir)
        
        dat_files = []
        if int_dir and os.path.exists(int_dir):
            # Получаем все .dat файлы и сортируем их
            dat_files = sorted([
                os.path.join(int_dir, f) 
                for f in os.listdir(int_dir) 
                if f.endswith('.dat') and not f.startswith('.')
            ])
        
        # Сохраняем информацию о путях для других методов
        self.waxs_paths = {
            'main': waxs_dir,
            'int': int_dir,
            'edf': waxs_dir,
            'dat_files': dat_files
        }
        
        return dat_files
    
    def read_1d_dat(self, path_to_file: str) -> List[np.ndarray]:
        try:  # ← этот try должен быть с отступом
            data = np.loadtxt(path_to_file)
            if data.ndim == 2 and data.shape[1] >= 2:
                q = data[:, 0]
                intensity = data[:, 1]
                return [q, intensity]
            else:
                print(f"Некорректный формат данных в файле {path_to_file}")
                return [np.array([]), np.array([])]
        except Exception as e:  # ← и этот except
            print(f"Ошибка при чтении файла {path_to_file}: {e}")
            return [np.array([]), np.array([])]
    
    def get_1d_SAXS(self, index: int = 0) -> List[np.ndarray]:

        if self.saxs_paths is None:
            self.get_SAXS_paths()
        
        dat_files = self.saxs_paths['dat_files']
        
        if not dat_files:
            print("Нет доступных SAXS .dat файлов")
            return [np.array([]), np.array([])]
        
        if index < 0 or index >= len(dat_files):
            print(f"Индекс {index} вне диапазона. Доступно файлов: {len(dat_files)}")
            return [np.array([]), np.array([])]
        
        file_path = dat_files[index]
        #print(f"Чтение SAXS файла {index}: {os.path.basename(file_path)}")
        return self.read_1d_dat(file_path)
    
    def get_1d_WAXS(self, index: int = 0) -> List[np.ndarray]:
        if self.waxs_paths is None:
            self.get_WAXS_paths()
        
        dat_files = self.waxs_paths['dat_files']
        
        if not dat_files:
            print("Нет доступных WAXS .dat файлов")
            return [np.array([]), np.array([])]
        
        if index < 0 or index >= len(dat_files):
            print(f"Индекс {index} вне диапазона. Доступно файлов: {len(dat_files)}")
            return [np.array([]), np.array([])]
        
        file_path = dat_files[index]
       # print(f"Чтение WAXS файла {index}: {os.path.basename(file_path)}")
        return self.read_1d_dat(file_path)
    
    def get_T_profile(self, detector: str = 'SAXS') -> List[float]:
        T = []
        
        # Получаем пути к данным
        if detector.upper() == 'SAXS':
            if self.saxs_paths is None:
                self.get_SAXS_paths()
            edf_path = self.saxs_paths['edf']
        elif detector.upper() == 'WAXS':
            if self.waxs_paths is None:
                self.get_WAXS_paths()
            edf_path = self.waxs_paths['edf']
        else:
            print(f"Неизвестный детектор: {detector}")
            return T
        
        # Проверяем существование папки
        if not os.path.exists(edf_path):
            print(f"Папка {edf_path} не существует")
            return T
        
        try:
            # Получаем список EDF файлов, исключая скрытые файлы
            files_edf = sorted([f for f in os.listdir(edf_path) 
                              if not f.startswith('.') and f.endswith('.edf')])
            
            if not files_edf:
                print(f"В папке {edf_path} не найдено EDF файлов")
                return T
            
            print(f"Найдено {len(files_edf)} EDF файлов в {detector}")
            
            for file in files_edf:
                file_path = os.path.join(edf_path, file)
                try:
                    edf = fabio.open(file_path)
                    # Пытаемся извлечь температуру из заголовка
                    temp_str = edf.header.get('Tlinkam', '')
                    if temp_str:
                        T.append(float(temp_str))
                    else:
                        # Попробуем другие возможные ключи для температуры
                        for key in ['Temperature', 'Temp', 'T', 'temperature']:
                            temp_str = edf.header.get(key, '')
                            if temp_str:
                                try:
                                    T.append(float(temp_str))
                                    break
                                except ValueError:
                                    continue
                        else:
                            print(f"Температура не найдена в файле {file}")
                            T.append(np.nan)
                except Exception as e:
                    print(f"Ошибка при чтении файла {file}: {e}")
                    T.append(np.nan)
                    
        except Exception as e:
            print(f"Ошибка при доступе к папке {edf_path}: {e}")
        
        return T
