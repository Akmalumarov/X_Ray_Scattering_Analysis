import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    """Абстрактный базовый класс для всех моделей"""
    
    def __init__(self, name, prefix=''):
        self.name = name
        self.prefix = prefix  # Для lmfit: 'p1_', 'bkg_' и т.д.
        self.params = {}      # {имя: значение}
        self.param_names = [] # Порядок параметров
        self.bounds = {}      # Ограничения {имя: (min, max)}
        
    @abstractmethod
    def _function(self, x, **params):
        """Чистая математическая функция"""
        pass
    
    def __call__(self, x, **params):
        """Вызов модели: model(x) или model(x, amplitude=10, ...)"""
        # Если переданы параметры — используем их, иначе self.params
        actual_params = {**self.params, **params}
        return self._function(x, **actual_params)
    
    def set_param(self, name, value, min=None, max=None):
        """Установить параметр"""
        self.params[name] = value
        self.param_names.append(name)
        if min is not None or max is not None:
            self.bounds[name] = (min, max)
    
    def __add__(self, other):
        """Перегрузка + для создания CompositeModel"""
        from .composite import CompositeModel
        return CompositeModel(self, other)
    
    def __repr__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"