import numpy as np
from .base import Model

class CompositeModel(Model):
    """Сумма нескольких моделей"""
    
    def __init__(self, *models):
        super().__init__('Composite')
        self.models = list(models)
        
        # Собираем все параметры
        for i, model in enumerate(self.models):
            # Переименовываем параметры: model_prefix + param_name
            for param_name, value in model.params.items():
                full_name = f"{model.prefix or f'm{i}_'}{param_name}"
                self.set_param(full_name, value)
                
                # Копируем ограничения
                if param_name in model.bounds:
                    self.bounds[full_name] = model.bounds[param_name]
        
        # Обновляем имена параметров в дочерних моделях
        self._update_submodels()
    
    def _update_submodels(self):
        """Связываем параметры CompositeModel с дочерними моделями"""
        for i, model in enumerate(self.models):
            for param_name in model.params.keys():
                full_name = f"{model.prefix or f'm{i}_'}{param_name}"
                # Создаём свойство для синхронизации
                setattr(model, f'_{param_name}_link', full_name)
    
    def _function(self, x, **params):
        """Суммируем результаты всех моделей"""
        result = np.zeros_like(x)
        
        for model in self.models:
            # Выбираем только параметры этой модели
            model_params = {}
            for param_name in model.params.keys():
                link_name = getattr(model, f'_{param_name}_link', None)
                if link_name and link_name in params:
                    model_params[param_name] = params[link_name]
                elif param_name in params:  # Без префикса
                    model_params[param_name] = params[param_name]
            
            result += model._function(x, **model_params)
        
        return result
    
    def __add__(self, other):
        """Добавить ещё одну модель к композиту"""
        return CompositeModel(*self.models, other)