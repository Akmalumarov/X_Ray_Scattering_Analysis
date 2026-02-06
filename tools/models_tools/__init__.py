
from .peaks import Gaussian

from .background import PowerLaw

from .composite import CompositeModel

# Утилиты
from .utils import list_models, create_model

# Все доступные классы
__all__ = [
    'Gaussian',
    'PowerLaw',
    'CompositeModel',
    'list_models',
    'create_model',
]