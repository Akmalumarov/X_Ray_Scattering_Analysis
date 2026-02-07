import numpy as np
import inspect

# Декоратор для автоматического определения количества параметров
def model_function(func):
    """Decorator to mark functions as models and store param count"""
    sig = inspect.signature(func)
    # Считаем параметры, исключая первый (x) и параметры со значениями по умолчанию
    params = list(sig.parameters.keys())[1:]  # все кроме первого
    func._n_params = len(params)
    return func

# Все функции-модели
@model_function
def gaussian(x, amp, cen, sigma):
    """Gaussian: A * exp(-(x-cent)**2 / (2*sigma**2))"""
    return amp * np.exp(-(x-cen)**2 / (2*sigma**2))

@model_function
def lorentzian(x, amp, cen, gamma):
    """Lorentzian: A / (1 + ((x-cent)/gamma)**2)"""
    return amp / (1 + ((x-cen)/gamma)**2)

@model_function
def power_law(x, a, b, c):
    """Power law: a * x**b + c"""
    return a * x**b + c

@model_function
def linear(x, k, b):
    """Linear: k*x + b"""
    return k*x + b

@model_function
def exponential(x, A, tau, offset):
    """Exponential: A*exp(-x/tau) + offset"""
    return A * np.exp(-x/tau) + offset

@model_function
def voigt(x, amp, cen, sigma, gamma):
    """Voigt profile (approximation)"""
    # Simple approximation - can be improved
    g = np.exp(-(x-cen)**2 / (2*sigma**2))
    l = 1 / (1 + ((x-cen)/gamma)**2)
    return amp * (g + l) / 2

@model_function
def constant(x, c):
    """Constant background"""
    return np.full_like(x, c)

# Вспомогательная функция для списка доступных моделей
def list_available_models():
    """Return sorted list of available model names"""
    import sys
    current_module = sys.modules[__name__]
    models = []
    for name in dir(current_module):
        obj = getattr(current_module, name)
        if callable(obj) and hasattr(obj, '_n_params') and not name.startswith('_'):
            models.append(name)
    return sorted(models)