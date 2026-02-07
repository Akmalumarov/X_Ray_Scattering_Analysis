import numpy as np
import inspect

class CompositeModel:
    """Composite model - sum of multiple functions"""
    
    def __init__(self, *components):
        """
        components: callable functions with known parameter counts
        
        Example:
            model = CompositeModel(gaussian, power_law)
            model = CompositeModel(gaussian) + lorentzian + linear
        """
        self.components = []
        self.n_params = 0
        self.param_slices = []
        self.component_info = []
        
        for i, func in enumerate(components):
            if not callable(func):
                raise TypeError(f"Component {i} must be callable, got {type(func)}")
            
            # Определяем количество параметров
            if hasattr(func, '_n_params'):
                n = func._n_params
            else:
                # Пытаемся определить автоматически
                try:
                    sig = inspect.signature(func)
                    n = len(list(sig.parameters.keys())) - 1
                    if n < 0:
                        n = 0
                except:
                    n = 0
            
            self.components.append(func)
            comp_name = func.__name__ if hasattr(func, '__name__') else f"func_{i}"
            self.component_info.append((comp_name, n))
            self.param_slices.append(slice(self.n_params, self.n_params + n))
            self.n_params += n
    
    def __call__(self, x, *params):
        """Evaluate model at points x with given parameters"""
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        
        result = np.zeros_like(x)
        for func, param_slice in zip(self.components, self.param_slices):
            if param_slice.start == param_slice.stop:  # No parameters
                result += func(x)
            else:
                result += func(x, *params[param_slice])
        
        return result
    
    def __add__(self, other):
        """Add another component or model to this one"""
        if isinstance(other, CompositeModel):
            return CompositeModel(*self.components, *other.components)
        elif callable(other):
            return CompositeModel(*self.components, other)
        else:
            raise TypeError(f"Can only add callables or CompositeModels, got {type(other)}")
    
    def get_param_info(self):
        """Return information about parameters for each component"""
        info = []
        param_idx = 0
        for name, n in self.component_info:
            if n > 0:
                info.append(f"{name}: parameters {param_idx} to {param_idx+n-1}")
                param_idx += n
            else:
                info.append(f"{name}: no parameters")
        return info
    
    def get_initial_guess(self):
        """Return simple initial guess (all ones)"""
        return np.ones(self.n_params)
    
    def __repr__(self):
        """String representation of the model"""
        comp_names = [name for name, _ in self.component_info]
        return f"CompositeModel({', '.join(comp_names)}) with {self.n_params} params"