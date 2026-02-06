import numpy as np
from ..base import Model

class PowerLaw(Model):
    """Степенная функция: A * x^exponent"""
    
    def __init__(self, amplitude=1.0, exponent=-2.0, prefix=''):
        super().__init__('PowerLaw', prefix)
        
        self.set_param('amplitude', amplitude, min=0)
        self.set_param('exponent', exponent)
    
    def _function(self, x, amplitude, exponent):
        # Защита от x=0 для отрицательных exponent
        x_safe = np.where(x == 0, 1e-10, x)
        return amplitude * x_safe ** exponent