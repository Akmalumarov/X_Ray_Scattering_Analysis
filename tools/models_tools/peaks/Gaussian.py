import numpy as np
from ..base import Model

class Gaussian(Model):
    """Гауссов пик: A * exp(-(x-center)²/(2*sigma²))"""
    
    def __init__(self, amplitude=1.0, center=0.0, sigma=1.0, prefix=''):
        super().__init__('Gaussian', prefix)
        
        # Определяем параметры
        self.set_param('amplitude', amplitude, min=0)
        self.set_param('center', center)
        self.set_param('sigma', sigma, min=1e-10)
    
    def _function(self, x, amplitude, center, sigma):
        return amplitude * np.exp(-((x - center) / sigma) ** 2 / 2)
    
    # Метод для удобства
    @property
    def fwhm(self):
        """Полуширина на полувысоте"""
        return 2.35482 * self.params['sigma']