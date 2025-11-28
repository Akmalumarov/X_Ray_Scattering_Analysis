# tools/waxs_tools/Lorenz_correction.py
import numpy as np

def Lorenz(q, I):
    return I * q ** 2