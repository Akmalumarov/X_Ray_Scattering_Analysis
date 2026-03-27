import numpy as np

def corr_func(q, I, res=1, zmax=None):
    dq = np.diff(q).min()
    dz = 2 * np.pi / q.max() / res
    if zmax == None:
        z = np.arange(dz, 2 * np.pi / dq, dz)
    else:
        z = np.arange(dz, zmax, dz)
    
    K = np.trapezoid(I[:, None] * np.cos(q[:, None] * z[None, :]), q, axis=0)
    return z, K