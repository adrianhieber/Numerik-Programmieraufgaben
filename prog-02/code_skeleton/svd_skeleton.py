import numpy as np
import matplotlib.pyplot as plt

#%%
def svd(A, variant = 'standard', t=None):
    if np.ndim(A) == 2:
        m, n = np.shape(A)
    else:
        raise ValueError('Eingabematirx hat falsche Form!')
        
    U, S, VH = None   
    
    if variant=='standard':
        pass
    else:
        if variant=='truncated':
            t = None
        elif variant=='thin':
            t = None
        elif variant=='compact':
            t = None
        else:
            raise ValueError('Unknown Variant specified!')
        
        U = None
        S = None
        V = None
        
    return U, S, V

#%% image compression
class compressed_img:
    def __init__(self, I, factor=1.0):
        self.factor = factor
        self.shape = I.shape

        self.t = None
        self.compress_svd(I, self.t)
        
    def __call__(self):
        pass
        
    def compress_svd(self, I, t):
        self.U = np.zeros((self.shape[0], t, self.shape[2]))
        self.S = np.zeros((t, self.shape[2]))
        self.V = np.zeros((self.shape[1], t, self.shape[2]))
        
        self.num_pixels = None

#%% Visualisiere Kompression


