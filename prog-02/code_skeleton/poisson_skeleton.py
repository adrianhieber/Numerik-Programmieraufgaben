import numpy as np
import matplotlib.pyplot as plt
import solvers
import time

#%% Laplace Matrix
def LaplaceOperator(N):
    return (2*np.eye(N-1)-np.eye(N-1,k=-1)-np.eye(N-1,k=1))

#%% Teilaufgabe (i): Simple_DGL
class Simple_DGL:
    def __init__(self, a=0., b=1.):
        pass
        
    def __call__(self, x):
        pass
    
    def boundary(self,):
        pass
        
    def rhs(self, x):
        pass
    
#%% Teilaufgabe (ii): Solve Poisson
def solvePoisson(DGL, N=50, solver='Gauss'):
    u_ret = np.zeros((N+1,))
    u_ret[[0,N]] = 0
    x_disc = np.linspace(0,0)
    
    if N == 2:
        return x_disc, u_ret
    
    h = 0
    Ah = None
    F = None
    
    if solver == 'Gauss':
        u = None
        
    elif solver == 'Cholesky':
        u = None
    
    elif solver == 'numpy':
        u = None
    
    
    u_ret[1:-1] = u
    return u_ret, x_disc
    
#%% Teilaufgabe (iii): plotte Lösungen
plt.close('all')
DGL = Simple_DGL(a=-1,b=2.)
fig = plt.figure()
max_N = 8
N_range = [2**i for i in range(2,max_N)]

# plotte Löungen
# ...

#%% Teilaufgabe (iv): Laufzeiten
max_N = 12
N_range = [2**i for i in range(1,max_N)]
times_Gauss = np.zeros(len(N_range))
times_Cholesky = np.zeros(len(N_range))
times_numpy = np.zeros(len(N_range))

# Stoppe Zeit mit time
# ...

# Visualisiere Laufzeiten
# ...
    
