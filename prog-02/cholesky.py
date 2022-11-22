# %%
import numpy as np
import time
import matplotlib.pyplot as plt

# %% Implentierung des Cholesky Algorithmus
def issymmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def cholesky_slow(A):
    """cholesky_slow
    ========
    Implementierung der Cholesky-Zerlegung nach Algorithmus 1.36 
    der Vorlesung, unoptimiert.
    ----------
    Arguments:
        A (np.array): A positive definite Matrix. 
    
    Returns:
        L (np.array): Untere Dreiecksmatrix.
    """ 
    sizeA = np.shape(A)

    if sizeA[0] != sizeA[1]:
        raise ValueError("A ist nicht quadratisch.")
    if not issymmetric(A):
        raise ValueError("A ist nicht symmetrisch.")
    
    n = sizeA[0]
    L = np.zeros((n, n))

    for j in range(0, n):
        tmp = A[j,j]
        for i in range(j):
            tmp += -L[j, i]**2
        # überprüfe ob der Radikand positiv ist
        if tmp <= 0.:
            raise RuntimeError("A ist nicht positiv definit!")
        
        L[j, j] = np.sqrt(tmp)
        if j == 0:
            for i in range(1,n):
                L[i,0] = A[i,0] / L[0,0]
        else:
            for i in range(j+1, n):
                L[i,j] = A[i,j]
                for k in range(j):
                    L[i,j] +=  -L[i,k] * L[j,k]
                L[i,j] *= 1/ L[j, j]
    return L


def cholesky(A):
    """cholesky
    ========
    Implementierung der Cholesky-Zerlegung nach Algorithmus 1.36 
    der Vorlesung, optimiert.
    ----------
    Arguments:
        A (np.array): A positive definite Matrix. 
    
    Returns:
        L (np.array): Untere Dreiecksmatrix.
    """ 
    sizeA = np.shape(A)

    if sizeA[0] != sizeA[1]:
        raise ValueError("A ist nicht quadratisch.")
    if not issymmetric(A):
        raise ValueError("A ist nicht symmetrisch.")
    
    n = sizeA[0]
    L = np.zeros((n, n))

    for j in range(0, n):
        tmp = A[j, j] - np.sum(L[j,:j]**2)
        # überprüfe ob der Radikand positiv ist
        if tmp < 0:
            raise RuntimeError("A ist nicht positiv definit!")
        
        L[j, j] = np.sqrt(tmp)
        if j == 0:
            L[1:,0] = A[1:,0] / L[0,0]
        else:
            L[j+1:,j] = (A[j+1:,j] - L[j+1:,:j] @ L[j,:j]) / L[j, j]
    return L

def random_spd_matrix(N, delta=0.):
    A = np.random.uniform(-1, 1, (N,N))    
    B = 0.5*(A + A.T) + delta * np.eye(N)
    return B


#%% Teilaufgabe (ii)
A = np.array([[1, 2, 2], [2, 5, 6], [2, 6, 9]], dtype=float)
L_slow = cholesky_slow(A)
L = cholesky(A)
diff = np.linalg.norm(A - L@L.T)
diff_slow = np.linalg.norm(A - L_slow@L_slow.T)
print('Differenz zwischen A und der Cholesky-Zerlegung: ' + \
      str(diff))
    
print('Differenz zwischen A und der langsamen Cholesky-Zerlegung: ' + \
      str(diff))

#%% Teilaufgabe (iii)
def LaplaceOperator(N):
    return (2*np.eye(N-1)-np.eye(N-1,k=-1)-np.eye(N-1,k=1))/(N**2)

LM = LaplaceOperator(15)
LLM = cholesky(LM)
diff = np.linalg.norm(LM - LLM@LLM.T)
print('Differenz zwischen LM und der Cholesky-Zerlegung: ' + \
      str(diff))
    
plt.close('all')
plt.figure()
plt.imshow(LLM)
plt.title('Struktur der Matrix Lh')


#%% Teste langsame Cholesky-Zerlegung
testrange = [2**i for i in range(1,10)]
elapsedtime = np.zeros(np.size(testrange))

for i, n in enumerate(testrange):
    LM = LaplaceOperator(n)

    starttime = time.perf_counter()
    L = cholesky_slow(LM)
    elapsedtime[i] = time.perf_counter() - starttime
    
    print(f'Lauf: {i+1} (n={n}, time={np.round(elapsedtime[i], 5)}s)')
  
print('Finished Calculation')

# %% Plotte Rechenaufwand
plt.close('all')
fig, ax = plt.subplots(1,2)

p = np.polyfit(np.log(testrange), np.log(elapsedtime), deg=1)
print('Die Rechenzeit skaliert mit Potenz: ' + str(p[0]))

ax[0].plot(testrange, elapsedtime)
ax[1].loglog(testrange, elapsedtime, label='Time')
ax[1].loglog(testrange, testrange**p[0]*np.exp(p[1]), label='Fit')
ax[1].legend()

ax[0].set_xlabel(r"Dimension der Matrix")
ax[1].set_xlabel(r"Dimension der Matrix")
plt.ylabel(r"Berechnungszeit in Sekunden")

plt.show()

#%% Teste schnelle Cholesky-Zerlegung
testrange = [2**i for i in range(4,13)]
elapsedtime = np.zeros(np.size(testrange))

for i, n in enumerate(testrange):
    LM = LaplaceOperator(n)

    starttime = time.perf_counter()
    L = cholesky(LM)
    elapsedtime[i] = time.perf_counter() - starttime
    
    print(f'Lauf: {i+1} (n={n}, time={np.round(elapsedtime[i], 5)}s)')
  
print('Finished Calculation')

# %% Plotte Rechenaufwand
plt.close('all')
fig, ax = plt.subplots(1,2)

p = np.polyfit(np.log(testrange), np.log(elapsedtime), deg=1)
print('Die Rechenzeit skaliert mit Potenz: ' + str(p[0]))

ax[0].plot(testrange, elapsedtime)
ax[1].loglog(testrange, elapsedtime, label='Time')
ax[1].loglog(testrange, testrange**p[0]*np.exp(p[1]), label='Fit')
ax[1].legend()

ax[0].set_xlabel(r"Dimension der Matrix")
ax[1].set_xlabel(r"Dimension der Matrix")
plt.ylabel(r"Berechnungszeit in Sekunden")

plt.show()
