import numpy as np
import time

# %%
class Timer(object):
    """Timer
    ========
    Eine Timer Klasse um diese als Context Manager zu verwenden. 
    Context Manager können mit dem python keyword "with" verwendet werden. 
    Diese greifen auf interne __enter__() und __exit__() Funktionen zurück.
    """ 
    
    def __init__(self, name=None):
        self.name = name
        self.tstart = 0.
        self.exec_time = 0.

    def __enter__(self):
        self.tstart = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.exec_time = time.perf_counter() - self.tstart
        used_name = f' ["{self.name}"]' if self.name else ""
        print(used_name + " finished in: " + "{:10.6f}".format(self.exec_time)\
                        + " seconds")
#%%
def issymmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)

#%%
def check_input(A, b, criterions=[], tol=1e-12):
    # Überprüfe ob A zweidimensional
    if ('2-dim' in criterions) and (np.ndim(A) != 2):
        raise(ValueError("Die Matrix ist nicht zwei-dimensional!"))

    # Überprüfe ob A quadratisch
    if ('quadratic' in criterions) and (A.shape[0] != A.shape[1]): 
        raise(ValueError("Die Matrix ist nicht quadratisch."))
        
    # Überprüfe ob A obere Dreiecksmatirx
    if ('triangular' in criterions) and np.any(np.tril(A, -1)): 
        raise(ValueError("Die Matrix ist keine obere Diagonalmatrix."))
        
    # Überprüfe ob A nullen auf der Diagonale hat
    if ('zero-diag' in criterions) and (np.min(np.abs(np.diag(A)))<tol):
        raise(ValueError("Die Matrix hat Nullen auf der Diagonale."))

    # Überprüfe, dass b im richtigen Format ist
    if np.ndim(b) != 1:
        raise(ValueError("Der Vektor soll Shape (n,) haben!"))
        
    # Überprüfe, dass und b kompatibel sind.
    if b.shape[0] != A.shape[1]:
        raise(ValueError("Matrix und Vektor sind nicht kompatibel!"))

#%% backSubstitution slow
# Aufgabe 1a
def backSubstitution_slow(R, b):
    """backSubstitution_slow
    ========
    Implementiert das Rückwärtseinsetzten zum Lösen eines LGS (langsam).
    ----------
    Arguments:
        R (np.array): Untere Dreiecksmatrix.
        b (np.array): Rechte Seite.
    
    Returns:
        x (np.array): Lösungsvektor.
    """ 
    
    # Überprüfe notwendige Eigenschaften der Variablen R und b
    check_input(R,b, criterions=['2-dim', 'quadratic', 'zero-diag', 'triangular'])
    n = np.shape(R)[0]
    
    # initialisiere ein Vektor, welcher später zurückgegeben wird.
    x = np.zeros(n)

    # berechne letzten Eintrag von x
    x[-1] = b[-1] /  R[-1, -1]

    # durchlaufe die übergebenen Gleichungen in umgekehrter Reihenfolge
    # und, ziehe den jeweiligen Anteil der bereits berechneten Variablen
    # von aktueller Gleichung ab und berechne die aktuelle Variable.
    for i in range(n-2, -1, -1):
        tmp = 0
        for j in range(i,n):
            tmp += R[i, j] * x[j]
            
        x[i] = (b[i] - tmp) / R[i, i]

    # gib Lösungsvektor x zurück
    return x

def backSubstitution(R, b):
    """backSubstitution
    ========
    Implementiert das Rückwärtseinsetzten zum Lösen eines LGS.
    ----------
    Arguments:
        R (np.array): Untere Dreiecksmatrix.
        b (np.array): Rechte Seite.
    
    Returns:
        x (np.array): Lösungsvektor.
    """ 
    n = np.shape(R)[0]
    # Überprüfe notwendige Eigenschaften der Variablen R und b
    check_input(R,b, criterions=['2-dim', 'quadratic', 'zero-diag', 'triangular'])
    
    # initialisiere ein Vektor, welcher später zurückgegeben wird.
    x = np.zeros(n)

    # berechne letzten Eintrag von x
    x[-1] = b[-1] /  R[-1, -1]

    # durchlaufe die übergebenen Gleichungen in umgekehrter Reihenfolge
    # und, ziehe den jeweiligen Anteil der bereits berechneten Variablen
    # von aktueller Gleichung ab und berechne die aktuelle Variable.
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - R[i, i:]@x[i:]) / R[i, i]

    # gib Lösungsvektor x zurück
    return x


#%%
def solveLES_slow(A, b, tol=1e-12):
    """solveLES
    ========
    
    ----------
    Arguments:

    
    Returns:
        
    """ 
    A = A.copy()
    b = b.copy()
    
    check_input(A,b, criterions=['2-dim', 'quadratic'])
    n = np.shape(A)[0]

    # Erzeuge Vektor, welcher die Permutation durch Pivotisierung
    # speichert
    p = np.arange(n)

    # Sei j die aktuell betrachtete Zeile: Durchlaufe alle nacheinander.
    for j in range(n):
        # Bestimme Zeile in der das größte Spaltenelement enthalten ist
        p_idx = np.argmax(np.abs(A[j:, j])) + j
        
        if np.abs(A[p_idx,j]) < tol:
            raise ValueError('No pivot element found in step ' + str(j) + ', matrix A is singular to working precision.')

        # Setze Zeilenpermutation
        p[[j, p_idx]] = p[[p_idx, j]]
        # Wende Permutation auf A und b an
        A[[j, p_idx], :] = A[[p_idx, j], :]
        b[[j, p_idx]]    = b[[p_idx, j]]        

        # Sei i alle Zeilen UNTERHALB der aktuellen Zeile j
        for i in range(j+1, n):
            # Einträge unterhalb der Diagonalen in A werden sind nach
            # Verfahren immer 0, daher nutzen wir diese um die Matrix
            # L zu erzeugen. 
            # Schreibe Einträge von L unterhalb der Diagonale:
            A[i,j] = A[i,j] / A[j,j]
            for k in range(j+1, n):
                # Schreibe Werte von R auf bzw. oberhalb von Diagonale:
                A[i, k] = A[i, k] - A[i,j] * A[j, k]
            # Aktualisiere rechts Seite
            b[i] = b[i] - A[i,j] * b[j]
            
    # Erzeuge Einheitsmatrix und wende darauf die gespeicherte 
    # Permutation an 
    P = np.eye(n)
    P = P[p, :]

    # Erzeuge L aus den Skalierungswerten welche unterhalb der 
    # Diagonalen in der Matrix A gespeichert sind.
    L = np.tril(A, -1) + np.eye(n)

    # Erzeuge obere Dreieckmatrix für Rückwärtseinsetzen aus A
    R = np.triu(A)

    # Erzeugen des Lösungsvektors durch Rückwärtseinsetzen
    x = backSubstitution_slow(R, b)
    return x, P, L, R

#%%
def solveLES(A, b, tol=1e-12):
    """solveLES
    ========
    
    ----------
    Arguments:

    
    Returns:
        
    """ 
    A = A.copy()
    b = b.copy()
    
    check_input(A,b, criterions=['2-dim', 'quadratic'])
    n = np.shape(A)[0]

    # Erzeuge Vektor, welcher die Permutation durch Pivotisierung
    # speichert
    p = np.arange(n)

    # Sei j die aktuell betrachtete Zeile: Durchlaufe alle nacheinander.
    for j in range(n):
        # Bestimme Zeile in der das größte Spaltenelement enthalten ist
        p_idx = np.argmax(np.abs(A[j:, j])) + j
    	
        if np.abs(A[p_idx,j]) < tol:
            raise ValueError('No pivot element found in step ' + str(j) + ', matrix A is singular to working precision.')

        # Speichere Zeilenpermutation in Permutationsvektor p
        p[[j, p_idx]] = p[[p_idx, j]]
        # Wende Permutation auf A und b an
        A[[j, p_idx], :] = A[[p_idx, j], :]
        b[[j, p_idx]]    = b[[p_idx, j]]        

        # Schreibe Einträge von L unterhalb der Diagonale:
        A[j+1:,j] *= 1/ A[j,j]
        # Eliminationsupdate
        A[j+1:, j+1:] += - A[j+1:,j:j+1] * A[j:j+1, j+1:]
        # Update rechte Seite
        b[j+1:] += - A[j+1:,j] * b[j]
        
 

    # Erzeuge Einheitsmatrix und wende darauf die gespeicherte 
    # Permutation an 
    P = np.eye(n)
    P = P[p, :]

    # Erzeuge L aus den Skalierungswerten welche unterhalb der 
    # Diagonalen in der Matrix A gespeichert sind.
    L = np.tril(A, -1) + np.eye(n)

    # Erzeuge obere Dreieckmatrix für Rückwärtseinsetzen aus A
    R = np.triu(A)

    # Erzeugen des Lösungsvektors durch Rückwärtseinsetzen
    x = backSubstitution(R, b)
    return x, P, L, R
#%%
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

#%%
def random_matrix(n, delta=0.1):
    return np.random.uniform(-1,1, (n,n)) + delta * np.eye(n)

def random_vector(n):
    return np.random.uniform(-1,1, (n,))
