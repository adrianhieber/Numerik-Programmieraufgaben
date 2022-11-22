import numpy as np
import time
import matplotlib.pyplot as plt


class Simple_DGL:
    def __init__(self, a=0.0, b=1.0):
        if a > b:
            raise Exception("a cant be bigger than b")
        self.a = a
        self.b = b

    def __call__(self, x):
        # if x < self.a or x> self.b:
        #    raise Exception("x not allowed: a<=x<=b")
        return x**4 - 3 * x**3 + 2 * x**2 + 1

    def boundary(self):
        return [self(self.a), self(self.b)]

    def rhs(self, x):
        new = (self(x + 1) - 2 * self(x) + self(x - 1)) / (1 * 1)
        delta = np.inf
        for hpot in range(1, 8):
            h = 10 ** (-hpot)
            old = new
            new = (self(x + h) - 2 * self(x) + self(x - h)) / (h * h)
            if abs(new - old) > delta:
                return old
            delta = abs(new - old)
        return new


def solvePoisson(DGL, N=50, solver="Gauss"):
    if N < 0:
        raise Exception("N has to be positive")

    mdelta = (
        (1 / (h**2)) * 2 * np.identity(N - 1)
        - np.eye(N - 1, k=1)
        - np.eye(N - 1, k=-1)
    )

    Fh = np.zeros(N - 1)
    for i in range(0, N):
        Fh[i] = DGL(i)

    if solver == "Gauss":
        return  # TODO

    if solver == "Cholesky":
        return  # TODO

    if solver == "numpy":
        return np.linalg.solve(DGL)

    raise Exception("Unkown solver, choose from Gauss, Cholesky or numpy.")


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

def backSubstitution(R, y): #helper gauss
    __checkMatrix(R, y, True)
    # has to be checked (can be called at any time, not only by solveLES)
    n = np.size(y)
    x = np.empty(n)
    for i in reversed(range(0, n)):
        x[i] = (y[i] - sum(R[i, j] * x[j] for j in range(i + 1, n))) / R[i, i]
    return x


def Gauss(A, b):
    __checkMatrix(A, b, False)
    n = np.size(b)
    l = np.empty([n, n])
    P = np.identity(n)
    R = A.copy().astype(float)
    b = b.copy().astype(float)
    for j in range(0, n - 1):
        for i in range(j + 1, n):
            if R[j, j] == 0:
                R, swapI = __pivotSearchSwap(R, j)  # swap R
                P[[swapI, j], :] = P[[j, swapI], :]  # swap P
                b[swapI], b[j] = b[j], b[swapI]  # swap b
            l[i, j] = R[i, j] / R[j, j]
            for k in range(j + 1, n):
                R[i, k] = R[i, k] - l[i, j] * R[j, k]
            R[i, j] = 0
            b[i] = b[i] - l[i, j] * b[j]
    return backSubstitution(R, b), P, P @ A @ np.linalg.inv(R), R


def plotit():
    print()
    for i in range(2, 6):
        print()
    # TODO


# plot()


def compare():
    print("Compare:\n")
    potNstart = 1
    potNend = 8
    varnames = ["Gauss", "Cholesky", "numpy"]
    var = np.empty(len(varnames))
    for i in range(0, len(varnames)):
        #var[i] = np.empty(potNend - potNstart)
        print()

    for pot in range(potNstart, potNend):
        N = 2**pot
        print("\t N=", N)
        for nameind in range(0, len(var)):
            start = time.time()
            # solvePoisson(DGL,N=N,solver=varnames[nameind])
            runtime = time.time() - start
            print(f"\t\t {varnames[nameind]}: {runtime}s")

    x = range(potNstart, potNend)
    y = var[0]

    fig = plt.figure()
    ax.plot(x, y, linestyle=":", color="r")
    plt.show()
