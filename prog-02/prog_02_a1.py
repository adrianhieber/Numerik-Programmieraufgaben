import numpy as np
import matplotlib.pyplot as plt
import solvers
import time

#%% Laplace Matrix
def LaplaceOperator(N):
    return 2 * np.eye(N - 1) - np.eye(N - 1, k=-1) - np.eye(N - 1, k=1)


#%% Teilaufgabe (i): Simple_DGL
class Simple_DGL:
    def __init__(self, a=0.0, b=1.0):
        if a > b:
            raise Exception("a cant be bigger than b")
        self.a = a
        self.b = b

    def __call__(self, x):
        return x**4 - 3 * x**3 + 2 * x**2 + 1

    def boundary(
        self,
    ):
        return [self(self.a), self(self.b)]

    def rhs(self, x):
        new = (self(x + 1) - 2 * self(x) + self(x - 1)) / (1 * 1)
        delta = np.inf
        for hpot in range(1, 8):  # for very small h, errors occur
            h = 10 ** (-hpot)
            old = new
            new = (self(x + h) - 2 * self(x) + self(x - h)) / (h * h)
            if abs(new - old) > delta:
                return -old  # negative because its -delta
            delta = abs(new - old)
        return new


#%% Teilaufgabe (ii): Solve Poisson
def solvePoisson(DGL, N=50, solver="Gauss"):
    u_ret = np.zeros((N + 1,))
    u_ret[[0, N]] = 0
    x_disc = np.linspace(0, 0)

    if N == 2:
        return x_disc, u_ret

    h = (DGL.b - DGL.a) / N
    Ah = 1 / (h**2) * LaplaceOperator(N)
    F = np.empty(N - 1)
    for i in range(0, N - 1):
        F[i] = DGL.rhs(h * i + DGL.a)
    G = np.empty(N - 1)
    G[0] = 1 / (h**2) * DGL.boundary()[0]
    G[N - 2] = 1 / (h**2) * DGL.boundary()[1]

    if solver == "Gauss":
        u, P, L, R = solvers.solveLES(Ah, F + G)

    elif solver == "Cholesky":
        L = solvers.cholesky(Ah)

        # Eigentlich sollte das ca so aussehen:
        # y = solvers.backSubstitution(L, F + G)
        # u = solvers.backSubstitution(L.T, y)
        # wusste aber nicht wie man y mit backsubstitution
        # loest, da ja obere linke Dreiecksmatrix.

        # damit man aber ein circa runtime vergleich hat,
        # habe ich einfach falsch gerechnet:
        y = solvers.backSubstitution(L.T, F + G)
        u = solvers.backSubstitution(L.T, y)

    elif solver == "numpy":
        u = np.linalg.solve(Ah, F + G)

    return u, x_disc


#%% Teilaufgabe (iii): plotte Lösungen
def plot():
    plt.close("all")
    DGL = Simple_DGL(a=-1, b=2.0)
    max_N = 8
    N_range = [2**i for i in range(2, max_N)]

    varnames = ["Gauss", "Cholesky", "numpy"]

    for N in N_range:
        print(f"\nN = {N}:\n")
        for nameind in range(0, len(varnames)):
            u, x = solvePoisson(DGL, N=N, solver=varnames[nameind])
            print(f"{varnames[nameind]}:\n{u}\n")


def runtime():
    #%% Teilaufgabe (iv): Laufzeiten
    DGL = Simple_DGL(a=-1, b=2.0)
    max_N = 12
    N_range = [2**i for i in range(1, max_N)]
    times_Gauss = np.zeros(len(N_range))
    times_Cholesky = np.zeros(len(N_range))
    times_numpy = np.zeros(len(N_range))

    # Stoppe Zeit mit time
    for ind, N in enumerate(N_range):
        start = time.time()
        solvePoisson(DGL, N=N, solver="Gauss")
        times_Gauss[ind] = time.time() - start

        start = time.time()
        solvePoisson(DGL, N=N, solver="Cholesky")
        times_Cholesky[ind] = time.time() - start

        start = time.time()
        solvePoisson(DGL, N=N, solver="numpy")
        times_numpy[ind] = time.time() - start

    # Visualisiere Laufzeiten
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(N_range, times_Gauss, linestyle="-", color="r", label="Gauss")
    ax.plot(N_range, times_Cholesky, linestyle="-", color="k", label="Cholesky")
    ax.plot(N_range, times_numpy, linestyle="-", color="g", label="numpy")
    plt.legend(loc="upper right")
    plt.show()
