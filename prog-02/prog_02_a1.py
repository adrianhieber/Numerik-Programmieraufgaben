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
        var[i] = np.empty(potNend - potNstart)

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
