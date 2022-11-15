import numpy as np


class Simple_DGL:
    def __init__(self, a=0.0, b=1.0):
        if a > b:
            raise Exception("a cant be bigger than b")
        self.a = a
        self.b = b

    def __call__(self, x):
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
    mdelta = (
        (1 / (h**2)) * 2 * np.identity(N - 1)
        - np.eye(N - 1, k=1)
        - np.eye(N - 1, k=-1)
    )
    if N < 0:
        raise Exception("N has to be positive")

    if solver == "Gauss":
        return  # TODO

    if solver == "Cholesky":
        return  # TODO

    if solver == "numpy":
        return np.linalg.solve(DGL)

    raise Exception("Unkown solver, choose from Gauss, Cholesky or numpy.")


def plot():
    print()
    # TODO


# plot()


def compare():
    print("Compare:")
    for i in range(1, 8):
        print()
