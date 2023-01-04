from prog_05_A1 import Horner_polyval as horner
import numpy as np
from math import prod


class Vandermonde_model:
    def fit(self, x, y):
        vander = np.vander(x, increasing=True)
        self.p = np.flip(np.linalg.solve(vander, y))

    def __call__(self, x):
        return horner(x, self.p)

    def add_points(self, x, y):
        # TODO
        pass


class Lagrange_model:
    def fit(self, x, y):
        # baryzentrische Gewicht
        self.x = x
        self.y = y
        self.w = [
            prod(1 / (x[k] - x[i]) if i != k else 1 for i in range(len(x)))
            for k in range(len(x))
        ]

    def __call__(self, x):
        return sum(
            [self.y[k] * (self.w[k] / (x - self.x[k])) for k in range(len(self.y))]
        ) / sum([self.w[k] / (x - self.x[k]) for k in range(len(self.y))])

    def add_points(self, x, y):
        # TODO
        pass


class Newton_model:
    def fit(self, x, y):
        # TODO
        pass

    def __call__(self, x):
        return horner(x, self.p)

    def add_points(self, x, y):
        # TODO
        pass


def test():
    x = np.array([-1, 0, 2, 3])
    y = np.array([-1, 3, 11, 27])
    mode = Lagrange_model()
    mode.fit(x, y)
    print(mode(0.5))
    print(mode(-2))
    print(mode(5))


def test_stutzwerte():
    x = np.array([-1, 0, 2, 3])
    y = np.array([-1, 3, 11, 27])

    x_test = np.array([-2, 0.5, 1, 4])
    y_test = np.array([-13, 3.875, 5, 59])

    model = [Vandermonde_model(), Lagrange_model()]
    for mode in model:
        print(mode)
        mode.fit(x, y)
        if [np.isclose(mode(x_test[i]), y_test[i]) for i in range(len(x_test))] == [
            True for i in range(len(x_test))
        ]:
            print("Stutzwerttest erfolgreich")
        else:
            print("Stutzwerttest failed")
            print([mode(x_test[i]) for i in range(len(x_test))])
            print("exp:", y_test)
        print()


if __name__ == "__main__":
    test_stutzwerte()
