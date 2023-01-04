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
        meas_num_old = len(self.x)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        mult_to_w = [
            prod(
                1 / (self.x[k] - self.x[i]) if i != k else 1
                for i in range(meas_num_old, len(self.x))
            )
            for k in range(meas_num_old)
        ]
        self.w = np.array(self.w) * np.array(mult_to_w)
        w_to_app = [
            prod(
                1 / (self.x[k] - self.x[i]) if i != k else 1 for i in range(len(self.x))
            )
            for k in range(meas_num_old, len(self.x))
        ]
        self.w = np.append(self.w, w_to_app)


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
    x_notall = np.array([-1, 0])
    y_notall = np.array([-1, 3])
    x = np.array([-1, 0, 2, 3])
    y = np.array([-1, 3, 11, 27])

    x_test = np.array([-2, 0.5, 1, 4])
    y_test = np.array([-13, 3.875, 5, 59])

    model = [Vandermonde_model(), Lagrange_model()]
    for mode in model:
        print(mode)

        # standart
        mode.fit(x, y)
        if [np.isclose(mode(x_test[i]), y_test[i]) for i in range(len(x_test))] == [
            True for i in range(len(x_test))
        ]:
            print("Stutzwerttest standart erfolgreich")
        else:
            print("Stutzwerttest satndart failed")
            print([mode(x_test[i]) for i in range(len(x_test))])
            print("exp:", y_test)

        # add
        mode.fit(x_notall, y_notall)
        mode.add_points([2, 3], [11, 27])
        if [np.isclose(mode(x_test[i]), y_test[i]) for i in range(len(x_test))] == [
            True for i in range(len(x_test))
        ]:
            print("Stutzwerttest add erfolgreich")
        else:
            print("Stutzwerttest add failed")
            print([mode(x_test[i]) for i in range(len(x_test))])
            print("exp:", y_test)

        print()


if __name__ == "__main__":
    test_stutzwerte()
