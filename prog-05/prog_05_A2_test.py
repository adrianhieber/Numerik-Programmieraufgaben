import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from prog_05_A1 import Horner_polyval as horner
from math import prod


def test_stutzwerte():
    x_notall = np.array([-1, 0])
    y_notall = np.array([-1, 3])
    x = np.array([-1, 0, 2, 3])
    y = np.array([-1, 3, 11, 27])

    x_test = np.array([-2, 0.5, 1, 4])
    y_test = np.array([-13, 3.875, 5, 59])

    model = [Vandermonde_model(), Lagrange_model(), Newton_model()]
    for mode in model:
        print(mode)

        # standart
        time_start = time.perf_counter()
        mode.fit(x, y)
        time_mode = time.perf_counter() - time_start
        print("Time fitting: ",time_mode,"s")
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

