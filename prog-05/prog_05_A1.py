import numpy as np
import matplotlib.pyplot as plt
import time


def simple_polyval(x, p):
    return sum([p_i * x**i for i, p_i in enumerate(reversed(p))])


def Horner_polyval(x, a):
    sum = 0
    for i, p in enumerate(a):
        sum = sum * x + p
    return sum


def compare():
    # example
    x = 2.3
    p = [2, 5, 6, 8, 13, 7, 4, 25, 74]

    # measure
    simple_start = time.perf_counter()
    simple_polyval(x, p)
    simple_time = time.perf_counter() - simple_start

    horner_start = time.perf_counter()
    Horner_polyval(x, p)
    horner_time = time.perf_counter() - horner_start

    numpy_start = time.perf_counter()
    np.polyval(p, x)
    numpy_time = time.perf_counter() - numpy_start

    # plot
    compare_names = ["Simple", "Horner", "numpy"]
    compare_times = [simple_time, horner_time, numpy_time]
    fig = plt.figure()
    plt.bar(compare_names, compare_times)
    plt.xlabel("Method")
    plt.ylabel("Time in s")
    plt.title("Comparison with deg(p)=" + str(len(p) - 1))
    plt.show()


if __name__ == "__main__":
    compare()
