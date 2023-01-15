import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def quad(f, a=0, b=1.0, method="trapezoidal", num_ints=1):
    # Error Handling
    if a >= b:
        raise Exception("a has to be smaller than b")
    if not isinstance(num_ints, int):
        raise Exception("num_ints has to be an integer")
    if num_ints <= 0:
        raise Exception("num_ints has to be greater than 0")
    if not callable(f):
        raise Exception("f has to be callable")

    # Recursion
    if num_ints > 1:
        ls = np.linspace(a, b, num_ints + 1)
        return sum(
            [quad(f, a=ls[i], b=ls[i + 1], method=method) for i in range(0, num_ints)]
        )

    # Methods
    if method == "trapezoidal":
        return 0.5 * (b - a) * (f(a) + f(b))

    if method == "Simpson":
        return 1 / 6 * (b - a) * (f(a) + 4 * f(0.5 * (a + b)) + f(b))

    if method == "pulcherrima":
        return (b - a) * (
            f(a) * 0.125
            + 3 * f(a + (b - a) / 3) * 0.125
            + 3 * f(a + (b - a) * (2 / 3)) * 0.125
            + f(b) * 0.125
        )

    if method == "Milne":
        return (b - a) * (
            7 * f(a) / 90
            + 32 * f(a + (b - a) / 4) / 90
            + 12 * f(a + (b - a) / 2) / 90
            + 32 * f(a + (3 / 4) * (b - a)) / 90
            + 7 * f(b) / 90
        )

    if method == "six-point":
        return (b - a) * (
            19 * f(a) / 288
            + 75 * f(a + (b - a) / 5) / 288
            + 50 * f(a + (b - a) * (2 / 5)) / 288
            + 50 * f(a + (b - a) * (3 / 5)) / 288
            + 75 * f(a + (b - a) * (4 / 5)) / 288
            + 19 * f(b) / 288
        )

    if method == "Weddle":
        return (b - a) * (
            41 * f(a) / 840
            + 216 * f(a + (b - a) / 6) / 840
            + 27 * f(a + (b - a) / 3) / 840
            + 272 * f(a + (b - a) * 0.5) / 840
            + 27 * f(a + (b - a) * (2 / 3)) / 840
            + 216 * f(a + (b - a) * (5 / 6)) / 840
            + 41 * f(b) / 840
        )

    raise Exception("unknown method")


def my_eval():
    # functions
    f = lambda x: (x + 1) * np.sin(x)
    f_integral = lambda x: np.sin(x) - (x + 1) * np.cos(x)

    # initial
    a = 0
    b = 1
    num_ints = 1
    methods = ["trapezoidal", "Simpson", "pulcherrima", "Milne", "six-point", "Weddle"]

    # calculate abs error
    exp = f_integral(b) - f_integral(a)
    print(f"a={a}, b={b} \nexp = {exp}\n")
    methods_errors = np.empty(len(methods))
    for i, method in enumerate(methods):
        act = quad(f, a=a, b=b, method=method, num_ints=num_ints)
        methods_errors[i] = abs(act - exp) / abs(exp) * 100
        print(f"{method} = {act}")
        print(f"Abs Error = {methods_errors[i]}%\n")

    # graph
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    (l,) = plt.plot(methods, methods_errors)
    plt.ylabel("Abs Error in %")
    plt.xlabel("Method")

    # make axes for slider
    ax_a = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_b = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_num_ints = plt.axes([0.25, 0.05, 0.65, 0.03])

    # make slider
    a_b_max = 7
    a_change = Slider(ax_a, "a", -a_b_max, a_b_max, a, valstep=0.2)
    b_change = Slider(ax_b, "b", -a_b_max, a_b_max, b, valstep=0.2)
    num_ints_change = Slider(ax_num_ints, "num_ints", 1, 10, num_ints, valstep=1)

    def update(val):
        # get values
        a = a_change.val
        b = b_change.val
        num_ints = num_ints_change.val

        # don't eval for wrong input
        if a >= b:
            return

        # calculate abs error
        for i, method in enumerate(methods):
            act = quad(f, a=a, b=b, method=method, num_ints=num_ints)
            methods_errors[i] = abs(act - exp) / abs(exp) * 100
        l.set_ydata(methods_errors)

        # arrange y axes
        max_y_plot = max(methods_errors) + 0.1 * (
            max(methods_errors) - min(methods_errors)
        )
        min_y_plot = min(methods_errors) - 0.1 * (
            max(methods_errors) - min(methods_errors)
        )
        ax.set_ylim([min_y_plot, max_y_plot])

    # make slider work
    a_change.on_changed(update)
    b_change.on_changed(update)
    num_ints_change.on_changed(update)

    plt.show()


if __name__ == "__main__":
    my_eval()

