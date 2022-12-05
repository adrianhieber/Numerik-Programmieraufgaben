import numpy as np
import substitution_msl
import matplotlib.pyplot as plt


def qr_householder(A, mode):

    # check for Matrix A
    m = A.shape[0]
    n = A.shape[1]

    if m < n:
        raise Exception("for matrix A=Mat(mxn), m>=n")

    # calculate
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)

    for j in range(0, n):
        vs = R.copy()[j:m, j]  # warum muss hier ein copy hin??
        vs[0] = vs[0] - np.linalg.norm(vs)

        vs = vs / np.linalg.norm(vs)

        Qs = np.eye(m - j, dtype=float) - 2 * np.array([vs]).T @ np.array([vs])

        R[j:m, j:n] = Qs.copy() @ R.copy()[j:m, j:n]
        if mode != "R":
            Q[j:m, :] = Qs @ Q.copy()[j:m, :]

    # return based on mode
    if mode == "full":
        return Q.T, R
    if mode == "reduced":
        return Q.T[:m, :n], R[:n, :n]
    if mode == "R":
        return R[:n, :n]


def qr_givens(A, mode):
    pass


def qr(A, mode="full", alg="Householder"):
    if not isinstance(A, np.ndarray):
        raise Exception("usage: qr_householder(A, mode, alg), A has to be numpy array")

    if alg == "Householder":
        return qr_householder(A, mode)
    if alg == "Givens":
        return qr_givens(A, mode)

    raise Exception("Unknwon algortihm")


def function_values(m, n):
    interval = [-3, 3]

    xi = np.linspace(interval[0], interval[1], m)
    epsi = np.random.normal(0, 0.05, m)
    yi = [np.sin(3 * xi[i]) + xi[i] + epsi[i] for i in range(m)]

    V = np.vander(xi, n)
    Q, R = qr(V, mode="reduced")

    p = substitution_msl.backwardSubstitution(R, Q.T @ yi)

    return xi, yi, p


def reversed_horner(p, x):
    y = 0
    for a in p:
        y = y * x + a
    return y


def my_plot():
    fig = plt.figure()
    realy_plotted = False
    m = 50

    for n in [5, 9, 20]:
        xi, realy, p = function_values(m, n)

        if not realy_plotted:
            plt.plot(xi, realy, ".", label="y")
            realy_plotted = True

        ylist = [reversed_horner(p, x) for x in xi]
        plt.plot(xi, ylist, label=f"n={n}")

    plt.title(f"Comparison with m={m}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    my_plot()
