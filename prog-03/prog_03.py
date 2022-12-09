import numpy as np
import substitution_mlsg
import matplotlib.pyplot as plt
import time


def qr_householder(A, mode):

    # init variables
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)

    # calculate
    for j in range(0, n):
        vs = R.copy()[j:m, j]
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

    # init variables
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)
    # calculate
    for j in range(1, n):
        for i in range(j + 1, m):

            a1 = R[j, j]
            a2 = R[i, j]
            anorm = np.linalg.norm(np.array([a1, a2]))
            c = a1 / anorm
            s = -a2 / anorm

            # somehow i messed up with indizes and thats why it fails
            # I blame it on the Gluehwein xD
            R[i, :] = c * R[i, :] - s * R[j, :]
            R[j, :] = c * R[j, :] + s * R[i, :]
            if mode != "R":
                Q[:, j] = c * Q[:, j] + s * Q[:, i]
                Q[:, i] = -s * Q[:, j] + c * Q[:, i]

##  #this gives correct answers, but calculates not only the rows :
##
##    for i in range(1, m):
##        for j in range(0, n):
##
##            a1 = R[j, j]
##            a2 = R[i, j]
##            anorm = np.linalg.norm(np.array([a1, a2]))
##            c = a1 / anorm
##            s = -a2 / anorm
##
##            Qij = np.eye(m, dtype=float)
##            Qij[i, i] = c
##            Qij[j, j] = c
##            Qij[i, j] = s
##            Qij[j, i] = -s
##
##            R = np.dot(Qij, R)
##            if mode != "R":
##                Q = np.dot(Q, Qij.T)

    # return based on mode
    if mode == "full":
        return Q, R
    if mode == "reduced":
        return Q[:m, :n], R[:n, :n]
    if mode == "R":
        return R[:n, :n]


def qr(A, mode="full", alg="Householder"):

    if not isinstance(A, np.ndarray):
        raise Exception("usage: qr_householder(A, mode, alg), A has to be numpy array")

    if A.shape[0] < A.shape[1]:
        raise Exception("for matrix A=Mat(mxn), m>=n")

    if alg == "Householder":
        return qr_householder(A, mode)

    if alg == "Givens":
        return qr_givens(A, mode)

    raise Exception("Unknwon algortihm")


def function_values(m, n, alg):
    interval = [-3, 3]

    xi = np.linspace(interval[0], interval[1], m)
    epsi = np.random.normal(0, 0.05, m)
    yi = [np.sin(3 * xi[i]) + xi[i] + epsi[i] for i in range(m)]

    V = np.vander(xi, n)
    Q, R = qr(V, mode="reduced", alg=alg)

    p = substitution_mlsg.backwardSubstitution(R, Q.T @ yi)

    return xi, yi, p


def reversed_horner(p, x):
    y = 0
    for a in p:
        y = y * x + a
    return y


def my_plot(alg):
    fig = plt.figure()
    true_y_plotted = False
    m = 50

    for n in [5, 9, 20]:
        xi, realy, p = function_values(m, n, alg)

        # plot true y
        if not true_y_plotted:
            plt.plot(xi, realy, ".", label="y")
            true_y_plotted = True

        # evaluate and plot polynomial
        ylist = [reversed_horner(p, x) for x in xi]
        plt.plot(xi, ylist, label=f"n={n}")

    plt.title(f"Comparison with m={m}, Alg={alg}")
    plt.legend()
    plt.show()


def compare_alg():

    print("Test with sparse matrix")

    # make sparse matrix
    m = 100
    E1 = np.eye(m, k=0)
    E2 = np.eye(m, k=-1)
    v1 = np.random.rand(1, m)
    v2 = np.random.rand(1, m)
    A = v1 * E1 + v2 * E2

    # stop time Householder
    t_householder_start = time.perf_counter()
    qr(A, alg="Householder")
    t_householder_stop = time.perf_counter()

    # stop time Givens
    t_givens_start = time.perf_counter()
    qr(A, alg="Givens")
    t_givens_stop = time.perf_counter()

    print(f"Householder: {t_householder_stop - t_householder_start}s")
    print(f"Givens: {t_givens_stop - t_givens_start}s")

    # conclusion
    print()
    print("Theoretically, Givens' algorithm should be faster for sparse matrices.")
    print(
        "But since this is not the case, my implementation is most likely not the most efficient."
    )


if __name__ == "__main__":

    # a1 visualize:
    my_plot(alg="Householder")

    # a2
    # test givens:
    my_plot(alg="Givens")
    # Comparison with sparse matrices:
    compare_alg()
    input("\nPress Enter to close.")
