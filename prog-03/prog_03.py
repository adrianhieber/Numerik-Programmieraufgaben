import numpy as np
import substitution_msl
import matplotlib.pyplot as plt


def qr_householder(A, mode):

    #init variables
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

    #init variables
    m = A.shape[0]
    n = A.shape[1]
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)

    #calculate
    for i in range(1, m):
        for j in range(0, n):

            a1 = R[j, j]
            a2 = R[i, j]
            anorm = np.linalg.norm(np.array([a1, a2]))
            c = a1 / anorm
            s = -a2 / anorm

            Qij = np.eye(m, dtype=float)
            Qij[i, i] = c
            Qij[j, j] = c
            Qij[i, j] = s
            Qij[j, i] = -s

            R = np.dot(Qij, R)
            if mode != "R":
                Q = np.dot(Q, Qij.T)
                
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
    Q, R = qr(V, mode="reduced",alg=alg)

    p = substitution_msl.backwardSubstitution(R, Q.T @ yi)

    return xi, yi, p


def reversed_horner(p, x):
    y = 0
    for a in p:
        y = y * x + a
    return y


def my_plot(alg):
    fig = plt.figure()
    realy_plotted = False
    m = 50

    for n in [5, 9, 20]:
        xi, realy, p = function_values(m, n,alg)

        if not realy_plotted:
            plt.plot(xi, realy, ".", label="y")
            realy_plotted = True

        ylist = [reversed_horner(p, x) for x in xi]
        plt.plot(xi, ylist, label=f"n={n}")

    plt.title(f"Comparison with m={m}, Alg={alg}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #a1 visualize:
    my_plot(alg="Householder")
