import numpy as np
import substitution_msl
import matplotlib.pyplot as plt

# just for test, delete after
def isNaN(num):
    return num != num


def qr_householder(A, mode):

    # check for Matrix A
    m = A.shape[0]
    n = A.shape[1]
    if not isinstance(A, np.ndarray):
        raise Exception("Wasn willst du denn übergeben")
    if m < n:
        raise Exception("Oi was willstn du, m nix kleiner n!")

    # calculate
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)

    for j in range(0, n):
        vs = R.copy()[j:m, j]  # warum muss hier ein copy hin??
        vs[0] = vs[0] - np.linalg.norm(vs)

        ###warum zum teufel kann vs norm 0 oder nan
        if isNaN(np.linalg.norm(vs)) or np.linalg.norm(vs) == 0:
            print(vs)
            print(np.linalg.norm(vs))
            raise Exception("tf")

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
    if alg == "Householder":
        return qr_householder(A, mode)
    if alg == "Givens":
        return qr_givens(A, mode)

    raise Exception("Unknwon algortihm")


def aii(m, n):
    interval = [-3, 3]

    xi = np.linspace(interval[0], interval[1], m)
    epsi = np.random.normal(0, 0.05, m)
    yi = [np.sin(3 * xi[i]) + xi[i] + epsi[i] for i in range(m)]

    V = np.vander(xi, n)
    Q, R = qr(V, mode="reduced")

    p = substitution_msl.backwardSubstitution(R, Q.T @ yi)

    return xi, p


def my_plot():
    fig = plt.figure()
    # for i in range(5,10):
    # n=2**i
    xi, p = aii(100, 50)
    plt.plot(xi, p, label=50)
    ###hä irgendwie machts ja kein sinn xi mit m elementen und p mit n-1 elementen zu plotten
    plt.legend()
    plt.show()


if __name__ == "__main__":
    my_plot()
