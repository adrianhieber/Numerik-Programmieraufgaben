import numpy as np
import substitution_msl
import matplotlib.pyplot as plt
from math import copysign, hypot


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

def qr_givens_online(A,mode):
    m, n = A.shape
    Q = np.eye(m)  # Initialize Q as the identity matrix
    R = A.copy()   # Initialize R as a copy of A
    
    # Iterate over the columns of R
    for i in range(n):
        # Iterate over the rows of R, starting from the i-th row
        for j in range(i+1, m):
            # Compute the Givens rotation matrix that zeros out the (j, i) element of R
            G = np.eye(m)
            r = np.sqrt(R[i, i]**2 + R[j, i]**2)
            c = R[i, i] / r
            s = -R[j, i] / r
            G[[i, j], [i, j]] = c
            G[i, j] = s
            G[j, i] = -s
            
            # Apply the Givens rotation to R and Q
            R = np.dot(G, R)
            Q = np.dot(Q, G.T)
            
    return Q[:m, :n], R[:n, :n]

def _givens_rotation_matrix_entries(a, b):
    """Compute matrix entries for Givens rotation."""
    r = hypot(a, b)
    c = a/r
    s = -b/r

    return (c, s)

def givens_rotation(A,mode):
    """Perform QR decomposition of matrix A using Givens rotation."""
    (num_rows, num_cols) = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = _givens_rotation_matrix_entries(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return Q[:num_rows, :num_cols], R[:num_cols, :num_cols]


def qr(A, mode="full", alg="givens_rotation"):
    if not isinstance(A, np.ndarray):
        raise Exception("usage: qr_householder(A, mode, alg), A has to be numpy array")

    if alg == "Householder":
        return qr_householder(A, mode)
    if alg == "givens_rotation":
        return givens_rotation(A, mode)

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
