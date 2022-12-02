import numpy as np


def qr(A, mode="full", alg="Householder"):
    #check alg
    if alg != "Householder":
        raise Exception("ich hab nur den algorithmus, also benutz ihn")

    # check for Matrix A
    m = A.shape[0]
    n = A.shape[1]
    if not isinstance(A, np.ndarray):
        raise Exception("Wasn willst du denn übergeben")
    if m < n:
        raise Exception("Oi was willstn du, m nix kleiner n!")

    #calculate
    Q = np.eye(m, dtype=float)
    R = np.array(A.copy(), dtype=float)

    for j in range(0, n):
        vs = R.copy()[j:m, j] #warum muss hier ein copy hin??
        vs[0] = vs[0] - np.linalg.norm(vs)
        vs = vs / np.linalg.norm(vs)
            
        Qs = np.eye(m - j, dtype=float) - 2 * np.array([vs]).T @ np.array([vs])
            
        R[j:m, j:n] = Qs.copy() @ R.copy()[j:m, j:n]
        if mode != "R":
            Q[j:m, :] = Qs @ Q.copy()[j:m, :]

    #return based on mode
    if mode == "full":
        return Q.T, R
    if mode == "reduced":
        return Q.T[:m,:n], R[:n,:n]
    if mode == "R":
        return R[:n,:n]


def test():
    A = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]])
    Q,R = qr(A)
    print()
    print("final")
    print("Q:")
    print(Q)
    print("R:")
    print(R)
test()
