import numpy as np
import matplotlib.pyplot as plt


def svd(A, variant="standard", t=None):
    if variant == "standard":
        u, s, v = np.linalg.svd(A)
        return u, np.diag(s), v.T

    if variant == "truncated":
        if not isinstance(t, int) or t < 1 or t > min(A.shape[0], A.shape[1]):
            raise Exception(
                "t has to be positive integer less or equal min(m,n) for truncated"
            )
        u, s, v = np.linalg.svd(A)
        v = v.T
        s = np.diag(s)

        # cut
        u = np.array(u, order="F")
        orginal_size_1 = u.shape[1]
        u.resize(u.shape[0], t)
        u.resize(u.shape[0], orginal_size_1)

        orginal_size_1 = v.shape[1]
        v = np.array(v, order="F")
        v.resize(v.shape[0], t)
        v.resize(v.shape[0], orginal_size_1)

        orginal_size_0 = s.shape[0]
        orginal_size_1 = s.shape[1]
        s = np.array(s, order="F")
        s.resize(s.shape[0], t)
        s = np.array(s, order="C")
        s.resize(t, s.shape[1])
        s.resize(orginal_size_0, s.shape[1])
        s = np.array(s, order="F")
        s.resize(s.shape[0], orginal_size_1)

        return u, s, v.T

    if variant == "thin":
        return svd(A, variant="truncated", t=min(A.shape[0], A.shape[1]))

    if variant == "compact":
        r = len([x for x in np.linalg.eigvals(a) if x > 0])
        return svd(A, variant="truncated", t=r)

    raise Exception("no valid variant")


class compressed_img:
    def ___init___(self, I, factor=1.0):
        self.I = I
        self.factor = factor

    def compress_svd(self, I, t):
        print()
        # TODO


def test():
    a = np.array([[1, 2, 2], [4, 2, 6], [6, 8, 9]])
    u, s, v = svd(a, variant="truncated", t=2)
    print(u @ s @ (v.T))
