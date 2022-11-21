import numpy as np
import matplotlib.pyplot as plt


def svd(A, variant="standard", t=None):
    if variant == "standard":
        u, s, v = np.linalg.svd(A)
        return u, np.diag(s), v.T

    if variant == "truncated":
        t = round(t)
        if t < 0 or t > min(A.shape[0], A.shape[1]):
            raise Exception(
                f"t={t} has to be positive integer less or equal min(m,n)={A.shape[0], A.shape[1]} for truncated"
            )
        u, s, v = np.linalg.svd(A)
        v = v.T
        s = np.diag(s)

        u = __my_resize(u, u.shape[0], t)
        s = __my_resize(s, t, t)
        v = __my_resize(v, v.shape[0], t)

        return u, np.diag(s), v

    if variant == "thin":
        return svd(A, variant="truncated", t=min(A.shape[0], A.shape[1]))

    if variant == "compact":
        r = len([x for x in np.linalg.eigvals(A.T @ A) if x >= 0])
        return svd(A, variant="truncated", t=r)

    raise Exception("no valid variant")


class compressed_img:
    coloranzahl = 2
    u = v = np.empty((1, 1, coloranzahl))
    s = np.empty((1, coloranzahl))

    def __init__(self, I, factor=1.0):
        nh = np.shape(I)[0]
        nw = np.shape(I)[1]
        type(self).coloranzahl = nc = np.shape(I)[2]
        t = (factor * nh * nw * nc) / (nc * (nh + 1 + nw))

        self.compress_svd(I, t)

    def compress_svd(self, I, t):
        for i in range(0, self.coloranzahl):
            ui, si, vi = svd(I[:, :, i], variant="truncated", t=t)

            type(self).u = np.resize(
                type(self).u, (ui.shape[0], ui.shape[1], self.coloranzahl)
            )
            type(self).s = np.resize(type(self).s, (si.shape[0], self.coloranzahl))
            type(self).v = np.resize(
                type(self).v, (vi.shape[0], vi.shape[1], self.coloranzahl)
            )

            type(self).u[:, :, i] = ui
            type(self).v[:, :, i] = vi
            type(self).s[:, i] = si

    def __call__(self):
        ret = np.empty((type(self).u.shape[0], type(self).v.shape[0], self.coloranzahl))
        for i in range(0, self.coloranzahl):
            ret[:, :, i] = (
                type(self).u[:, :, i]
                @ np.diag(type(self).s[:, i])
                @ type(self).v[:, :, i].T
            )
        ret[ret < 0] = 0
        ret[ret > 1] = 1
        return ret


def __my_resize(A, n, m):
    A = np.array(A, order="F")
    A.resize(A.shape[0], m)
    A = np.array(A, order="C")
    A.resize(n, A.shape[1])
    return A


def test():
    testpics = {"cat-1", "cat-2", "Lugano"}
    for pic in testpics:
        org = plt.imread(f"test-images/{pic}.png")

        fig = plt.figure()

        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(org)
        plt.title("Orginal")

        ax2 = plt.subplot(2, 2, 2)
        comp = compressed_img(org)()
        error = np.linalg.norm(org - comp) / np.linalg.norm(org)
        ax2.imshow(comp)
        plt.title(f"Compressed(factor=1.0)\n(Error={round(error,4)*100}%)")

        factor = 0.2
        ax3 = plt.subplot(2, 2, 3)
        comp2 = compressed_img(org, factor)()
        error = np.linalg.norm(org - comp2) / np.linalg.norm(org)
        ax3.imshow(comp2)
        plt.title(f"Compressed(factor={factor})\n(Error={round(error,4)*100}%)")

        factor = 0.01
        ax4 = plt.subplot(2, 2, 4)
        comp3 = compressed_img(org, factor)()
        error = np.linalg.norm(org - comp3) / np.linalg.norm(org)
        ax4.imshow(comp3)
        plt.title(f"Compressed(factor={factor})\n(Error={round(error,4)*100}%)")

        plt.suptitle(f"Imagename: {pic}")
        fig.tight_layout()
        plt.show()
