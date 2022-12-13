import numpy as np

# from autograd import jacobian


def newton_like(f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50, variant="standard"):
    # fange bullshit noch ab

    if variant == "standard":
        # TODO pruefen ob konvergiert
        x_k = x_0
        x_k2 = x_k - f(x_k) / f.derivative(x)
        n_iter = 0
        while abs(x_k2 - x_k) > tol and n_iter < max_iter:
            x_k = x_k2 if x_k != x_0 else x_0
            x_k2 = x_k - f(x_k) / f.derivative(x)
            n_iter += 1
        return x_k2

    if variant == "secant":
        x_k_m1 = x_0
        x_k = x_1
        x_k_p1 = x_k - ((x_k - x_k_m1) / (f(x_k) - f(x_k_m1))) *f(x_k)
        n_iter = 0
        
        while abs(x_k_p1 - x_k) > tol and n_iter < max_iter:
            
            x_k_m1 = x_k
            x_k = x_k_p1 
            x_k_p1 = x_k - ((x_k - x_k_m1) / (f(x_k) - f(x_k_m1))) *f(x_k)
            n_iter += 1
                
        print("Iterations:",n_iter)
        return x_k_p1

    if variant == "simple":
        # blatt 8
        # dfz is fixed
        # use f.derivative(x)
        pass


class func:
    def __init__(self, f, n, m):
        self.f = f
        self.n = n
        self.m = m

    def __call__(self, x):
        return self.f(x)

    def derivative(self, x):
        def helper(i, j):
            new_x = x.copy()
            new_x[j] += 1
            new = self.f(new_x)[i] - self.f(x)[i]
            eps = np.inf
            for hpot in range(1, 8):  # for very small h, errors occur
                h = 10 ** (-hpot)
                new_x = x.copy()
                new_x[j] += h
                old = new
                new = (self.f(new_x)[i] - self.f(x)[i]) / h
                if abs(new - old) > eps:
                    return old
                eps = abs(new - old)
            return new

        J = np.empty((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                J[i, j] = helper(i, j)
        return J


def test():
    # f: R^n->R^m
    # mytest
    f_abc = func(f=lambda x: (x[0], x[1], x[0] + x[1]), n=2, m=3)

    # f1
    f_1 = func(f=lambda x: (1 / np.tan(x)), n=1, m=1)

    # f2
    f_2 = func(f=lambda x: (np.sin(x[0] + np.cos(x[2]))), n=2, m=1)

    # f3
    n = 3  # just test
    f_2 = func(
        f=lambda x: (sum([np.exp(-(x[i] ** 2)) - n for i in range(n)])), n=n, m=1
    )

    f1_test=newton_like(f=f_1, x_0=4, x_1=8, variant="secant")
    print(f1_test)


if __name__ == "__main__":
    test()
