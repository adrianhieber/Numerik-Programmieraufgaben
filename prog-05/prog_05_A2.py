import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from prog_05_A1 import Horner_polyval as horner
from math import prod

# TODO schon vorhandende Punkte einfuegen

class Vandermonde_model:
    def fit(self, x, y):
        vander = np.vander(x, increasing=True)
        self.p = np.flip(np.linalg.solve(vander, y))
        self.x = x
        self.y = y

    def __call__(self, x):
        return horner(x, self.p)

    def add_points(self, x, y):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        return self.fit(self.x,self.y)


class Lagrange_model:
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.w = [
            prod(1 / (x[k] - x[i]) if i != k else 1 for i in range(len(x)))
            for k in range(len(x))
        ]

    def __call__(self, x):
        if x in self.x:
            return self.y[self.x.index(x)]
        return sum(
            [self.y[k] * (self.w[k] / (x - self.x[k])) for k in range(len(self.y))]
        ) / sum([self.w[k] / (x - self.x[k]) for k in range(len(self.y))])

    def add_points(self, x, y):
        meas_num_old = len(self.x)
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        mult_to_w = [
            prod(
                1 / (self.x[k] - self.x[i]) if i != k else 1
                for i in range(meas_num_old, len(self.x))
            )
            for k in range(meas_num_old)
        ]
        self.w = np.array(self.w) * np.array(mult_to_w)
        w_to_app = [
            prod(
                1 / (self.x[k] - self.x[i]) if i != k else 1 for i in range(len(self.x))
            )
            for k in range(meas_num_old, len(self.x))
        ]
        self.w = np.append(self.w, w_to_app)


class Newton_model:
    def __helper__(self, k, i, y):
        if self.ls[k][i] != "_":
            return self.ls[k][i]
        if k==0:
            self.ls[k][i] = y[i]
            return self.ls[k][i]
        
        self.ls[k][i] = (self.__helper__(k-1,i+1, y) -self. __helper__(k-1, i, y)) / (self.x[i+k] - self.x[i])
        return self.ls[k][i]
    
    def fit(self, x, y):
        self.x = x
        self.ls = [["_" for i in range(len(x)-j)] for j in range(len(x))]
        self.f = [self.__helper__(k, 0, y) for k in range(len(self.ls))]
        
    def __call__(self, x):
        sum = 0
        n = len(self.f)
        for i, p in enumerate(reversed(self.f)):
            sum = sum * (x-self.x[n-1-i]) + p
        return sum
        

    def add_points(self, x, y):
        
        n_add = len(x)
        n_old = len(self.x)
        n = n_add + n_old
        self.x = np.append(self.x, x)

        # resize self.ls
        self.ls.extend([[] for i in range(n_add)])
        empty = ["_" for i in range(n_add)]
        self.ls[0].extend(y)
        for i in range(1, n_old):
            self.ls[i].extend(empty)
        for i in range(n_old, n_add+n_old):
            self.ls[i].extend(empty[:n-i])
        
        # fill
        self.f.extend([self.__helper__(k, 0, []) for k in range(n_old, n)])
        

def compare():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([3, -1, 4.2, 2.4, 9, -3.2, 0, 9.7])
    x_add = np.array([9])
    y_add = np.array([1.9])

    x_short = np.array([1, 2])
    y_short = np.array([3, -1])
    x_short_add = np.array([3, 4, 5, 6, 7, 8, 9])
    y_short_add = np.array([4.2, 2.4, 9, -3.2, 0, 9.7, 1.9])
    n_calls = 5
    calls = [rd.random() for i in range(n_calls)]

    methods = [Vandermonde_model(), Lagrange_model(), Newton_model()]
    methods_names = ["Vander", "Lagr", "Newt"]
    time_fit = np.empty(len(methods))
    time_call = np.empty(len(methods))
    time_add = np.empty(len(methods))

    for i, method in enumerate(methods):
        #fit
        time_start = time.perf_counter()
        method.fit(x,y)
        time_fit[i] = time.perf_counter() - time_start

        #call
        time_start = time.perf_counter()
        [method(call) for call in calls]
        time_call[i] = time.perf_counter() - time_start

        #addPoint
        time_start = time.perf_counter()
        method.add_points(x_add, y_add)
        time_add[i] = time.perf_counter() - time_start

    #Vandermonde
    #fit
    van = Vandermonde_model()
    time_van_fit_start = time.perf_counter()
    van.fit(x,y)
    time_van_fit = time.perf_counter() - time_van_fit_start

    #call
    time_van_call_start = time.perf_counter()
    [van(call) for call in calls]
    time_van_call = time.perf_counter() - time_van_call_start

    #addPoint
    time_van_add_start = time.perf_counter()
    van.add_points(x_add, y_add)
    time_van_add = time.perf_counter() - time_van_add_start

    #addPoints
    van = Vandermonde_model()
    van.fit(x_short,y_short)
    time_van_add2_start = time.perf_counter()
    van.add_points(x_add, y_add)
    time_van_add2 = time.perf_counter() - time_van_add2_start


    #Lagrange
    #fit
    lag = Lagrange_model()
    time_lag_fit_start = time.perf_counter()
    lag.fit(x,y)
    time_lag_fit = time.perf_counter() - time_lag_fit_start

    #call
    time_lag_call_start = time.perf_counter()
    [lag(call) for call in calls]
    time_lag_call = time.perf_counter() - time_lag_call_start

    #addPoint
    time_lag_add_start = time.perf_counter()
    lag.add_points(x_add, y_add)
    time_lag_add = time.perf_counter() - time_lag_add_start

    #addPoints
    lag = Lagrange_model()
    lag.fit(x_short,y_short)
    time_lag_add2_start = time.perf_counter()
    lag.add_points(x_add, y_add)
    time_lag_add2 = time.perf_counter() - time_lag_add2_start


    #Newton
    #fit
    newt = Newton_model()
    time_newt_fit_start = time.perf_counter()
    newt.fit(x,y)
    time_newt_fit = time.perf_counter() - time_newt_fit_start

    #call
    time_newt_call_start = time.perf_counter()
    [newt(call) for call in calls]
    time_newt_call = time.perf_counter() - time_newt_call_start

    #addPoint
    time_newt_add_start = time.perf_counter()
    newt.add_points(x_add, y_add)
    time_newt_add = time.perf_counter() - time_newt_add_start

    #addPoints
    newt = Newton_model()
    newt.fit(x_short,y_short)
    time_newt_add2_start = time.perf_counter()
    newt.add_points(x_add, y_add)
    time_newt_add2 = time.perf_counter() - time_newt_add2_start


    # plot
    compare_names = ["Vander", "Lagr", "Newt"]
    compare_times_fit = [time_van_fit, time_lag_fit, time_newt_fit]
    compare_times_call = [time_van_call, time_lag_call, time_newt_call]
    compare_times_add = [time_van_add, time_lag_add, time_newt_add] 
    compare_times_add2 = [time_van_add2, time_lag_add2, time_newt_add2] 
    compare_times_sum = [sum(x) for x in zip(compare_times_fit, compare_times_call, compare_times_add, compare_times_add2)]
    fig = plt.figure()

    ax1 = plt.subplot(2,3,1)
    ax1.bar(methods_names, time_fit)
    plt.ylabel("Time in s")
    plt.title("Fitting-Comp. deg(p)=" + str(len(x) - 1))

    ax2= plt.subplot(2,3,2)
    plt.bar(methods_names, time_call)
    plt.ylabel("Time in s")
    plt.title("Call-Comp. n_calls="+ str(n_calls))

    ax2= plt.subplot(2,3,3)
    plt.bar(methods_names, time_add)
    plt.ylabel("Time in s")
    plt.title("Add-Comp. len(v_add)=" + str(len(x_add)))

    ax2= plt.subplot(2,3,4)
    plt.bar(compare_names, compare_times_add2)
    plt.ylabel("Time in s")
    plt.title("Add-Comp. len(v_add)=" + str(len(x_short_add)))

    ax2= plt.subplot(2,3,5)
    plt.bar(compare_names, compare_times_sum)
    plt.ylabel("Time in s")
    plt.title("Total")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare()
