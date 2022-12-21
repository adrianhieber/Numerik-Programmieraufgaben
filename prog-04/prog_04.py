import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def newton_like(f, x_0, dfz=None, x_1=None, tol=1e-8, max_iter=50, delta=0.01, variant="standard"):

    #Wrong user input
    if variant not in ["secant", "simple", "standard"]:
        raise Exception("unkown algorithm")
    if variant == "simple" and (dfz==None).all():
        raise Exception("dfz needed for simple algorithm")
    if variant =="secant" and np.shape(x_0) == (1,) and x_1==None:
        raise Exception("enter a value for x_1")
    if variant == "secant" and np.shape(x_0) == (1,) and x_1[0]==x_0[0]:
        raise Exception("x_0 cant be equal to x_1")


    #Variant: secant
    if variant == "secant":
        
        #Dimension 1:
        if np.shape(x_0) == (1,):
            
            x_k_m1 = x_0.copy()
            x_k = x_1.copy()
            x_k_p1 = np.array([ x_k[0] - ( (x_k[0] - x_k_m1[0]) / (f(x_k)[0]-f(x_k_m1)[0]) ) * f(x_k)[0] ])
            n_iter = 1
        
            while np.linalg.norm(x_k_p1 - x_k) > tol and n_iter <= max_iter:
            
                x_k_m1 = x_k.copy()
                x_k = x_k_p1.copy()
                x_k_p1 = np.array([ x_k[0] - ( (x_k[0] - x_k_m1[0]) / (f(x_k)[0]-f(x_k_m1)[0]) ) * f(x_k)[0] ])
                n_iter += 1
                
            return x_k_p1, n_iter

        #Dimension n>1:
        n=np.shape(x_0)[0]
        x_k = x_0.copy()
        x_k_p1 = x_0.copy()
        
        J = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                ej=np.eye(n)[j]
                J[i][j]= (1/delta) * ( f(x_k+delta*ej)[i] - f(x_k)[i])
        x_k_p1 = np.linalg.solve(J.copy(),J@x_k-f(x_k))
        
        n_iter = 1

        while np.linalg.norm(x_k_p1 - x_k) > tol and n_iter <= max_iter:
            x_k = x_k_p1.copy()

            J = np.empty((n,n))
            for i in range(n):
                for j in range(n):
                    ej=np.eye(n)[j]
                    J[i][j]= (1/delta) * ( f(x_k+delta*ej)[i] - f(x_k)[i])
                  
            x_k_p1 = np.linalg.solve(J.copy(),J@x_k-f(x_k))
            n_iter +=1
        
        return x_k_p1, n_iter


    #Variant: standard or simple
            
    x_k = x_0.copy()

    if variant == "standard":
        dfz=f.derivative(x_k)
            
    x_k_p1 = np.linalg.solve(dfz.copy(), dfz* x_k - f(x_k))[0]
    n_iter = 1
        
    while np.linalg.norm(x_k_p1 - x_k) > tol and n_iter <= max_iter:
                
        x_k = x_k_p1.copy()

        if variant == "standard":
            dfz=f.derivative(x_k)
                
        if( np.linalg.det(dfz) == 0): #F' is close to 0
            return x_k_p1, n_iter
            
        x_k_p1 = np.linalg.solve(dfz.copy(), dfz* x_k - f(x_k))[0]
        n_iter += 1
            
    return x_k_p1, n_iter


class func:
    def __init__(self, f, f_der):
        self.f = f
        self.f_der = f_der

    def __call__(self, x):
        return self.f(x)

    def derivative(self, x):
        return self.f_der(x)


def test():
    
    # f1
    f_1 = lambda x: np.array( [(1 / np.tan(x[0]))] )
    f_1_der = lambda x: np.array([[(-1 / np.sin(x[0])**2)]])
    f_1_func = func(f=f_1, f_der=f_1_der)

    f_1_standard_start = time.perf_counter()
    f_1_standard, f_1_standard_iter = newton_like(f=f_1_func, x_0=np.array([1]),variant="standard")
    f_1_standard_time = time.perf_counter() - f_1_standard_start

    f_1_secant_start = time.perf_counter()
    f_1_secant, f_1_secant_iter = newton_like(f=f_1_func, x_0=np.array([1]), x_1=np.array([1.5]) ,variant="secant")
    f_1_secant_time = time.perf_counter() - f_1_secant_start

    f_1_simple_start = time.perf_counter()
    f_1_simple, f_1_simple_iter = newton_like(f=f_1_func, x_0=np.array([1]), dfz=f_1_func.derivative(np.array([1])), variant="simple")
    f_1_simple_time = time.perf_counter() - f_1_simple_start

    print(f"f_1_standard\n{f_1_standard}\n")
    print(f"f_1_secant\n{f_1_secant}\n")
    print(f"f_1_simple\n{f_1_simple}\n")

          
    # f2
    f_2 = lambda x: np.array([np.sin(x[0]), np.cos(x[1])])
    f_2_der = lambda x: np.array( [ [np.cos(x[0]), 0], [0,-np.sin(x[1])] ])
    f_2_func = func(f=f_2, f_der=f_2_der)

    f_2_standard_start = time.perf_counter()
    f_2_standard , f_2_standard_iter= newton_like(f=f_2_func, x_0=np.array([0,1]),variant="standard")
    f_2_standard_time = time.perf_counter() - f_2_standard_start

    f_2_secant_start = time.perf_counter()
    f_2_secant, f_2_secant_iter = newton_like(f=f_2_func, x_0=np.array([0,1]), variant="secant")
    f_2_secant_time = time.perf_counter() - f_2_secant_start

    f_2_simple_start = time.perf_counter()
    f_2_simple, f_2_simple_iter = newton_like(f=f_2_func, x_0=np.array([0,1]), dfz=f_2_func.derivative(np.array([0,1])), variant="simple")
    f_2_simple_time = time.perf_counter() - f_2_simple_start

    print(f"f_2_standard\n{f_2_standard}\n")
    print(f"f_2_secant\n{f_2_secant}\n")
    print(f"f_2_simple\n{f_2_simple}\n")


    #f3
    n=37
    f_3 = lambda x: np.array([ np.exp(-(xi**2))-1 for xi in x])#np.array([np.sin(x[0]), np.cos(x[1])])
    f_3_der = lambda x: np.diag( [ -2*xi*np.exp(-(xi**2)) for xi in x])
    f_3_func = func(f=f_3, f_der=f_3_der)

    f_3_standard_start = time.perf_counter()
    f_3_standard, f_3_standard_iter = newton_like(f=f_3_func, x_0=np.array([1 for _ in range(n)]))
    f_3_standard_time = time.perf_counter() - f_3_standard_start

    f_3_secant_start = time.perf_counter()
    f_3_secant, f_3_secant_iter = newton_like(f=f_3_func, x_0=np.array([1 for _ in range(n)]),variant="secant")
    f_3_secant_time = time.perf_counter() - f_3_secant_start

    f_3_simple_start = time.perf_counter()
    f_3_simple, f_3_simple_iter = newton_like(f=f_3_func, x_0=np.array([1 for _ in range(n)]), dfz=f_3_func.derivative( np.array([1 for _ in range(n)])),variant="simple")
    f_3_simple_time = time.perf_counter() - f_3_simple_start

    print(f"f_3_standard\n{f_3_standard}\n")
    print(f"f_3_secant\n{f_3_secant}\n")
    print(f"f_3_simple\n{f_3_simple}\n")
    

    df_iter = pd.DataFrame({
    'var': ['standard', 'secant', 'simple'],
    'f1': [f_1_standard_iter, f_1_secant_iter, f_1_simple_iter],
    'f2': [f_2_standard_iter, f_2_secant_iter, f_2_simple_iter],
    'f3': [f_3_standard_iter, f_3_secant_iter, f_3_simple_iter]
    })

    df_time = pd.DataFrame({
    'var': ['standard', 'secant', 'simple'],
    'f1': [f_1_standard_time, f_1_secant_time, f_1_simple_time],
    'f2': [f_2_standard_time, f_2_secant_time, f_2_simple_time],
    'f3': [f_3_standard_time, f_3_secant_time, f_3_simple_time]
    })

    #f3 secant takes to long to compare
    df_time_manipulated = pd.DataFrame({
    'var': ['standard', 'secant', 'simple'],
    'f1': [f_1_standard_time, f_1_secant_time, f_1_simple_time],
    'f2': [f_2_standard_time, f_2_secant_time, f_2_simple_time],
    'f3': [f_3_standard_time, 0, f_3_simple_time]
    })
    
    fig, ax = plt.subplots(1,2)
    df_iter.plot.bar(x='var', ax=ax[0], title="Iteration")
    df_time_manipulated.plot.bar(x='var', ax=ax[1], title="Time( without f3_secant)")
    plt.show()



if __name__ == "__main__":
    test()

