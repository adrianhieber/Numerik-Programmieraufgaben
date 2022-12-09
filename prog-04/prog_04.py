import numpy as np
from autograd import jacobian

def newton_like(f, x_0, dfz=None,x_1=None, tol=1e-8, max_iter=50, variant='standard'):
    #fange bullshit noch ab

    if variant=="standard":
        #use f.derivative(x)
        pass
    
    if variant=="secant":
        #x_1 has to be used
        pass

    if variant=="simple":
        #blatt 8
        #dfz is fixed
        #use f.derivative(x)
        pass

    
class func:
    __call__(self,x[]):
        return (x[0]**2,x[1]) #JUST TEST
    derivative(self, x):

def test():
    f1=func(1/np.tan)

if __name__=="__main__":
    test()
        
