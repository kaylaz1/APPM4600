import numpy as np
    
def fixedpt(f,x0,tol,Nmax):
    a = 0
    x = np.zeros((Nmax, 1))

    count = 0
    while (count < Nmax):
       x1 = f(x0)
       if (abs(x1-x0) < tol and a == 0):
          xstar = x1
          ier = 0
          a = count
       x0 = x1
       x[count] = x1
       count += 1

    xstar = x1
    ier = 1
    return (x, a)

f = lambda x: (10/(x+4))**(1/2)
tol = 10**(-10)
Nmax = 1000

x0 = 1.5
(x, a) = fixedpt(f, x0, tol, Nmax)
print(a)

import math
print(math.log(abs(x[11]-x0),abs(x[10]-x0)))