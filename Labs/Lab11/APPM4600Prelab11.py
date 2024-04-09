import numpy as np

def composite_Trapezoidal(a, b, f, N):
    h = (b-a)/N
    total = 0
    xj = a + h
    
    for j in range(N):
        total += f(xj)
        xj = a + (j+1)*h

    return (h/2) * (f(a) + 2*total + f(b))

def composite_Simpson(a, b, f, N):
    h = (b-a)/N
    total1 = 0
    total2 = 0
    x2j = a + 2*h
    x2j_1 = a + h

    for j in range(N//2 - 1):
        total1 += f(x2j)
        total2 += f(x2j_1)

        x2j = a + (2*(j+1))*h
        x2j_1 = a + (2*(j+1)-1)*h

    total2 += f(x2j_1)

    return (h/3)*(f(a) + 2*total1 + 4*total2 + f(b))