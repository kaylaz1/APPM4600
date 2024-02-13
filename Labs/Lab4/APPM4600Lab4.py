import numpy as np

def fixedpt(f,x0,tol,Nmax):
    x = np.zeros((Nmax, 1))

    count = 0
    while (count < Nmax):
        x1 = f(x0)
        x0 = x1
        x[count] = x1
        count += 1

    return x

f = lambda x: (10/(x+4))**(1/2)
tol = 10**(-10)
Nmax = 10

x0 = 1.5
x = fixedpt(f, x0, tol, Nmax)
    
def func(pn, pn1, pn2):
   p = (pn2*pn - pn1**2)/(pn + pn2 - 2*pn1)
   return p

def subroutine(x, tol, Nmax):
    ret_list = []
    for i in range(Nmax - 2):
        ret_list.append(func(x[i], x[i+1], x[i+2]))

    return ret_list

y = subroutine(x, tol, Nmax)

print('fixed point:')
for i in x:
    print(i)
print('Aitken')
for i in y:
    print(i)