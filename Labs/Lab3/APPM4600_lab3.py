import numpy as np
import math

# bisection function from class
def bisection(f,a,b,tol,Nmax):
    '''
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    '''

    '''     first verify there is a root we can find in the interval '''
    fa = f(a); fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

    ''' verify end point is not a root '''
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    while (count < Nmax):
      c = 0.5*(a+b)
      fc = f(c)

      if (fc ==0):
        astar = c
        ier = 0
        return [astar, ier]

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar, ier]

      if (abs(b-a)<tol):
        astar = a
        ier =0
        return [astar, ier]
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier] 

# use routines    
# f = lambda x: x**3+x-4
# a = 1
# b = 4

# Nmax = 100
# tol = 1e-3

# [astar,ier] = bisection(f,a,b,tol,Nmax)
# print('the approximate root is',astar)
# print('the error message reads:',ier)

# Exercise 1
f_1 = lambda x: (x**2)*(x-1)
Nmax_1 = 100
tol_1 = 1e-3

# Exercise 1a
a_1a = 0.5
b_1a = 2
[approxroot_1a, error_mes_1a] = bisection(f_1, a_1a, b_1a, tol_1, Nmax_1)
print('the approximate root of 1a is', approxroot_1a)
print('the error message of 1a reads:', error_mes_1a)

# Exercise 1b
a_1b = -1
b_1b = 0.5
[approxroot_1b, error_mes_1b] = bisection(f_1, a_1b, b_1b, tol_1, Nmax_1)
print('the approximate root of 1b is', approxroot_1b)
print('the error message of 1b reads:', error_mes_1b)

# Exercise 1c
a_1c = -1
b_1c = 2
[approxroot_1c, error_mes_1c] = bisection(f_1, a_1c, b_1c, tol_1, Nmax_1)
print('the approximate root of 1c is', approxroot_1c)
print('the error message of 1c reads:', error_mes_1c)

# Exercise 2
Nmax_2 = 100
tol_2 = 10**-5

# Exercise 2a
f_2a = lambda x: (x-1)*(x-3)*(x-5)
a_2a = 0
b_2a = 2.4
[approxroot_2a, error_mes_2a] = bisection(f_2a, a_2a, b_2a, tol_2, Nmax_2)
print('the approximate root of 2a is', approxroot_2a)
print('the error message of 2a reads:', error_mes_2a)

# Exercise 2b
f_2b = lambda x: ((x-1)**2)*(x-3)
a_2b = 0
b_2b = 2
[approxroot_2b, error_mes_2b] = bisection(f_2b, a_2b, b_2b, tol_2, Nmax_2)
print('the approximate root of 2b is', approxroot_2b)
print('the error message of 2b reads:', error_mes_2b)

# Exercise 2c
f_2c = lambda x: math.sin(x)
a_2c = 0
b_2c = 0.1
[approxroot_2c, error_mes_2c] = bisection(f_2c, a_2c, b_2c, tol_2, Nmax_2)
print('the approximate root of 2c ( (a,b) = (0, 0.1) ) is', approxroot_2c)
print('the error message of 2c ( (a,b) = (0, 0.1) ) reads:', error_mes_2c)

a_2c = 0.5
b_2c = (3/4) * math.pi
[approxroot_2c, error_mes_2c] = bisection(f_2c, a_2c, b_2c, tol_2, Nmax_2)
print('the approximate root of 2c ( (a,b) = (0.5, 3pi/4) ) is', approxroot_2c)
print('the error message of 2c ( (a,b) = (0.5, 3pi/4) ) reads:', error_mes_2c)

# fixed point function from class
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

# use routines 
# f1 = lambda x: 1+0.5*np.sin(x)
# ''' 
# fixed point is alpha1 = 1.4987....
# '''

# f2 = lambda x: 3+2*np.sin(x)
# ''' 
# fixed point is alpha2 = 3.09... 
# '''

# Nmax = 100
# tol = 1e-6

# ''' test f1 '''
# x0 = 0.0
# [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
# print('the approximate fixed point is:',xstar)
# print('f1(xstar):',f1(xstar))
# print('Error message reads:',ier)
    
# ''' test f2 '''
# x0 = 0.0
# [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
# print('the approximate fixed point is:',xstar)
# print('f2(xstar):',f2(xstar))
# print('Error message reads:',ier)

# Exercise 3
Nmax = 100
tol = 1e-6

# Exercise 3a
f_3a = lambda x: x*((1+((7-x**5)/(x**2)))**3)
x0 = 7**(1/5)
[xstar_3a, ier_3a] = fixedpt(f_3a, x0, tol, Nmax)