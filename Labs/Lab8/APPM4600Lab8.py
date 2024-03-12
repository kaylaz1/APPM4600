# linear spline
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 10;
    ''' interval'''
    a = 0;
    b = 1;

    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1);

    ''' create interpolation data'''
    yint = f(xint);

    Neval = 1000;
    xeval = np.linspace(a,b,Neval+1);

    ''' Linear spline evaluation '''
    yeval_ls = eval_lin_spline(xeval,xint,yint,N);

    ''' create points for evaluating the Lagrange interpolating polynomial'''
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)

    '''Initialize and populate the first columns of the
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )

    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)

    ''' create vector with exact values'''
    fex = f(xeval)

    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--')
    plt.plot(xeval,yeval_dd,'c.--')
    plt.plot(xeval,yeval_ls,'g--')
    plt.legend()

    plt.figure()
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_ls = abs(yeval_ls-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.semilogy(xeval,err_ls,'g--',label='lin spline')
    plt.legend()
    plt.show()

def eval_line(x,x0,y0,x1,y1):
    lin = (1/(x1-x0))*(y0*(x1-x) + y1*(x-x0));
    return lin;

def find_int(xeval,a,b):
    ind = np.where(np.logical_and(xeval>=a,xeval<=b));
    return ind;

def eval_lin_spline(xeval,xint,yint,N):
    Neval = len(xeval);
    yeval = np.zeros(Neval);

    for n in range(N):
        indn = find_int(xeval,xint[n],xint[n+1]);
        yeval[indn] = eval_line(xeval[indn],xint[n],yint[n],xint[n+1],yint[n+1]);

    return yeval;

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)

    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.

    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]

    return(yeval)


''' create divided difference matrix'''
def dividedDiffTable(x, y, n):

    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;

def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)

    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])

    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]

    return yeval



driver()


# cubic spline
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():
    
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1
    
    
    ''' number of intervals'''
    Nint = 8
    xint = np.linspace(a,b,Nint+1)
    yint = f(xint)

    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)

    
    
    (M,C,D) = create_natural_spline(yint,xint,Nint)
    
    print('M =', M)
    print('C =', C)
    print('D=', D)
    
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D)
    
#    print('yeval = ', yeval)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='natural spline') 
    plt.legend
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    for i in range(1,N):
        A[i,i]= 4
        A[i,i-1] = 2*h[i]/(h[i]+h[i+1])
        A[i,i+1] = 2*h[i+1]/(h[i]+h[i+1])
    
    A = A/12

    A[0,0] = 1
    A[N, N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b) 

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi

    yeval = 1/hi*(-Mi*(xeval-xip)**3/6+Mip*(xeval-xi)**3/6) -C*(xeval-xip)+D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()       