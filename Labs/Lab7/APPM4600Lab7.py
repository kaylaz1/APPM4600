import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():


    f = lambda x: np.sinc(5*x)

    N = 12
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    h = 2/(N - 1)
    xint = np.array([-1 + (j-1) * h for j in range(N+1)])

    ''' create interpolation data'''
    yint = f(xint)

    ''' create monomial interpolation '''
    result = []
    for i in xint:
        curr = []
        for j in range(N+1):
            curr.append(i**j)
        result.append(curr)
    vinv = np.linalg.inv(result)
    coeffs = []
    for i in vinv:
        coeffs.append(np.dot(i, yint))
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)

    mn = []
    for i in xeval:
        ans = 0
        for j in range(N+1):
            ans += coeffs[j] * i**j
        mn.append(ans)
    yeval_mn = np.array(mn)

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
    plt.plot(xeval,fex,'ro-', label='function')
    plt.plot(xeval,yeval_l,'bs--', label='lagrange') 
    plt.plot(xeval,yeval_dd,'c.--', label='divided differences')
    plt.plot(xeval,yeval_mn,'g--', label='monomial expansion')
    plt.title('Approximations')
    plt.legend()

    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_mn = abs(yeval_mn-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.semilogy(xeval,err_mn,'g--',label='monomial expansion')
    plt.title('Absolute Error')
    plt.legend()
    plt.show()


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
