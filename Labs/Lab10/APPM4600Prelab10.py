def eval_legendre(n, x):
    p = [1, x]
    
    if n == 0:
        return [1]
    elif n == 1:
        return p

    for i in range(1, n):
        p.append(1/(n+1) * ((2*n+1)*i*p[i] - n*p[i-1]))
    return p

print(eval_legendre(4, 2))
from scipy.special import legendre
print(legendre(4, 2))
from scipy.special import eval_legendre
print(eval_legendre(4, 2))