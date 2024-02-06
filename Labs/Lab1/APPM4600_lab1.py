import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 11) # Initializing variables x and y to have the same space and values
y = np.arange(0, 11, 1)

x.size # Verifying size of arrays x and y to be the same
y.size # Removed unused code and fixed structure

x[1:4]

print('The first three entries of x are', x[:3])

w = 10**(-np.linspace(1,10,10)) # Initializing variable w for plotting
x = np.arange(1, w.size+1, 1)

plt.semilogy(x,w) # Fixed formatting and structure to make plotting code more readable
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()