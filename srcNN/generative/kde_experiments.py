import numpy as np
import math
import matplotlib.pyplot as plt


n = 50

h = (4.0/(3.0 * n)) ** 0.2

#h = 0.4216846
#h = 0.85028
K = 1.0 / math.sqrt(2.0 * math.pi)

def p(z):
    """Isotropic Gaussian"""
    p_exp = -0.5 * z ** 2.0
    return K * np.exp(p_exp)

def f(z, zs):
    m = len(zs)
    return K/float(m)/h * np.sum([np.exp(-0.5 * ((z - zi)/h) ** 2.0) for zi in zs])

#Generate n z's
zs = [np.random.normal() for i in xrange(n)]
#Calculate densities for each function
gauss = [p(z) for z in zs]
kde = [f(z, zs) for z in zs]

#Plot both sets of densities
plt.scatter(zs, gauss, color='r')
plt.scatter(zs, kde, color='b')
plt.show()

