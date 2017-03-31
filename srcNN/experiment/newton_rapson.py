import numpy as np

def f(z):
    h = (4.0 / 9.0) ** 0.2
    return np.exp(-0.5 * z ** 2.0) - (1 + np.exp(-0.5 * (z/h) ** 2.0) + np.exp(-(z/h) ** 2.0))/3.0

def df(z):
    h = (4.0 / 9.0) ** 0.2
    return -z*np.exp(-0.5 * z ** 2.0) + (z * np.exp(-0.5 * (z / h) ** 2.0) + 2*z*np.exp(-(z / h) ** 2.0)) / 3.0 /h ** 2.0

z = 2.0
for i in xrange(10):
    fz = f(z)
    dfz = df(z)
    z = z - fz/dfz
print z