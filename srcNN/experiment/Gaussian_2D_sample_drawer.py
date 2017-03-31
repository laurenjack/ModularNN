import numpy as np
import matplotlib.pyplot as plt

d = 2
n = 70000
x = [np.random.randn(1) for i in xrange(n)]
y = [np.random.randn(1) for i in xrange(n)]
ax = plt.gca()
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
plt.scatter(x, y, color='b')
plt.show()