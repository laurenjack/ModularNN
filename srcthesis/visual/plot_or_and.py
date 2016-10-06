#from mpl_toolkits.mplot3d import Axes3D
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
E = np.arange(0.01, 1.0, 0.01)
X = np.arange(0.0, 1.0, 0.01)
E, X = np.meshgrid(E, X)
A = E**X
#surf = ax.plot_surface(E, X, A, rstride=1, cstride=1, cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
wireframe = ax.plot_wireframe(E, X, A, rstride=5, cstride=5)
ax.set_zlim(0.0, 1.0)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.xlabel(r"$\varepsilon$", fontsize=20)
plt.ylabel("x")
zlabel = ax.set_zlabel(r"$\varepsilon^x$", fontsize=20)
zlabel.set_rotation(180)
plt.show()