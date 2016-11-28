import matplotlib.pyplot as plt
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.rcParams["font.family"] = "cmr"

def __draw_act(title, f):
    x = np.arange(0, 5, 0.01)
    y = f(x)
    plt.figure(facecolor='white')
    plt.title(title)
    plt.plot(x, y)
    plt.show()

def draw_or():
    x = np.arange(0, 5, 0.01)
    y = 1.0 - np.exp(-x)
    plt.figure(facecolor='white')
    plt.title('Continuous Or Activation Function')
    plt.plot(x, y)
    plt.show()

def draw_and():
    f = lambda x : np.exp(-x)
    __draw_act('Continuous And Activation Function', f)

def plot_ep_to_x():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    E = np.arange(0.01, 1.0, 0.01)
    X = np.arange(0.0, 1.0, 0.01)
    E, X = np.meshgrid(E, X)
    A = E ** X
    # surf = ax.plot_surface(E, X, A, rstride=1, cstride=1, cmap=cm.coolwarm,
    # linewidth=0, antialiased=False)
    wireframe = ax.plot_wireframe(E, X, A, rstride=5, cstride=5)
    ax.set_zlim(0.0, 1.0)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    plt.xlabel(r"$\varepsilon$", fontsize=20)
    plt.ylabel("x")
    zlabel = ax.set_zlabel(r"$\varepsilon^x$", fontsize=20)
    zlabel.set_rotation(180)
    plt.show()

draw_and()