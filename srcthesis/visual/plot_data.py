import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def reshape_data(data_set):
    n = len(data_set)
    X1 = np.zeros(n)
    X2 = np.zeros(n)
    Y = np.zeros(n)
    for i in xrange(n):
        x, y = data_set[i]
        X1[i] = x[0][0]
        X2[i] = x[1][0]
        Y[i] = y[0][0]
    return X1, X2, Y

def reshape_data_2D(data_set):
    n = len(data_set)
    X1 = np.zeros(n)
    Y = np.zeros(n)
    for i in xrange(n):
        x, y = data_set[i]
        X1[i] = x[0][0]
        Y[i] = y[0][0]
    return X1, Y


def plot_data_3D(data_set):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X1, X2, Y = reshape_data(data_set)
    wireframe = ax.scatter(X1, X2, Y)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #plt.show()

def plot_data_2D(data_set, network):
    #Plot the data
    x, y = reshape_data_2D(data_set)
    plt.scatter(x, y)

    #Produce the curve the nn sits on
    x = np.arange(-4, 4, 0.01)
    n = x.shape[0]
    a = np.zeros(n)
    for i in xrange(n):
        xi = x[i]
        x_vec = np.array([[xi]])
        a_vec = network.feedforward(x_vec)
        a[i] = a_vec[0][0]
    plt.plot(x, a, color='r')




