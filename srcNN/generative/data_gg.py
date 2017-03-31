import numpy as np
import sklearn.datasets as sk
import srcNN.data.mnist_loader as mnist
import random

"""Module responsible for loading data for a Gaussian Generator"""


def __normalise(X):
    """Return a dataset normalized between zero and 1 and the
    max and min elements used to do so"""
    min = np.amin(X)
    max = np.amax(X)
    X_norm = (X - min)/(max - min)
    return X_norm, min, max

def __remove_unwanted_digits(just_get, X_mat, targets, is_mnist=False):
    if just_get == None:
        return X_mat, targets
    ind_to_delete = []
    for i in xrange(targets.shape[0]):
        t = targets[i]
        if not t in just_get or (is_mnist and random.random() > 0.1):
            ind_to_delete.append(i)
    X_mat = np.delete(X_mat, ind_to_delete, axis=0)
    targets = np.delete(targets, ind_to_delete)
    return X_mat, targets

def load_sk_digits(just_get=None):
    """Load the sk-learn digits data set"""
    digits = sk.load_digits()
    X_mat = digits.data
    targets = digits.target

    #Remove unwanted digits, (only if just_get is specified)
    X_mat, targets = __remove_unwanted_digits(just_get, X_mat, targets)
    n, d = X_mat.shape

    #Normalise pixels between zero and 1
    X_norm, _, _ = __normalise(X_mat)

    #Express digits as a list of vectors
    xs = [X_norm[i].reshape(d, 1) for i in xrange(n)]

    return xs, targets

def load_mnist(just_get=None):
    """Load the sk-learn digits data set"""
    train_data, _, _ = mnist.load_data()
    X_mat, targets = train_data

    #Remove unwanted digits, (only if just_get is specified)
    X_mat, targets = __remove_unwanted_digits(just_get, X_mat, targets, is_mnist=True)
    n, d = X_mat.shape

    #Express digits as a list of vectors
    xs = [X_mat[i].reshape(d, 1) for i in xrange(n)]

    return xs, targets



#load_mnist([1,3])