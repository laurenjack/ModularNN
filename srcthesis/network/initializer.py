import numpy as np

"""
Contains methods that are responsible for initializing the weights of a neural network
"""

def min_init(first_weight, train):
    """Args:
          first_weight: The initial value for the first weight, the only weight that will
          learn initially. This paramater is only used for testing purposes, in production
          the first weight should be random
          train (ndarray, ndarray): An (x, y) tuple where x is the matrix of covariates and
          y the matrix of response variables.
          sizes [int]: A list of integers specifying the size of each layer of the network

      Returns:
          ([ndarray], [ndarray]): A tuple (weights, biases) which contains the weights and
          biases for the new network
    """
    X = train[0]
    Y = train[1]
    x0 = X[:,0:1]
    covariatesLessCol0 = X[:,1:]
    n = Y.shape[0]
    X1 = np.append(covariatesLessCol0, np.ones((n, 1)), 1)

    XtX = np.dot(X1.transpose(), X1)
    XtXinv = np.linalg.inv(XtX)
    XtY = np.dot(X1.transpose(), Y)
    w0_vector = first_weight * np.dot(X1.transpose(), x0)

    optimal_params = np.dot(XtXinv, XtY - w0_vector)

    first_weight_mat = np.array([[first_weight]])
    other_weights = optimal_params[:-1]
    weights = np.append(first_weight_mat, other_weights, 0).transpose()
    bias = optimal_params[-1:]

    return [weights], [bias]