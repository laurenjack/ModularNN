import numpy as np
from optimizers import *

class Activation:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def weighted_sum(self, w, b, a):
        return np.matmul(w, a) + b

    def weight_grad(self, delta, a):
        dw = np.matmul(delta, a.transpose())
        db = delta
        return dw, db

    def opt(self):
        return self.optimizer

class Linear(Activation):

    def __init__(self, optimizer):
        Activation.__init__(self, optimizer)

    def apply(self, z):
        return z

    def prime(self, a):
        return 1

class Relu(Activation):

    def __init__(self, width, optimizer):
        """Initialises with a vector of zeros of the layer width corresponding
        to this activation. These are used to compute the gradient/ feedforward
        appropriately"""
        Activation.__init__(self, optimizer)
        self.zeros = np.zeros((width, 1))
        self.ones = np.ones((width, 1))

    def apply(self, z):
        return np.maximum(self.zeros, z)

    def prime(self, a):
        return np.where(a > self.zeros, self.ones, self.zeros)

class NoisyOr(Activation):

    def apply(self, z):
        return 1.0 - np.exp(-z)

    def prime(self, a):
        return 1.0 - a

    def optimzer(self):
        return self.optimizer

class NoisyAnd(Activation):

    def apply(self, z):
        return np.exp(z)

    def prime(self, a):
        return a

    def weighted_sum(self, w, b, a):
        return np.dot(w, a - np.ones(a.shape)) - b

    def weight_grad(self, delta, a):
        a_neg = a - 1.0
        dw = np.matmul(delta, a_neg.transpose())
        db = -delta
        return dw, db


class Sigmoid(Activation):

    def apply(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, a):
        return a*(1.0-a)

class Softmax(Activation):

    def apply(self, z):
        ez = np.exp(z)
        return ez/np.sum(ez)

    def prime(self, a):
        return a*(1-a)





