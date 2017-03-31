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

class NoisyOrNegable(Activation):

    def apply(self, z):
        neg = np.exp(-z)
        pos = 1.0 - neg
        return np.concatenate((pos, neg), axis=0)

    def prime(self, a):
        s1 = a.shape[0]/2
        #Make array that allows for 1 operation to compute gradient
        ones = np.ones((s1, 1))
        two_as = 2* a[s1:]
        one_op = np.concatenate((ones, two_as), axis=0)
        return one_op - a

    def weight_grad(self, delta, a):
        #Must join the positive and negative parts of delta first
        s1 = delta.shape[0]/2
        delta = delta.reshape((s1, 2), order='F')
        delta = np.sum(delta, axis=1).reshape(30, 1)
        dw = np.matmul(delta, a.transpose())
        db = delta
        return dw, db



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
        return ez/np.sum(ez, axis=0)

    def prime(self, a):
        return a*(1-a)

class Tanh(Activation):

    def apply(self, z):
        return np.tanh(z)

    def prime(self, a):
        return 1.0 - a**2







