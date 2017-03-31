import numpy as np

"""File used to track implement cost functions"""

class Quadratic:

    def prime(self, y, a):
        """The derivative of the cost funtion with repsect
        to the outputs of the network, dC/daL"""
        return y - a

    def apply(self, y, a):
        """Evaluate the cost of the network outputs a relative to the targets y"""
        return 1 / 2.0 * np.linalg.norm(y - a) ** 2

class CrossEntropy:

    def prime(self, y, a):
        """The derivative of the cost funtion with repsect
        to the outputs of the network, dC/daL"""
        return y - a/(a*(1-a))

    def apply(self, y, a):
        """Evaluate the cost of the network outputs a relative to the targets y"""
        return np.sum(y*np.log(a) + (1 - y)*(np.log(1 - a)))