import numpy as np

reg = 0.0001

class Optimizer:

    def __init__(self, eta, reg=None):
        self.eta = eta
        self.reg = reg

    def decay_learning_rate(self, decay_factor):
        self.eta*=decay_factor

class Sgd(Optimizer):

    def update_weights(self, w, dw, m, a=None):
        return w - (self.eta / m) * dw

    def update_biases(self, b, db, m):
        return b - (self.eta / m) * db

class Sgd_Regularisation(Optimizer):

    def update_weights(self, w, dw, m, a=None):
        return (1-self.eta*reg)*w - (self.eta / m) * dw

    def update_biases(self, b, db, m):
        return b - (self.eta / m) * db

class No_Biases(Sgd):
    """An sgd optimizer designed to avoid ever updating the biases.
    This hack is required because the network class is inherently
    coupled to the biases"""

    def update_biases(self, b, db, m):
        return b




class KeepPositiveRegSgd(Optimizer):

    def update_weights(self, w, dw, batch, a=None):
        w = (1-self.eta*reg)*w
        return positive_gradient_Update(w, dw, batch, self.sta)

    def update_biases(self, b, db, batch):
        return positive_gradient_Update(b, db, batch, self.eta)

class KeepPositiveSgd(Optimizer):

    def update_weights(self, w, dw, batch, a=None):
        return positive_gradient_Update(w, dw, batch,self.eta)

    def update_biases(self, b, db, batch):
        return positive_gradient_Update(b, db, batch, self.eta)

class DualAndSgd(KeepPositiveSgd):
    """Optimzer specifically for an and gate with two inputs,
    which is regularized so that P(On) + P(Off) = 1
    """

    def update_weights(self, w, dw, batch, a):
        return positive_gradient_Update(w, dw, batch,self.eta)

def positive_gradient_Update(matrix, grads, batch, eta):
    matrix = matrix - (eta / len(batch)) * grads
    return np.maximum(matrix, 0)

def create_sgd(eta, reg=False, no_biases=False):
    if no_biases:
        return No_Biases(eta)
    if reg:
        return Sgd_Regularisation(eta, reg)
    return Sgd(eta, reg)

def create_pos_sgd(eta, reg=False):
    if reg:
        return KeepPositiveSgd(eta, reg)
    return KeepPositiveSgd(eta, reg)



