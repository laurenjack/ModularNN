import numpy as np

reg = 0.0001

class Optimizer:

    def __init__(self, eta, reg=None):
        self.eta = eta
        self.reg = reg

    def decay_learning_rate(self, decay_factor):
        self.eta*=decay_factor

class Sgd(Optimizer):

    def update_weights(self, w, dw, mini_batch):
        return w - (self.eta / len(mini_batch)) * dw

    def update_biases(self, b, db, mini_batch):
        return b - (self.eta / len(mini_batch)) * db

class Sgd_Regularisation(Optimizer):

    def update_weights(self, w, dw, mini_batch):
        return (1-self.eta*reg)*w - (self.eta / len(mini_batch)) * dw

    def update_biases(self, b, db, mini_batch):
        return b - (self.eta / len(mini_batch)) * db

class Saxe_Hack(Sgd):
    """An sgd optimizer designed to avoid ever updating the biases.
    This hack is required because the network class is inherently
    coupled to the biases"""

    def update_biases(self, b, db, mini_batch):
        return b




class KeepPositiveRegSgd(Optimizer):

    def update_weights(self, w, dw, batch):
        w = (1-self.eta*reg)*w
        return positive_gradient_Update(w, dw, batch, self.sta)

    def update_biases(self, b, db, batch):
        return positive_gradient_Update(b, db, batch, self.eta)

class KeepPositiveSgd(Optimizer):

    def update_weights(self, w, dw, batch):
        return positive_gradient_Update(w, dw, batch,self.eta)

    def update_biases(self, b, db, batch):
        return positive_gradient_Update(b, db, batch, self.eta)

def positive_gradient_Update(matrix, grads, batch, eta):
    matrix = matrix - (eta / len(batch)) * grads
    return np.maximum(matrix, 0)

def create_sgd(eta, reg=False, saxe_hack=False):
    if saxe_hack:
        return Saxe_Hack(eta)
    if reg:
        return Sgd_Regularisation(eta, reg)
    return Sgd(eta, reg)

def create_pos_sgd(eta, reg=False):
    if reg:
        return KeepPositiveSgd(eta, reg)
    return KeepPositiveSgd(eta, reg)



