import numpy as np

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


class KeepPositiveSgd(Optimizer):

    def update_weights(self, w, dw, batch):
        return self.__positive_gradient_Update(w, dw, batch)

    def update_biases(self, b, db, batch):
        return self.__positive_gradient_Update(b, db, batch)

    def __positive_gradient_Update(self, matrix, grads, batch):
        matrix = matrix - (self.eta / len(batch)) * grads
        return np.maximum(matrix, 0)
