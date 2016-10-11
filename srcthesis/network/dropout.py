import numpy as np

class DropOut:
    """Encapsulates all the actions required to carry out dropout"""
    def __init__(self, sizes):
        self.zeros = [np.zeros((s1, s0)) for s0, s1 in zip(sizes[:-1], sizes[1:])]
        self.drop_mask = None

    def half_weights(self, weights):
        weights[-1] *= 0.5

    def double_weights(self, weights):
        weights[-1] *= 2.0

    def new_batch(self):
        self.drop_mask = [np.random.uniform(0, 1, z.shape) for z in self.zeros]

    def drop_grads(self, nabla_w):
        return [np.where(dm > 0.5, dw, z) for dm, dw, z in zip(self.drop_mask, nabla_w, self.zeros)]

    def drop_weights(self, weights):
        return [np.where(dm > 0.5, w, z) for dm, w, z in zip(self.drop_mask, weights, self.zeros)]

class DropNull:
    """Class for network instances that do not need drop-out, this
    is a null-object."""
    def half_weights(self, weights):
        pass

    def double_weights(self, weights):
        pass

    def new_batch(self):
        pass

    def drop_grads(self, nabla_w):
        return nabla_w

    def drop_weights(self, weights):
        return weights