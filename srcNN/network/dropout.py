import numpy as np
import random

class DropOut:

    def __init__(self, sizes, n):
        self.keep_prob = 1.0/float(n)
        self.sizes = sizes
        self.drop_mask = None

    def half_weights(self, weights):
        weights[-1] *= self.keep_prob

    def double_weights(self, weights):
        weights[-1] /= self.keep_prob

    def new_batch(self):
        node_probs = [np.random.uniform(0, 1, (s, 1)) for s in self.sizes[1:-1]]
        node_mask = [np.where(node_p > 1 - self.keep_prob, 1, 0) for node_p in node_probs]
        node_mask = [np.ones((self.sizes[0], 1))] + node_mask
        node_mask.append(np.ones((self.sizes[-1], 1)))
        self.drop_mask = [nm1.dot(nm0.transpose()) for nm0, nm1 in zip(node_mask[:-1], node_mask[1:])]

    def drop_grads(self, nabla_w):
        return [dm*nw for dm, nw in zip(self.drop_mask, nabla_w)]

    def drop_weights(self, weights):
        return [dm*(w) for dm, w in zip(self.drop_mask, weights)]


class DropConnect:
    """Encapsulates all the actions required to carry out dropConnect"""
    def __init__(self, sizes, n):
        self.keep_prob = 1.0/float(n)
        self.zeros = [np.zeros((s1, s0)) for s0, s1 in zip(sizes[:-1], sizes[1:])]
        self.drop_mask = None

    def half_weights(self, weights):
        weights[-1] *= self.keep_prob

    def double_weights(self, weights):
        weights[-1] /= self.keep_prob

    def new_batch(self):
        self.drop_mask = [np.random.uniform(0, 1, z.shape) for z in self.zeros]

    def drop_grads(self, nabla_w):
        return [np.where(dm > 1 - self.keep_prob, dw, z) for dm, dw, z in zip(self.drop_mask, nabla_w, self.zeros)]

    def drop_weights(self, weights):
        return [np.where(dm > 1 - self.keep_prob, w, z) for dm, w, z in zip(self.drop_mask, weights, self.zeros)]

class DropSys:

    def __init__(self, sizes, n):
        #self.zeros = [np.zeros((s1, s0)) for s0, s1 in zip(sizes[:-1], sizes[1:])]
        self.sizes = sizes
        self.n = n
        self.drop_mask = None

    def half_weights(self, weights):
        weights[-1] /= self.n

    def double_weights(self, weights):
        weights[-1] *= self.n

    def new_batch(self):
        begins = [random.randint(0, self.n - 1) for i in xrange(len(self.sizes) - 2)]
        self.drop_mask = [self.__random_sect(s1, s0, b1, b0)
                          for s1, b1, s0, b0 in zip(self.sizes[2:-1], begins[1:],  self.sizes[1:-2], begins[:-1])]
        first_layer = self.__first_rand_sect(self.sizes[1], self.sizes[0], begins[0])
        last_layer = self.__last_rand_sect(self.sizes[-1], self.sizes[-2], begins[-1])
        self.drop_mask = [first_layer] + self.drop_mask
        self.drop_mask.append(last_layer)


    def drop_grads(self, nabla_w):
        return [np.where(dm > 0.5, dw, dm) for dm, dw in zip(self.drop_mask, nabla_w)]

    def drop_weights(self, weights):
        return [np.where(dm > 0.5, w, dm) for dm, w in zip(self.drop_mask, weights)]

    def __random_sect(self, s1, s0, b1, b0):
        sect = np.zeros((s1, s0))
        width1 = s1/self.n
        width0 = s0/self.n
        begin1 =  b1 * width1
        begin0 = b0 * width0
        sect[begin1:begin1+width1, begin0:begin0+width0] = np.ones((width1, width0))
        return sect

    def __first_rand_sect(self, s1, s0, b1):
        sect = np.zeros((s1, s0))
        width1 = s1 / self.n
        begin1 = b1 * width1
        sect[begin1:begin1+width1] = np.ones((width1, s0))
        return sect;

    def __last_rand_sect(self, s1, s0, b0):
        sect = np.zeros((s1, s0))
        width0 = s0 / self.n
        begin0 = b0 * width0
        sect[:, begin0:begin0+width0] = np.ones((s1, width0))
        return sect;

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

