import numpy as np

class Relu:
    def __init__(self, sizes, weights=None, biases=None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes

        if(biases == None):
            self.biases = [np.ones((y, 1)) for y in sizes[1:]]
        else:
            self.biases = biases
        if(weights == None):
            self.weights = [np.random.randn(y, x) * 0.1
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = weights

        self.zeros = [np.zeros((y, 1)) for y in sizes[1:]]
        self.ones = [np.ones((y, 1)) for y in sizes[1:]]

    def feedforward(self, a):
        # Return the output of the network of 'a' is input
        for b, w, zero in zip(self.biases, self.weights, self.zeros):
            a = np.maximum(zero, np.dot(w, a) + b)
        return a

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation using a single mini batch.
        The 'mini_batch' is a list of tuples (x, y) and eta
        is the learning rate"""
        # A list of zeroed out vectors, corresponding to each bias vector
        # and its dimensions
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # A list of zeroed out matricies corresponding to each weight matrix
        # and its dimensions
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        act = x
        activations = [x]  # list to store all the activations, layer by layer
        for b, w, zero in zip(self.biases, self.weights, self.zeros):
            z = np.dot(w, act) + b
            act = np.maximum(zero, z)
            activations.append(act)

        # backward pass
        delta = self.cost_derivative(activations[-1], y, self.zeros[-1]) * \
                self.relu_prime(activations[-1], self.zeros[-1], self.ones[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            rp = self.relu_prime(activations[-l], self.zeros[-l], self.ones[-l])
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * rp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, a, y, zeros):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return a-y

    def relu_prime(self, a, zeros, ones):
        return np.where(a > zeros, ones, zeros)

    def cost(self, coord):
        x, y = coord
        a = self.feedforward(x)
        return 1 / 2.0 * np.linalg.norm(y - a) ** 2