import numpy as np

class SigAND:
    """A sigmoid function with a noisy OR gate at the output layer"""
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #initialize the sigmoid weights and biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:-1]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-2], sizes[1:-1])]
        #initialize the OR weights and biases
        self.and_weights = positive_normal((sizes[-1], sizes[-2]))
        self.and_biases = positive_normal((sizes[-1], 1))
        self.zero_weights = np.zeros(self.and_weights.shape)
        self.zero_biases = np.zeros(self.and_biases.shape)


    def feedforward(self, a):
        # Return the output of the network of 'a' is input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        w, b = self.and_weights, self.and_biases
        a = cont_and(np.dot(w, a - np.ones(a.shape)) - b)
        return a

    def update_mini_batch(self, mini_batch, eta_sched):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation using a single mini batch.
        The 'mini_batch' is a list of tuples (x, y) and eta
        is the learning rate"""
        sig_eta = eta_sched[0]
        and_eta = eta_sched[1]
        # A list of zeroed out vectors, corresponding to each bias vector
        # and its dimensions
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # A list of zeroed out matricies corresponding to each weight matrix
        # and its dimensions
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_or_b = np.zeros(self.and_biases.shape)
        nabla_or_w = np.zeros(self.and_weights.shape)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, db_or, dw_or = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_or_b += db_or
            nabla_or_w += dw_or

        self.weights = [w - (sig_eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (sig_eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.and_weights = self.positive_gradient_Update(self.and_weights, nabla_or_w, self.zero_weights, and_eta, mini_batch)
        self.and_biases = self.positive_gradient_Update(self.and_biases, nabla_or_b, self.zero_biases, and_eta, mini_batch)

    def positive_gradient_Update(self, matrix, gradients, zeros, eta, mini_batch):
        matrix = matrix - (eta / len(mini_batch)) * gradients
        matrix = np.maximum(matrix, zeros)
        return matrix

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        w, b = self.and_weights, self.and_biases
        z = np.dot(w, activation - np.ones(activation.shape)) - b
        zs.append(z)
        activation = cont_and(z)
        activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                cont_and_prime(zs[-1])
        db_or = -delta
        a2 = activations[-2]
        a2Neg = a2 - np.ones(a2.shape)
        dw_or = np.dot(delta, a2Neg.transpose())

        #backward pass to first sigmoid layer
        sp = sigmoid_prime(zs[-2])
        delta = np.dot(self.and_weights.transpose(), delta) * sp
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-3].transpose())
        for l in xrange(3, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 2].transpose(), delta) * sp
            nabla_b[-l+1] = delta
            nabla_w[-l+1] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w, db_or, dw_or

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y) #/(np.ones(y.shape) - output_activations)

    def cost(self, a, y):
        return 1/2.0 * np.linalg.norm(y - a)**2

def sigmoid(z):
    #The sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    #The derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

def cont_and(z):
    return np.exp(z)

def cont_and_prime(z):
    return np.exp(z)

def positive_normal(shape):
    matrix = np.random.randn(shape[0], shape[1])
    for e in np.nditer(matrix, op_flags=['readwrite']):
        e[...] = abs(e)
    return matrix