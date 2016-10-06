"""
network.py

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that i have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits manu desirable features.
"""

#### Libraries
# Standard libraries
import random

#Third party libraries
import numpy as np

class Network(object):
    def __init__(self, sizes):
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
        """ A list of vectors, each of height y (2 vectors for the two
        non-input layers"""
        self.biases = [np.random.randn(y,1) for y in sizes [1:]]
        """The second line of the expression below pairs the first layer
        of neurons outputs with the second layers inputs (i.e.) (2,3)
        and pair the second layers outputs with the third layers inputs
        i.e. (3,1). So there is a tuple for each layer of connections.Each
        tuple is used to construct a weights matrix for that layer of
        connections. The first is a 3x2 matrix, the second is a 1x3 matrix.
        The matrix is aligned such that the input layer is the number of rows
        and the output layer is the number of columns, so that the weights
        are correctly align with the biases and can therefore be applied to the
        activation function"""
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        #Return the output of the network of 'a' is input
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        if test_data: n_test = len(test_data)

        if (test_data):
            print "Before training: {0} / {1}".format(self.evaluate(test_data), n_test)

        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            #Make a list of lists, i.e. a lsit of distinct training data subsets
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                #run backpropagation on the neural net using the current batch
                #of training data and the learning rate eta
                self.update_mini_batch(mini_batch, eta)
            if(test_data):
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete.".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation using a single mini batch.
        The 'mini_batch' is a list of tuples (x, y) and eta
        is the learning rate"""
        #A list of zeroed out vectors, corresponding to each bias vector
        #and its dimensions
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #A list of zeroed out matricies corresponding to each weight matrix
        #and its dimensions
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b- (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]               

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #list to store all the activations, layer by layer
        zs = [] #list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the class the activation is closest to."""
        test_results = [(self.feedforward(x), y)
                        for (x, y) in test_data]
        return sum(int(abs(y2 - x2) < 0.5) for (x2, y2) in test_results)


    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
#### Miscellaneous Functions
def sigmoid(z):
    #The sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    #The derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))
