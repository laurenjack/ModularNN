import numpy as np
from unittest import TestCase
import network as n
import network_factory as nf
from activations import *
from optimizers import *

class NetworkSpec(TestCase):

    def test_relu_with_final_linear_feedforward(self):
        network, x, y = self.__setup_test_example()

        r0 = network.feedforward(x[0])
        r1 = network.feedforward(x[1])
        r2 = network.feedforward(x[2])

        e0 = np.array([[19.0]])
        e1 = np.array([[0.0]])
        e2 = np.array([[24.0]])
        self.assertTrue(np.allclose(e0, r0))
        self.assertTrue(np.allclose(e1, r1))
        self.assertTrue(np.allclose(e2, r2))

    def test_feed_forward_sigmoid_then_softmax(self):
        sgd = Sgd(0.5)
        activations = [Sigmoid(sgd), Softmax(sgd)]
        w = [np.array([[1,2,3], [-4,-5,-6]]), np.array([[-1, 2], [2, -2], [3,3]])]
        b = [0.5*np.ones((2,1)), np.ones((3, 1))]
        network = n.Network(w, b, activations)

        x = np.array([[2], [4], [6]])
        a = network.feedforward(x)

        exp = np.dot(w[0], x)
        exp = 1.0/(1.0 + np.exp(-exp))
        exp = np.dot(w[1], exp)
        exp = np.exp(exp)/np.sum(np.exp(exp))
        self.assertTrue(np.allclose(exp, a))

    def test_relu_with_final_linear_backprop(self):
        network, x, y = self.__setup_test_example()

        r0 = network.backprop(x[0], y[0])
        r1 = network.backprop(x[1], y[1])
        r2 = network.backprop(x[2], y[2])


        #e1 = np.array([[0.0]])
        #e2 = np.array([[24.0]])

        #First coord
        eb = [16 * np.array([[4], [3]]), np.array([[16]])]
        ew = [16 * np.array([[-4, 8], [-3, 6]]), 16 * np.array([1, 5])]
        self.assertTrue(np.allclose(eb[0], r0[0][0]))
        self.assertTrue(np.allclose(eb[1], r0[0][1]))
        self.assertTrue(np.allclose(ew[0], r0[1][0]))
        self.assertTrue(np.allclose(ew[1], r0[1][1]))

        # Second coord
        eb = [np.array([[0], [0]]), np.array([[-2]])]
        ew = [np.array([[0, 0], [0, 0]]), np.array([0, 0])]
        self.assertTrue(np.allclose(eb[0], r1[0][0]))
        self.assertTrue(np.allclose(eb[1], r1[0][1]))
        self.assertTrue(np.allclose(ew[0], r1[1][0]))
        self.assertTrue(np.allclose(ew[1], r1[1][1]))

        # Third coord
        eb = [23*np.array([[4], [0]]), np.array([[23]])]
        ew = [23*np.array([[8, 4], [0, 0]]), 23*np.array([6, 0])]
        self.assertTrue(np.allclose(eb[0], r2[0][0]))
        self.assertTrue(np.allclose(eb[1], r2[0][1]))
        self.assertTrue(np.allclose(ew[0], r2[1][0]))
        self.assertTrue(np.allclose(ew[1], r2[1][1]))


    def __setup_test_example(self):
        weights = [np.array([[2, 1], [-2, 2]]), np.array([[4, 3]])]
        biases = [np.array([[1], [-1]]), np.array([0])]
        network = nf.relu_with_linear_final([2,2,1], 0.5, weights, biases)
        x = [np.array([[-1], [2]]), np.array([[1], [-3]]), np.array([[2], [1]])]
        y = [np.array([[3]]), np.array([[2]]), np.array([[1]])]
        return network, x, y

