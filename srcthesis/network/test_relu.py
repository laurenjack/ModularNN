from unittest import TestCase
import numpy as np
from srcthesis.network.relu import Relu

class ReluSpec(TestCase):

    def test_single_layer_relu_problem(self):
        weights = [np.array([[1.0]])]
        biases = [np.array([[0.0]])]
        sizes = [1,1]
        relu = Relu(sizes, weights, biases)

        # Below the y axis, no activation, zero gradient
        self.__assert_backprop(relu, -1.0, 0.0, 0, 0)

        #Above the actual function, postive gradients i.e. cost is increasing
        self.__assert_backprop(relu, 1.5, 1.0, 0.5, 0.75)

        #Below the actual function

    def __assert_backprop(self, relu, x, y, expDw, expDb):
        xArr = np.array([[x]])
        yArr = np.array([[y]])

        result = relu.backprop(xArr, yArr)
        dw = result[0][0][0, 0]
        db = result[1][0][0, 0]
        self.assertAlmostEqual(dw, expDw)
        self.assertAlmostEqual(db, expDb)