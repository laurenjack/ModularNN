# from unittest import TestCase
# from srcNN.network import noisy_and_neg as na
# from srcNN.testUtils import test_utils as tu
# import numpy as np
#
# class NoisyAndSpec(TestCase):
#
#     def test_feedforward(self):
#         network = na.NoisyAnd([2, 1])
#         network.weights[0][0,0] = -0.1
#         network.weights[0][0,1] = -0.3
#         network.biases[0][0,0] = -0.7
#         x = tu.numpy_array_vector([1, 0.5])
#
#         result = network.feedforward(x)
#
#         self.assertAlmostEqual(0.4274, result[0,0], places=4)
#
#     def test_gradient(self):
#         network = na.NoisyAnd([2, 1])
#         network.weights[0][0, 0] = -0.1
#         network.weights[0][0, 1] = -0.3
#         network.biases[0][0, 0] = -0.7
#         x = tu.numpy_array_vector([1, 0.5])
#         y = tu.numpy_array_vector([1])
#         out1 = network.feedforward(x)
#         cost1 = 1/2.0*(y[0,0] - out1[0,0])**2
#
#         #compute gradient
#         dbList, dwList = network.backprop(x, y)
#         db = dbList[0]
#         dw = dwList[0]
#
#         # Update weights
#         network.weights[0] = network.weights[0] + 0.000001
#         network.biases[0] = network.biases[0] + 0.000001
#
#         out2 = network.feedforward(x)
#         cost2 = 1/2.0*(y[0,0] - out2[0,0])**2
#         print(cost2 - cost1)
#         print(np.dot(dw, 0.000001) + np.dot(db, 0.000001))
#
#         taylor = cost1 + np.dot(dw, 0.000001) + np.dot(db, 0.000001)
#         self.assertAlmostEqual(taylor, cost2, places=10)