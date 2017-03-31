import numpy as np
from unittest import TestCase
from mock import *
from srcNN.network.network import Network
from gaussian_generator import  GaussianGenerator

class GaussianGeneratorSpec(TestCase):

    def test_update_mini_batch(self):
        """Checks if the batch is normailzed correctly prior to handing it
        off to the network, futhermore, checks that the updated z's are
        correct, given a mock gradient returned by the network"""
        d_phis = [[np.array([1.0, -2.0, 3.0])], [np.array([4.0, 5.0, 6.0])], [np.array([7.0, 8.0, -9.0])]]
        W = [np.array([[1.0, 2.0],
                       [1.0, -1.0],
                       [-4.0, 3.0]])]
        #Create mock network object
        mock_net = create_autospec(Network)
        #Assign the list of weight matrices, the GaussianGenerator should only access the first element
        mock_net.weights = W
        #Mock the gradients returned by the network
        mock_net.update_mini_batch = Mock(return_value=(None, None, d_phis))
        #Create the Gaussian Generator using the mock net
        gg = GaussianGenerator(mock_net)

        #Create the latent variables, just some simple z vectors, the gaussian generator
        #does not touch the x's so the can be anything!
        # x0 = np.array([4.2, 6.1, -0.8, -2.4])
        # x1 = np.array([5.2, 6.3, -0.1, -2.7])
        # x2 = np.array([-7.2, -1.1, 0.85, 2.476])
        mini_batch = [(np.array([1.0, 2.0]), "x0"), (np.array([-1.0, 4.0]), "x1"), (np.array([5.0, -3.0]), "x2")]

        #Make the call to update mini batch
        result = gg.update_mini_batch(mini_batch, 0.5)

        #First check to see if the network was updated with the correctly normalized z's
        normed_batch = [(np.array([-2.0/3.0, 1.0]), "x0"), (np.array([-8.0/3.0, 3.0]), "x1"), (np.array([10.0/3.0, -4.0]), "x2")]
        called_batch = mock_net.update_mini_batch.call_args[0][0]
        self.assert_batches_equal(normed_batch, called_batch)

        #Second, check the batch was updated properly
        updated_batch = [(np.array([5+1.0/3.0, -2-1.0/3.0]), "x0"), (np.array([4.0, -3.0]), "x1"), (np.array([-12.0, 4.0]), "x2")]
        self.assert_batches_equal(updated_batch, result)

    def assert_batches_equal(self, expected, actual):
        for exp, act in zip(expected, actual):
            np.testing.assert_allclose(exp[0], act[0])
            self.assertEqual(exp[1], act[1])







