import numpy as np
from unittest import TestCase
import mock
from srcthesis.network.relu import Relu
from srcthesis.visual.weight_tracker import WeightStats

class WeightStatsSpec(TestCase):

    def test_when_update_three_times_then_returned_weights_correct(self):
        mock_net = mock.create_autospec(Relu)
        mock_net.cost = mock.Mock(side_effect=cost_mock)
        mock_train = [[10, 20], [30, 40]]
        weight_stats = WeightStats(mock_net, mock_train)

        ws = [np.array([[1,2],[3,4]]), np.array([[5],[6]])]
        bs = [np.array([[1]]), np.array([2])]
        self.__fake_weight_update(mock_net, ws, bs)
        weight_stats.update()

        ws = [np.array([[7, 8], [9, 10]]), np.array([[11], [12]])]
        bs = [np.array([[3]]), np.array([4])]
        self.__fake_weight_update(mock_net, ws, bs)
        weight_stats.update()

        ws = [np.array([[13, 14], [15, 16]]), np.array([[17], [18]])]
        bs = [np.array([[5]]), np.array([6])]
        self.__fake_weight_update(mock_net, ws, bs)
        weight_stats.update()

        w = weight_stats.get_weights_for_plot()
        self.assertEqual([[1,7,13], [2,8,14], [3, 9, 15], [4, 10, 16], [5, 11, 17], [6, 12, 18]], w)
        b = weight_stats.get_biases_for_plot()
        self.assertEqual([[1,3,5], [2,4,6]], b)
        c = weight_stats.get_cost_for_plot()
        self.assertEqual([100, 100, 100], c)



    def __fake_weight_update(self, mock_net, ws, bs):
        mock_net.weights = []
        mock_net.biases = []
        self.__fake_update(mock_net.weights, ws)
        self.__fake_update(mock_net.biases, bs)

    def __fake_update(self, to_update, ws):
        for i in xrange(len(ws)):
            to_update.append(ws[i])

def cost_mock(coord):
    """Make a fake function that lets us implictly check if cost took the right arguments
    and summed across them (bad practice, should really test these seperately)"""
    x, y = coord;
    return x + y


