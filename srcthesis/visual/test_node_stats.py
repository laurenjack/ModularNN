from unittest import TestCase
import mock
from srcthesis.network.sigmoid import Sigmoid
from srcthesis.problems.problem_domain import Observation
from srcthesis.visual.visual_domain import NodeStats
from srcthesis.experiment.experiment import BinaryObservation
import numpy as np


# 00 -> 1
in1 = np.zeros((2, 1))
obs1 = Observation(in1, np.ones((1, 1)))
# 11 -> 0
in2 = np.ones((2, 1))
obs2 = Observation(in2, np.zeros((1, 1)))

class NodeStatsSpec(TestCase):

    def test_whenSingleNodeWithMultipleWeightsAndExpWithMultipleInstances_ThenUpdateCorrect(self):
        #Mock up the network
        mock_net = mock.create_autospec(Sigmoid)
        mock_net.feedforward = mock.Mock(side_effect=feedforward_mock)
        mock_net.weights = [np.zeros((1, 3))]
        mock_net.biases = [np.zeros((1, 1))]

        #mock the experiment
        mock_exp = mock.create_autospec(BinaryObservation)
        mock_exp.base_obs = [obs1, obs2]

        node_stats = NodeStats(mock_net, mock_exp)

        # fake the update of the network
        ws = mock_net.weights[0]
        ws[0,0] = 11
        ws[0,1] = 22
        ws [0,2] = 33
        mock_net.biases[0][0,0] = 44

        #call update on node_stats
        node_stats.update()

        #verify calls made correctly
        calls = [mock.call(in1), mock.call(in2)]
        mock_net.feedforward.assert_has_calls(calls)

        #verify updated correctly
        self.assertEquals([[11], [22], [33]], node_stats.weights_lists)
        self.assertEquals([44], node_stats.biases)
        self.assertEquals({str(obs1): [888], str(obs2): [999]}, node_stats.obs_outputs)




mock_act1 = np.zeros((1,1))
mock_act1[0,0] = 8
mock_act2 = np.zeros((1,1))
mock_act2[0,0] = 9

class MockNetwork:

    def __init__(self):
        self.weights = [np.zeros((1,3))]
        self.biases = [np.zeros((1,1))]

    def feedforward(self, input):
        pass

class MockExp:

    def __init__(self):
        # 00 -> 1
        obs1 = Observation(np.zeros((2,1)), np.ones(1,1))
        # 11 -> 0
        obs2 = Observation(np.ones((2,1)), np.zeros(1, 1))
        self.base_obs = [obs1, obs2]


def feedforward_mock(*args, **kwargs):
    act = np.zeros((1, 1))
    if np.array_equal(args[0], in1):
        act[0,0] = 888
    elif np.array_equal(args[0], in2):
        act[0,0] = 999
    return act