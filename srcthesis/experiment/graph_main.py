import srcthesis.visual.plot_data as pd
from srcthesis.network import network_runner as nr
from srcthesis.network import network as nn
from srcthesis.network import relu as r
import experiment
from srcthesis.network.network_domain import HyperParameters
from srcthesis.visual.graph_drawer import GraphDrawer
import ThreeD_parabola as tdp
import repeat_report as rr
from srcthesis.network import weight_init as wi
import numpy as np

"""Make the data"""
n = 50
c = 3
#train = tdp.make_data_set(n, c)
#train = tdp.make_bulls_eye_plain(n, 1)
#train = tdp.lop_sided_zig_zag()
train = tdp.concave_uniform()
#train = tdp.make_variable_bulls_eye(n, 3)



"""Prepare the network"""
network_runner = nr.NetworkRunner()
sizes = [1 ,2, 1]
#biases = [0.2*np.ones((y, 1)) for y in sizes[1:]]
#non_flip = wi.non_flip_normal_2layer(sizes)
#balanced_non_flip = wi.balanced_non_flip(sizes)
biases = wi.relu_simple_biases(0.6, sizes)
network = nn.relu_with_linear_final(sizes, weights=None, biases=biases)
#network = nn.relu_with_linear_final(sizes)

"""Run the experiment"""
exp = experiment.Experiment(train)
hp = HyperParameters(10, 500, 0.01)
weight_stats = network_runner.sgd_experiment(network, hp, exp)

"""Plot the data and the results"""
#pd.plot_data_2D(train, network)
GraphDrawer().draw(weight_stats)