import srcNN.visual.plot_data as pd
from srcNN.network import network_runner as nr
from srcNN.network import network as nn
import experiment
from srcNN.network.network_domain import HyperParameters
from srcNN.visual import weight_stats_drawer as w_drawer
import ThreeD_parabola as tdp
import repeat_report as rr
from srcNN.network import weight_init as wi

"""Make the data"""
n = 50
c = 3
#train = tdp.make_data_set(n, c)
train = tdp.make_bulls_eye(n, 1)


"""Prepare the network"""
network_runner = nr.NetworkRunner()
sizes = [1 ,2, 1]
#non_flip = wi.non_flip_normal_2layer(sizes)
#balanced_sign = wi.balanced_sign_init(sizes)
#network = nn.relu_with_linear_final(sizes, weights=balanced_sign)
#network = nn.relu_with_linear_final(sizes)

"""Run the experiment"""
exp = experiment.Experiment(train)
hp = HyperParameters(50, 500, 0.1)
rr.below_thresh(100, 5, hp, exp)
#weight_stats = network_runner.sgd_experiment(network, hp, exp)

"""Plot the data and the results"""
#pd.plot_data_2D(train, network)
#GraphDrawer().draw(weight_stats)