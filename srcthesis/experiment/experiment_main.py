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
#train = tdp.make_bulls_eye(n, 1)
#train = tdp.lop_sided_zig_zag()
train = tdp.concave_uniform()

"""network setup"""
network_runner = nr.NetworkRunner()
sizes = [1 ,2, 1]

"""Run the experiment"""
exp = experiment.Experiment(train)
hp = HyperParameters(10, 500, 0.1)
rr.below_thresh(200, 5, hp)