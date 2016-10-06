from srcthesis.network.network_runner import NetworkRunner
from srcthesis.network import network as nn
from srcthesis.network import weight_init as wi
import experiment
import ThreeD_parabola as tdp

network_runner = NetworkRunner()

def below_thresh(t, thresh, hp, exp=None):
    """Run an experiment for t trials, report how often the error gets below
    the threshold thresh."""
    under_count = 0
    for i in xrange(t):
        #network = nn.relu_with_linear_final([1, 2, 1])
        sizes = [1, 2, 1]
        #non_flip = wi.non_flip_normal_2layer(sizes)
        train = tdp.concave_uniform()
        exp = experiment.Experiment(train)
        #balanced_non_flip = wi.balanced_non_flip(sizes)
        just_non_flip = wi.non_flip_normal_2layer(sizes)
        biases = wi.relu_simple_biases(0.6, sizes)
        network = nn.relu_with_linear_final(sizes, weights=just_non_flip, biases=biases)
        weight_stats = network_runner.sgd_experiment(network, hp, exp)
        cost = weight_stats.get_cost_for_plot()
        if cost[-1] < thresh:
            under_count = under_count + 1
        print "Experiment {0}".format(i)
    under_percent = float(under_count)/float(t)*100
    print "{0}/{1} below the threshold: {2}%".format(under_count, t, under_percent)


