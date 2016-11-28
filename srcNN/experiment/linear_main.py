import numpy as np
from srcNN.problems.problem_generator import ProblemGenerator
from srcNN.problems.problem_domain import ProblemFactory
from srcNN.network import network_domain as nd
from srcNN.network import network_runner as nr
from srcNN.visual import weight_stats_drawer as w_drawer
from srcNN.problems.random_wrapper import Rand
from experiment import ExperimentFactory
from srcNN.network import network_factory as nf

runner = nr.NetworkRunner()


rand = Rand()
pg = ProblemGenerator(rand)
exp_fact = ExperimentFactory(pg)
obs_fact = ProblemFactory()

#tup_list = [([1],[4]), ([2],[3]), ([0],[5]), ([1],[4]), ([2],[3]), ]
#exp = pg.create_experiment_from_tuple_list(tup_list)
#train = pg.create_matricies_from_tuple_list(tup_list);
#weights, biases = initializer.min_init(0.8, train)
#weights = [np.array([[0.8]])]
#biases = [np.array([[-1]])]

#([1.5, 0.5], [-1.5]),
y_eq_2x1_x2_minus_5 = [([0.5, 1.5], [-2.5]), ([1, 1], [-2]), ([1.5, 0.5], [-1.5]), ([0.2, 1], [-3.6])]
exp = pg.create_experiment_from_tuple_list(y_eq_2x1_x2_minus_5)
train = pg.create_matricies_from_tuple_list(y_eq_2x1_x2_minus_5)
#weights, biases = initializer.min_init(-0.5, train)
weights = [np.array([[-0.5, 0.5]])]
biases = [np.array([[1]])]


#network = ln.Linear([2, 1])
#network = ln.Linear([2, 1], weights, biases)
#hyperParams = nd.HyperParameters(batchSize=4, epochs=100, eta=0.3)
#result = runner.sgd_experiment(network, hyperParams, exp)

true_w, true_b, exp = pg.random_linear_model(30)
xs, ys = zip(*exp.train);
train = np.array(xs)[:,:,0], np.array(ys)[:,:,0]
#weights, biases = initializer.min_init(-0.5, train)
network = nf.mix_network([30, 1], ['lin'], [1.0])
hyperParams = nd.HyperParameters(batchSize=62, epochs=50, eta=1.0)
result = runner.sgd_experiment(network, hyperParams, exp)

print "OG weights: "+str(true_w[0])+"\n"

print network.weights[0]
print network.biases[0]
w_drawer.draw_node_stats(result)
