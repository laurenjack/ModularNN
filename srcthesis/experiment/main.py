from srcthesis.problems.problem_generator import ProblemGenerator
from srcthesis.problems.problem_domain import Observation
from srcthesis.problems.problem_domain import ProblemFactory
from srcthesis.network import network_domain as nd
from srcthesis.network import network_runner as nr
from srcthesis.network import sigmoid as s
from srcthesis.network import noisy_or as nor
from srcthesis.network import noisy_and as nand
from srcthesis.network import linear as ln
from srcthesis.visual.weight_tracker import NodeStats
from srcthesis.visual.graph_drawer import GraphDrawer
from srcthesis.problems.random_wrapper import Rand
from experiment import ExperimentFactory
import numpy as np

print np.version

runner = nr.NetworkRunner()
# network = s.Sigmoid([2, 1])
# hyperParams = nd.HyperParameters(batchSize=30, epochs=1, eta=3.0)
# network = nor.NoisyOr([2, 1])
# hyperParams = nd.HyperParameters(batchSize=30, epochs=100, eta=0.3)
# network = nand.NoisyAnd([2, 1])
# hyperParams = nd.HyperParameters(batchSize=30, epochs=1, eta=3.0)
network = ln.Linear([1, 1])
hyperParams = nd.HyperParameters(batchSize=5, epochs=10, eta=0.3)
rand = Rand()
pg = ProblemGenerator(rand)
exp_fact = ExperimentFactory(pg)
obs_fact = ProblemFactory()





#obs = obs_fact.create_observation([1, 0], [1])
#exp = exp_fact.create_single_binary_observation(obs, 3000)

"""
    0, 0 -> 1
    0, 1 -> 0
    1, 0 -> 1
    1, 1 -> 0
    """
# baseProblems = pg.generate_problems(2).baseProblems
# obs = baseProblems[7].observations
# exp = exp_fact.create_uniform_experiment(obs, 1000)
#tup_list = [([-2],[7]), ([-1],[6]), ([0],[5]), ([1],[4]), ([2],[3]), ]
tup_list = [([1],[4]), ([2],[3]), ([0],[5]), ([1],[4]), ([2],[3]), ]
#tup_list = [([-2],[7]), ([-1],[6]), ([0],[5]), ([-3],[8]), ([-4],[9]), ]
#tup_list = [([3],[2]), ([4],[1]), ([0],[5]), ([-3],[8]), ([-4],[9]), ]
#exp = pg.create_experiment_from_tuple_list(tup_list)
#result = runner.sgd_experiment(network, hyperParams, exp)
#GraphDrawer().draw_graph(result)