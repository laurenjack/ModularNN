

class Experiment:

    def __init__(self, train, base_obs=None, test=None):
        self.base_obs = base_obs
        self.train = train
        self.test = test


class ExperimentFactory:

    def __init__(self, problem_generator):
        self.pg = problem_generator

    def create_single_binary_observation(self, observation, train_n):
        """
        Create an experiment using a single binary observation, i.e. the training set is just
        train_n of the observation passed in

        :param observation: The single binary observation
        :param train_n: The size of the training set generated, which is just observation
        repeated train_n times

        :return: Experiment ready to be passed to the network runner
        """
        base_obs = [observation]
        return self.create_uniform_experiment(base_obs, train_n)

    def create_uniform_experiment(self, base_observations, train_n):
        """Create an experiment with a training set of size train_n, where each element is uniform random
        selected from base_observations (with replacement)

        :param base_observations: The base observations which form this problem
        :param train_n: The size of the training set generated, which is filled of repeats of the base
        observations

        :return: Experiment ready to be passed to the network runner
        """
        train = self.pg.uniform_random_observations(base_observations, train_n)
        return Experiment(train, base_observations)


class BinaryObservation(Experiment):

    def __init__(self, base_obs, train):
        super(BinaryObservation, self).__init__(train, base_obs)
        # self.base_obs = [prob_gen.generate_base_observations(2)[4]]
        # self.train = prob_gen.uniform_random_observations(self.base_obs, 3000)
        # self.test = None

    def update(self, network):
        pass
        # self.node_stats.weights.append(network.weights[0])
        # w_vector = network.weights[0]
        # num_weights = np.shape(w_vector)
        # for i in xrange(num_weights):
        #     self.weight_lists[i].append(w_vector[0, i])
        # self.node_stats.biases.append(network.biases[0][0,0])
        # for obs in self.base_obs:
        #     out_act = network.feedforward(obs.inputs)
        #     self.node_stats.obs_outputs[str(obs)].append(out_act[0,0])

    def result(self):
        pass
        #return self.node_stats



