import numpy as np


class WeightStats:
    """Tracks the weights and their gradients after a full bout of training"""

    def __init__(self, network, train):
        self.network = network
        self.train = train
        self.w_over_time = []
        self.b_over_time = []
        self.cost_over_time = []

    def update(self):
        """Put the next entries of weights and bias in the weights stats"""
        next_w = self.network.weights
        next_b = self.network.biases
        self.w_over_time.append(next_w)
        self.b_over_time.append(next_b)
        cost = reduce(lambda tot, coord: tot + self.network.cost(coord), self.train, 0)
        self.cost_over_time.append(cost)

    def get_weights_for_plot(self):
        """Get the weights as a 2D list, reflecting each weight in the network over time"""
        return self.__get_weights(self.w_over_time)


    def get_biases_for_plot(self):
        """Get the biases as a 2D list, reflecting each weight in the network over time"""
        return self.__get_weights(self.b_over_time)

    def get_cost_for_plot(self):
        return self.cost_over_time

    def __get_weights(self, weights):
        num_weights = self.__get_num_of(weights)
        weights_2D = []
        for i in xrange(num_weights):
            weights_2D.append([])
        for w_list in weights:
            i = 0
            for w_mat in w_list:
                for w in np.nditer(w_mat):
                    weights_2D[i].append(w)
                    i = i + 1
        return weights_2D

    def __get_num_of(self, weights):
        #Get the first weights updated to gauge distance
        return reduce(lambda tot, w: tot + w.size, weights[0], 0)



class NodeStats:
    """Information regarding a nodes output and its weights after a full bout of training.

    """

    def __init__(self, network, exp):
        self.network = network
        self.exp = exp

        # Create dictionary for each observation, with a list of activiations
        self.obs_outputs = {}
        self.obs_costs = {}
        for obs in exp.base_obs:
            obs_key = str(obs)
            self.obs_outputs[obs_key] = []
            self.obs_costs[obs_key] = []

        #set up lists for tracking weights and biases
        self.num_weights = np.shape(network.weights[0])[1]
        self.weights_lists = [];
        for i in xrange(self.num_weights):
            self.weights_lists.append([])
        self.biases = []

    def get_outputs(self):
        """
        :return: A dictionary from the name of each base observation, to a list of outputs, the outputs
        present after each update.
        """
        #
        obs_out_scalars = {}

        for name, out_vector in self.obs_outputs.iteritems():
            obs_out_scalars

    def get_weights(self):
        """
        Get the list, of weight lists associated with this node. i.e. the weights that were present
        after each update.

        For example, suppose this node had 3 weights. This method would return a list of 3 lists,
        each list is for a single weight. Each list has the value of that weight after the (j+1)th update
        of the node.

        :return: A list of weight lists, where weight_lists[i][j] represents the ith weight on the node
        after the (j+1)th update.
        """

        weight_lists = []
        #Create list for each weight
        w_vector = self.weights[0]
        num_weights = np.shape(w_vector)
        for i in xrange(num_weights):
            weight_lists.append([])

        #populate each list
        for w_vector in self.weights:
            for i in xrange(num_weights):
                weight_lists[i].append(w_vector[i,0])

        return weight_lists

    def get_biases(self):
        """
        Get the list of biases associated with this node, i.e the bias that was present after each update.

        :return: A list of biases across update iterations, as scalars for graphing
        """
        bias_scalars = []
        for b in self.biases:
            bias_scalars.append(b[0, 0])
        return bias_scalars

    def update(self):
        """
        Update the stored sets of outputs, weights and biases.

        Should only be called after a gradient descent update has been peformed on this node.
        :return:
        """
        self.__update_outputs_and_cost()
        self.__updateWeights()
        self.__updateBiases()

    def __update_outputs_and_cost(self):
        for obs in self.exp.base_obs:
            obs_key = str(obs)
            out_act = self.network.feedforward(obs.inputs)
            cost = self.network.cost(out_act, obs.outputs)
            self.obs_outputs[obs_key].append(out_act[0, 0])
            self.obs_costs[obs_key].append(cost)

    def __updateWeights(self):
        w_vector = self.network.weights[0]
        for i in xrange(self.num_weights):
            self.weights_lists[i].append(w_vector[0, i])

    def __updateBiases(self):
        self.biases.append(self.network.biases[0][0, 0])

class NullStats:
    """
    A do nothing class, for use when a network is not running an experiment but trying to
    solve a problem in production mode.
    """

    def __init__(self):
        pass

    def update(self):
        pass

def create_stats(network, experiment):
    """Create stats that will track a network as it trains on an experiment.

    If the experiment has base observations, this will produce a NodeStats
    object that tracks how the network performs on each of these observations.

    Otherwise, this will produce a WeightStats object that just tracks weights,
    biases and error"""
    if(experiment.base_obs == None):
        return WeightStats(network, experiment.train)
    return NodeStats(network, experiment)