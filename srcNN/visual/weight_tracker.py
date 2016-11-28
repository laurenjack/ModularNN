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


class ModeStats:
    """Responsible for storing weights and mode strength over time
    It's assumed the network used is a two layer network"""

    def __init__(self, network, train, X, U, Vt):
        self.net = network

        #Used to keep track of mode strength
        self.train = list(train)
        self.X = X
        self.Ut = U.transpose()
        self.V = Vt.transpose()

        self.n = len(train)
        self.num_out = network.weights[-1].shape[0]

        self.mode_mats = []

    def update(self):
        #matrix of activations currently produced by the network
        A = np.zeros((self.num_out, self.n))
        for xy, i in zip(self.train, xrange(self.n)):
            a = self.net.feedforward(xy[0])
            A[:,i] = a[:,0]
        #Find the covariance matrix of the network output
        #relative to the input.
        sigma31_net = (A - A.mean(axis=0)).dot(self.X.transpose())
        #self.mode_mats.append(sigma31_net)
        self.mode_mats.append(self.Ut.dot(sigma31_net).dot(self.V))

    def get_diagonals(self):
        """Return the modes, i.e the diagonals of the weight products
        that were stored at the """
        #Create a dictionary with an entry for each mode
        diags = {}
        for i in xrange(self.mode_mats[0].shape[0]):
            diags[i] = []

        for mat in self.mode_mats:
            for i in xrange(mat.shape[0]):
                diags[i].append(mat[i, i])
        return diags

    def get_max_non_diags(self):
        """Return the maximum non diagonals of the matrix A.Sigma31_net.Xt
        This will allow us to get a rough idea of how independent the modes
        were"""
        #Find the top 5 max non-diagonals
        num_maxes = 5
        final_mat = self.mode_mats[-1]
        m, n = final_mat.shape
        max_ijs = self.init_max_array(num_maxes)
        for i in xrange(m):
            for j in xrange(n):
                e = final_mat[i, j]
                if i != j and abs(e) > max_ijs[0][2]:
                    max_ijs[0] = (i, j, abs(e))
                #Sort so that the smallest max is last
                max_ijs = sorted(max_ijs, key=lambda ije : ije[2])

        #Return the mode stregnths of the top 5 non-diagonals over time
        non_diags = {}
        for i, j, _ in max_ijs:
           non_diags[(i, j)] = []

        for mat in self.mode_mats:
            for i, j, _ in max_ijs:
                non_diags[(i, j)].append(mat[i, j])
        return non_diags


    def init_max_array(self, num_maxes):
        """Initialize an array with a bunch of negatives, so that any positive
        number will be greater than the final element of each tuple 'e'"""
        return [(-1, -1, -1)]*num_maxes


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