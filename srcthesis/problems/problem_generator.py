import copy
import numpy as np

from problem_domain import *
from srcthesis.experiment.experiment import *

class ProblemGenerator:

    def __init__(self, rand):
        self.rand = rand

    def create_experiment_from_tuple_list(self, tuple_list):
        """Given a list of binary tuples, where each tuple has an input list and an output list,
        turn these input output pairs into a list of tuples where the input/outputs are ndarrays
        ready for use in a network
        """
        network_ready = []
        base_obs = []
        for tup in tuple_list:
            x = self.__to_ndarray(tup[0])
            y = self.__to_ndarray(tup[1])
            obs = Observation(x, y)
            base_obs.append(obs)
            nd_tup = x, y
            network_ready.append(nd_tup);
        return Experiment(network_ready, base_obs)

    def create_matricies_from_tuple_list(self, tuple_list):
        """Given a list of binary tuples, where each tuple has an input list and an output list,
        turn these input output pairs into a tuple of matricies (X, Y)
        """
        x_list, y_list = zip(*tuple_list)
        X = np.array(x_list)
        Y = np.array(y_list)
        return X, Y


    def generate_base_observations(self, numInputs):
        """
        Generate 2^{numInputs+1} base binary observations, i.e. all possible binary mapping from numInputs
         binary inputs to 1 binary output

        :param numInputs: the number of binary inputs e.g. 1
        :return: for numInputs 1:
        0 -> 0
        0 -> 1
        1 -> 0
        1 -> 1
        """
        return self.__gen_base_observations(numInputs)

    def generate_problems(self, numInputs):
        """Generate a set of binary problems, each problem specifies a unique set of mappings,
         where each mapping  maps 'numInputs' binary inputs to a single binary output

        :param numInputs: The number of binary digits on an input
        :param numProblems: For each problem, how many instances should be generated?
        :return:
        """
        baseObservations = self.__gen_base_observations(numInputs)
        problems = self.__gen_problems(baseObservations, numInputs**2)
        return ProblemSet(baseObservations, problems)

    def uniform_random_observations(self, base_obs, num_to_make):
        """
        Given a generated set of base observations, create a test/train set of observations of size
        num_to_make.

        Each observation is selected randomly in a uniform fashion from that base set.

        :param base_obs:
         The base observations, for w
        :return:
           the problem set, for training/testing a machine learning algorithm. As a list
           of (input, output) tuples
        """
        num_base_obs = len(base_obs)
        data_set = []
        for i in xrange(num_to_make):
            #uniform random choice of base observation
            randIndex = self.rand.randrange(0, num_base_obs)
            selected = base_obs[randIndex]
            asTuple = selected.as_tuple()
            data_set.append(asTuple)
        return data_set

    def random_linear_model(self, num_weights):
        weights = np.random.randn(1, num_weights)
        bias = np.random.randn(1,1)
        observations = []
        tup_list = []
        for i in xrange(num_weights*2+2):
            x = np.random.randn(num_weights, 1)*0.5
            y = np.dot(weights, x)+bias
            obs = Observation(x, y)
            observations.append(obs)
            tup_list.append((x, y))
        return [weights], [bias], Experiment(tup_list, observations)

    def __gen_base_observations(self, numInputs):
        inputs = self.__gen_inputs(numInputs, numInputs)
        return self.__pair_outputs_with(inputs)

    def __gen_inputs(self, numInputs, index):
        if index == 0:
            return [np.zeros((numInputs, 1))]
        zeros = self.__gen_inputs(numInputs, index-1)
        ones = copy.deepcopy(zeros)
        for array in ones:
            array[-index] = 1
        return zeros + ones

    def __pair_outputs_with(self, inputs):
        problems = []
        for input in inputs:
            problems.append(Observation(input, np.zeros((1, 1))))
            problems.append(Observation(input, np.ones((1, 1))))
        return problems

    def __gen_problems(self, baseObservations, instIndex):
        if instIndex == 0:
            evenInstances = baseObservations[::2]
            return [Problem(evenInstances)]
        initialProblems = self.__gen_problems(baseObservations, instIndex - 1)
        changedProblems = copy.deepcopy(initialProblems)
        for p in changedProblems:
            p.observations[-instIndex] = baseObservations[-instIndex * 2 + 1]
        return initialProblems + changedProblems

    def __to_ndarray(self, list):
        """Turn a list of numbers into a vector of numbers"""
        dim = len(list)
        vector = np.zeros(shape=(dim, 1))
        for i in xrange(dim):
            vector[i, 0] = list[i]
        return vector




