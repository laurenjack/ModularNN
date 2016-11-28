import numpy as np

print np.version

class Observation:
    """An individual mapping of inputs to outputs:

    Attributes:
        inputs (numpy array)
        outputs(numpy array)

    eg1: 0, 0 -> 1
    eg2: 0.5 -> 2
    eg3: 1, 0, 1, 0 -> 1, 1
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        in_str = self.__commaSeparated(self.inputs)
        out_str = self.__commaSeparated(self.outputs)
        return  in_str + " -> " + out_str

    def as_tuple(self):
        """
        :return: The inputs and outputs as a tuple for convenience
        """
        return (self.inputs, self.outputs)

    def __commaSeparated(self, to_sep):
        com_seperated = str(to_sep[0])
        if len(to_sep) == 1:
            return com_seperated
        for i in xrange(1, len(to_sep)):
            com_seperated+=", "+str(to_sep[i])
        return com_seperated


class Problem:
    """ A set of observations, for each Observation, an algorithm hopes to solve a problem by
    correctly identifying the mapping between each input and output.

    Attributes:
        observations ( list of Observation ): The observations that make up the problem

    e.g1:
    0, 0 -> 0
    0, 1 -> 1
    0, 1 -> 1
    1, 1 -> 1
    """
    def __init__(self, observations):
        self.observations = observations

    def __str__(self):
        obs_list=""
        for obs in self.observations:
            obs_list+= str(obs)+"\n"
        return obs_list

class ProblemSet:
    """A problem set describes a base set of observations and problems, which can be used to produce a range
    of problems. The point of a problem set is to have a way to compare machine learning algorithms by
    training on anything from a single instance, to a difficult problem:

    Attributes:
        baseObservations (list of Observation): Every unique observation for this problem set, the base problems
        are all part of the super set of these base observations.

        baseProblems (list of Problem): A list of unique problems, the observations of each
        problem are formed from the baseObservations.

    e.g1: The set of all binary functions where there are two binary outputs and 1 binary input, such that all
        four combinations of the two binary inputs are present in each problem
    """
    def __init__(self, baseObservations, baseProblems):
        self.baseObservations = baseObservations
        self.baseProblems = baseProblems

class ProblemFactory:

    def create_observation(self, inputs, outputs):
        in_size = len(inputs)
        out_size = len(outputs)
        array_in = np.empty((in_size, 1))
        array_out = np.empty((out_size, 1))
        self.__fill(array_in, inputs)
        self.__fill(array_out, outputs)
        return Observation(array_in, array_out)

    def __fill(self, array, list):
        for i in xrange(len(list)):
            array[i, 0]= list[i]