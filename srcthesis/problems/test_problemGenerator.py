import numpy as np
from unittest import TestCase
from problem_generator import ProblemGenerator
from srcthesis.problems.problem_domain import Observation
from srcthesis.problems.random_wrapper import RandRangeMock


class ProblemGeneratorSpec(TestCase):

    def test_whenTwoInputs_ThenAll16Problems(self):
        pg = ProblemGenerator(None)
        pSet = pg.generate_problems(2);
        baseObs = pSet.baseObservations
        problems = pSet.baseProblems

        #Assertions for baseObservations
        """ 00 -> 0
            00 -> 1
            01 -> 0
            01 -> 1
            10 -> 0
            10 -> 1
            11 -> 0
            11 -> 1
             """
        self.__assertObservation(0, 0, 0, baseObs[0])
        self.__assertObservation(0, 0, 1, baseObs[1])
        self.__assertObservation(0, 1, 0, baseObs[2])
        self.__assertObservation(0, 1, 1, baseObs[3])
        self.__assertObservation(1, 0, 0, baseObs[4])
        self.__assertObservation(1, 0, 1, baseObs[5])
        self.__assertObservation(1, 1, 0, baseObs[6])
        self.__assertObservation(1, 1, 1, baseObs[7])

        #Assertions for problems
        self.assertEqual(16, len(problems))
        for p in problems:
            self.assertEqual(4, len(p.observations))
        p0 = problems[0]
        """ 00 -> 0
            01 -> 0
            10 -> 0
            11 -> 0
            """
        self.__assertObservation(0, 0, 0, p0.observations[0])
        self.__assertObservation(0, 1, 0, p0.observations[1])
        self.__assertObservation(1, 0, 0, p0.observations[2])
        self.__assertObservation(1, 1, 0, p0.observations[3])

        p3 = problems[3]
        """ 00 -> 0
            01 -> 0
            10 -> 1
            11 -> 1
            """

        self.__assertObservation(0, 0, 0, p3.observations[0])
        self.__assertObservation(0, 1, 0, p3.observations[1])
        self.__assertObservation(1, 0, 1, p3.observations[2])
        self.__assertObservation(1, 1, 1, p3.observations[3])


    def __assertObservation(self, in0, in1, output, obs):
        self.assertEqual(in0, obs.inputs[0])
        self.assertEqual(in1, obs.inputs[1])
        self.assertEqual(output, obs.outputs[0])


    def test_whenGenerateUniformRand_ThenObservationsFromBase(self):
        rand = RandRangeMock([1,1, 0, 1])
        pg = ProblemGenerator(rand)
        #Create the base observations
        obs0 = Observation([0,0], 0)
        obs1 = Observation([1, 0], 1)

        result = pg.uniform_random_observations([obs0, obs1], 4)
        self.assertEqual(obs1.as_tuple(), result[0])
        self.assertEqual(obs1.as_tuple(), result[1])
        self.assertEqual(obs0.as_tuple(), result[2])
        self.assertEqual(obs1.as_tuple(), result[3])

    def test_whenCreateProblemFromTupleListOfDiffSizes_ThenReturnCorrectVectors(self):
        pg = ProblemGenerator(None)
        tuple_list = [([-2], [7, 5]), ([7], [-1]), ([3, 3], [-10])]

        result = pg.create_experiment_from_tuple_list(tuple_list)
        base_obs = result.base_obs
        train = result.train

        in0 = train[0][0]
        self.assertEqual(in0.shape, (1,1))
        self.assertEqual(-2, in0[0,0])

        out0 = train[0][1]
        self.assertEqual(out0.shape, (2, 1))
        self.assertEqual(7, out0[0, 0])
        self.assertEqual(5, out0[1, 0])

        obs2 = base_obs[2]
        self.assertEqual(3, obs2.inputs[1,0])
        self.assertEqual(-10, obs2.outputs[0, 0])

    def test_when_create_matrix_tuple_list_then_right_size_and_order(self):
        pg = ProblemGenerator(None)
        tuple_list = [([0.5, 1.5],[-2.5]), ([1, 1],[-2]), ([1.5, 0.5],[-1.5])]

        X, Y = pg.create_matricies_from_tuple_list(tuple_list)

        xExp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5]])
        yExp = np.array([[-2.5], [-2], [-1.5]])

        self.assertTrue(np.array_equal(xExp, X))
        self.assertTrue(np.array_equal(yExp, Y))




