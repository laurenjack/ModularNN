from unittest import TestCase
from srcthesis.problems.problem_generator import ProblemGenerator
from srcthesis.problems.random_wrapper import Rand
from srcthesis.network import initializer

class TestInitializerSpec(TestCase):

    def test_when_bivariate_with_three_obs_then_minimised_weights(self):
        rand = Rand()
        pg = ProblemGenerator(rand)
        y_eq_2x1_x2_minus_5 = [([0.5, 1.5],[-2.5]), ([1, 1],[-2]), ([1.5, 0.5],[-1.5])]
        #exp = pg.create_experiment_from_tuple_list(y_eq_2x1_x2_minus_5)
        train = pg.create_matricies_from_tuple_list(y_eq_2x1_x2_minus_5)

        result = initializer.min_init(-0.5, train)
        weights = result[0][0]
        biases = result[1][0]

        self.__assert_array_equals([-0.5, -1.5], weights.transpose(), 10)
        self.__assert_array_equals([0], biases, 10)


    def __assert_array_equals(self, expected_list, actual, places):
        size = len(expected_list)
        for i in xrange(size):
            self.assertAlmostEqual(expected_list[i], actual[i, 0], places)
