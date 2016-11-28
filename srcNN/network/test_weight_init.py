from unittest import TestCase
import numpy as np
import weight_init as wi
from srcNN.problems.problem_generator import ProblemGenerator
from srcNN.problems.random_wrapper import Rand


class WeightInitSpec(TestCase):

    def test_non_flip_nlayer(self):
        x = np.array([1,2,3,4])
        sizes = [3, 3, 4, 2, 2]
        weights = wi.non_flip_normal_nlayer(sizes)
        last_min = 100
        min = last_min
        for w_mat in weights:
            for w in np.nditer(w_mat):
                self.assertLess(abs(w), last_min)
                if abs(w) < min:
                    min = abs(w)
            last_min = min


    def test_elements_and_totals(self):
        result = wi.elements_and_totals([5, 10, 16, 2])
        expected = [(0.4, 2), (0.2, 2), (0.125, 2), (0.0625, 0.125)]
        self.__assert_elem_tots(expected, result)

    def test_when_bivariate_with_three_obs_then_minimised_weights(self):
        rand = Rand()
        pg = ProblemGenerator(rand)
        y_eq_2x1_x2_minus_5 = [([0.5, 1.5], [-2.5]), ([1, 1], [-2]), ([1.5, 0.5], [-1.5])]
        # exp = pg.create_experiment_from_tuple_list(y_eq_2x1_x2_minus_5)
        train = pg.create_matricies_from_tuple_list(y_eq_2x1_x2_minus_5)

        result = wi.min_init(-0.5, train)
        weights = result[0][0]
        biases = result[1][0]

        self.__assert_array_equals([-0.5, -1.5], weights.transpose(), 10)
        self.__assert_array_equals([0], biases, 10)

    def __assert_array_equals(self, expected_list, actual, places):
        size = len(expected_list)
        for i in xrange(size):
            self.assertAlmostEqual(expected_list[i], actual[i, 0], places)


    def __assert_elem_tots(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for i in xrange(len(expected)):
            self.assertAlmostEqual(expected[i][0], result[i][0])
            self.assertAlmostEqual(expected[i][1], result[i][1])


