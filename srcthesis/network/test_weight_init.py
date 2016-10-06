from unittest import TestCase
import numpy as np
import weight_init as wi


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


    def __assert_elem_tots(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for i in xrange(len(expected)):
            self.assertAlmostEqual(expected[i][0], result[i][0])
            self.assertAlmostEqual(expected[i][1], result[i][1])


