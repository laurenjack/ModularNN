from unittest import TestCase
from mock import *
from dropout import *
import numpy as np

class DropSysSpec(TestCase):

    @patch('random.randint', return_value = 3)
    def test_when_drop_weights_3_layer_then_last_two_dropped(self, rand_int):
        #Set up the Drop Sys scheme
        sizes = [15, 30, 45, 70]
        n = 5
        drop_sys = DropSys(sizes, n)

        #Generate some generic weights
        weights = [42*np.ones((s1, s0)) for s1, s0 in zip(sizes[1:], sizes[:-1])]

        #Make a call to generate drop mask, then get dropped weights and grads
        drop_sys.new_batch()
        dropped_weights = drop_sys.drop_weights(weights)

        #Verify first layer
        w0 = dropped_weights[0]
        row_start = 18
        row_end = 23
        for i in xrange(w0.shape[0]):
            row = w0[i]
            if i >= row_start and i <= row_end:
                self.assertTrue(np.array_equal(row, 42*np.ones((15))))
            else:
                self.assertTrue(np.array_equal(row, np.zeros((15))))


        start_end = [(18, 23), (27, 35), (42, 55)]
        #Verify other layers
        l = 0
        for w_mat in dropped_weights[1:-1]:
            col_start, col_end = start_end[l]
            row_start, row_end = start_end[l+1]
            for i in xrange(w_mat.shape[0]):
                row = w_mat[i]
                for j in xrange(row.shape[0]):

                    #Check the weight
                    w = row[j]
                    self.assertTrue(w == 42 or w == 0)
                    self.assertTrue(w == 0 or col_start <= j <= col_end and row_start <= i <= row_end)
                    self.assertTrue(w == 42 or j < col_start or j > col_end or i < row_start or i > row_end)
            l += 1

        #Verify last layer
        wL = dropped_weights[-1]
        col_start = 27
        col_end = 35
        for i in xrange(wL.shape[1]):
            col = wL[:,i]
            if i >= col_start and i <= col_end:
                self.assertTrue(np.array_equal(col, 42 * np.ones((70))))
            else:
                self.assertTrue(np.array_equal(col, np.zeros((70))))



