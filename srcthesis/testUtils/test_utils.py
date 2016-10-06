import numpy as np

def numpy_array_vector(values):
    num_values = len(values)
    arr = np.zeros((num_values, 1))
    for i in xrange(num_values):
        arr[i,0]= values[i]
    return arr