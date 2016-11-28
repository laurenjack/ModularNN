import numpy as np
import random

def get_first_direction(data, epochs, eta):
    #Get the number of input dimensions
    p = data[0][0].shape[0]
    #Create the weight matrix that specifies direction
    w = np.random.uniform(0, 1, (1, p))
    w = w/np.linalg.norm(w)

    #Train the weights to maximally relu seperate across 2 dimensions
    for i in xrange(epochs):
        random.shuffle(data)
        for x_batch, y_batch in data:
            pass


