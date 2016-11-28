import numpy as np
import random

def bin_vector(dim, j):
    """Puts a one in the jth position of the vector"""
    e = np.zeros(dim)
    e[j] = 1.0
    return e

def load_digits(name, nn=True):
    #Load the data from the file
    digit_data = np.loadtxt(name, dtype=int)
    n, num_pixels = digit_data.shape
    num_pixels -= 1

    #Split data into inputs and outputs
    inputs = []
    outputs = []
    for i in xrange(n):
        #Get all the pixels, i.e. all in this row but the last
        inputs.append(digit_data[i, 0:-1].astype(float).reshape(num_pixels, 1))
        #The last element (at num_pixels) is the class
        outputs.append(digit_data[i][num_pixels])

    #The inputs are nicely sorted in a balance fashion,
    #I will keep them balanced when doing the test/train
    #split
    train_in = []
    train_out = []
    test_in = []
    test_out = []

    #Generate a random starting point in the data,
    #Always a multiple of 10
    begin = 10*random.randint(0, 99)
    test_end = begin+500
    train_end = begin+ 1000

    #Fill up the test set
    for i in xrange(begin, test_end):
        k = i % 1000
        test_in.append(inputs[k])
        test_out.append(outputs[k])

    #Fill up the training set with the remaining entries
    for i in xrange(test_end, train_end):
        k = i%1000
        train_in.append(inputs[k])
        train_out.append(outputs[k])

    #Create outputs in their one hot vectorized form,
    #for training the neural network
    if nn:
        train_out = [bin_vector((10, 1), o) for o in train_out]

    #Pair inputs with their corresponding outputs
    train = zip(train_in, train_out)
    test = zip(test_in, test_out)
    random.shuffle(train)
    random.shuffle(test)
    return train, test
