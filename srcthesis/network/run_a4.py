import numpy as np
import random

import assignment_network as an

def prepare_data(filename):
    """Responsible for:
         -reading the data and removing redundant info
         -replacing missing values
         -converting the data into (X, Y) matricies (numpy ndarrays) which can be used by the network   """
    # Read the csv file
    pre_set = np.genfromtxt(filename, delimiter=',')

    # remove index column
    pre_set = np.delete(pre_set, 0, 1)

    # Find mean for each predictor, ignoring nans
    means = np.nanmean(pre_set, axis=0)

    # Find nan indicies
    inds = np.where(np.isnan(pre_set))

    # Replace nans with respective means
    pre_set[inds] = np.take(means, inds[1])

    # Split into predictors and covariates
    X = np.delete(pre_set, 0, 1)
    Y = np.delete(pre_set, np.s_[1:], 1)
    X = X.transpose()
    Y = Y.transpose()
    X = X / np.linalg.norm(X)
    num_splits = X.shape[1]
    xList = np.split(X, num_splits, 1)
    yList = np.split(Y, num_splits, 1)

    return zip(xList, yList)


data = prepare_data("train_no_headers.csv")

random.shuffle(data)

train = data[:135000]
validation = data[135000:]

network = an.Network([10,30,1])
network.SGD(train, 3, 30, 0.03, validation)
test = prepare_data("test_no_headers.csv")


predictions = []
for x in test:
    predictions.append(network.feedforward(x[0]))

import csv

with open('predictions.csv', 'wb') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(predictions)