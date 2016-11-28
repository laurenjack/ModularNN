import random

import matplotlib.pyplot as plt

import srcNN.data.mnist_loader as ml


def pick(training_data, digit):
    """For example, if digit=2, pick the first 2 in the training data"""
    for x, y in training_data:
        if y[digit][0] == 1.0:
            return x
    return None

# Load the data
training_data, validation_data, test_data = ml.load_data_wrapper()
random.shuffle(training_data)

a_two = pick(training_data, 2)
plt.figure(1, facecolor='white')
ax = plt.gca()
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
plt.imshow(a_two.reshape(28, 28), interpolation="nearest" ,cmap='Greys')
plt.show()