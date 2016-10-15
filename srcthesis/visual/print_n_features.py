import random
import numpy as np
import srcthesis.data.optimal_nets as on
import srcthesis.network.network_factory as nf
import srcthesis.network.network_runner as nr
import srcthesis.run.mnist_loader as ml
import matplotlib.pyplot as plt

# Load network hyper-parameters
sizes, act_strings, hypers = on.get_optimal('sig-or-sm')
hypers = [hypers[1], hypers[1]]
network = nf.mix_network([784, 30, 10], ['or', 'and'], hypers)
#network = nf.mix_network(sizes, act_strings, hypers)

# Load the data
training_data, validation_data, test_data = ml.load_data_wrapper()
# Make validation data trainable
val_inputs = [x for x, y in validation_data]
validation_results_vec = [ml.vectorized_result(y) for x, y in validation_data]
validation_data = zip(val_inputs, validation_results_vec)
training_data.extend(validation_data)
runner = nr.NetworkRunner()

#Train the network
test_errors = runner.sgd_tracking_error(network, training_data, 10, 30, test_data)

def pick_n(first_layer, n):
    num_feat = first_layer.shape[0]
    row_inidicies = np.random.choice(num_feat - 1, n, replace=False)
    return first_layer[row_inidicies]


#Randomly pick 5 features and plot them
n = 5
random_feats = pick_n(network.weights[0], n)
sub = 101 + 10*n
plt.figure(1, facecolor='white')
for feat in random_feats:
    ax = plt.subplot(sub)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    plt.imshow(feat.reshape(28, 28), interpolation="nearest", cmap='Greys')
    sub += 1

#Now illustrate the second layer via histograms
plt.figure(2, facecolor='white')
ax = plt.gca()
ax.yaxis.set_visible(False)
plt.hist(network.weights[1].flatten(), bins=50, normed=True, stacked=True)
plt.show()