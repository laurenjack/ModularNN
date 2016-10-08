import mnist_loader
from srcthesis.network import network_runner as nr
from srcthesis.network import network_factory as nf

#Load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#Make validation data trainable
val_inputs = [x for x, y in validation_data]
validation_results_vec = [mnist_loader.vectorized_result(y) for x, y in validation_data]
validation_data = zip(val_inputs, validation_results_vec)
training_data.extend(validation_data)

runner = nr.NetworkRunner()

#Setup the network
sig_eta = 1.0
sizes = [784, 30, 30, 10]
acts = ['sig', 'or', 'sm']
etas = [0.3, (0.3, 0.1), 0.3]
network = nf.mix_network(sizes, acts, etas)

#run the experiment
runner.sgd_tracking_error(network, training_data, 10, 60, test_data)

