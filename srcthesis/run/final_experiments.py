import mnist_loader
from srcthesis.network import network_runner as nr
from srcthesis.network import network_factory as nf

#Load the data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
runner = nr.NetworkRunner()

#Setup the network
sig_eta = 3.0
sizes = [784, 30, 30, 10]
acts = ['sig', 'or', 'sm']
etas = [sig_eta, (0.1, 1.0), sig_eta]
network = nf.mix_network(sizes, acts, etas)

#run the experiment
runner.sgd_tracking_error(network, training_data, 10, 60, test_data)

