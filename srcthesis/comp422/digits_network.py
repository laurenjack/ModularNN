import numpy as np
import srcthesis.network.network_runner as nr
import srcthesis.network.network_factory as nf
import data_processor as dp

train, test = dp.load_digits("digits/digits20")

#Construct the network
runner = nr.NetworkRunner()


network = nf.mix_network([49, 100, 100, 10], ['relu', 'relu', 'sm'], [0.1, 0.1, 0.1], dropout=False)
errors = runner.sgd_tracking_error(network, train, 50, 1000, test)

