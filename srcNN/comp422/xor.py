import numpy as np
import srcNN.network.network_runner as nr
import srcNN.network.network_factory as nf
import data_processor as dp
import matplotlib.pyplot as plt

#Construct the XOR problem
dim = (2, 1)


inputs = [np.zeros(dim), dp.bin_vector(dim, 0), dp.bin_vector(dim, 1), np.ones(dim)]
outputs = [0, 1, 1, 0]
#Make the outputs in 1-hot vectorized form so the network may train
#vec_outputs = [np.array([[o]]) for o in outputs]
vec_outputs = [dp.bin_vector(dim, o) for o in outputs]
data = zip(inputs, vec_outputs)
data_eval = zip(inputs, outputs)

#Construct the network
runner = nr.NetworkRunner()
#network = nf.mix_network([2,16,2], ['relu', 'sm'], [0.1, 0.1])
# weights = copy.deepcopy(network.weights)
# biases = copy.deepcopy(network.biases)

#Run and track the errors
network = nf.mix_network([2, 8, 2], ['relu', 'sm'], [0.1, 0.1])
errors = runner.sgd(network, data, 2, 100, data_eval)
plt.title("XOR classification Error")
plt.plot(errors)
plt.show()