import numpy as np
import srcthesis.network.network_runner as nr
import srcthesis.network.network_factory as nf
import data_processor as dp

#Construct the XOR problem
dim = (2, 1)


inputs = [np.zeros(dim), dp.bin_vector(dim, 0), dp.bin_vector(dim, 1), np.ones(dim)]
outputs = [0, 1, 1, 0]
#Make the outputs in 1-hot vectorized form so the network may train
#vec_outputs = [np.array([[o]]) for o in outputs]
vec_outputs = [bin_vector(dim, o) for o in outputs]
data = zip(inputs, vec_outputs)
data_eval = zip(inputs, outputs)

#Construct the network
runner = nr.NetworkRunner()
#network = nf.mix_network([2,16,2], ['relu', 'sm'], [0.1, 0.1])
# weights = copy.deepcopy(network.weights)
# biases = copy.deepcopy(network.biases)

#Run and track the errors
count = 0
for i in xrange(1000):
    network = nf.mix_network([2, 8, 2], ['relu', 'sm'], [0.1, 0.1])
    errors = runner.sgd_tracking_error(network, data, 2, 500, data_eval)
    if errors[-1] < 0.2:
        count += 1
print count