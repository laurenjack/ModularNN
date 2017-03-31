import numpy as np
import srcNN.network.network_factory as nf
from srcNN.network.network_runner import NetworkRunner
nr = NetworkRunner()

#Create the data
x = np.array([3.0, 1.0]).reshape(2, 1)
y = np.array([1.0]).reshape(1, 1)
train = [(x, y)]
zero = np.zeros((1, 1))

def eqZero(a):
    return np.allclose(a, zero)

count = 0
n = 1000
for i in xrange(n):
    network = nf.mix_network([2, 1], ['relu'], [0.2], no_biases=True)
    before = network.feedforward(x)
    nr.sgd(network, train, 1, 50)
    after = network.feedforward(x)
    if not eqZero(after):
        #print network.feedforward(x)
        count += 1
    if not eqZero(before) and  eqZero(after):
        print network.weights[0]
    #print ""

print count/float(n)
