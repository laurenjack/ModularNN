from srcthesis.visual import error_drawer as ed

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


from srcthesis.network import network_runner as nr
from srcthesis.network import sigmoid as s
from srcthesis.network import relu as r
from srcthesis.network import noisy_and as na
from srcthesis.network.sigOR import SigOr
from srcthesis.network.sigAND import SigAND

runner = nr.NetworkRunner()
#network = na.NoisyAnd([784, 30, 10])
#print "Pur Sigmoid"
#network = s.Sigmoid([784, 30, 10])
#network = r.Relu([784, 30, 10])
#sig_error = runner.sgd_tracking_error(network, training_data, 10, 30, 0.06, test_data=test_data)

#print "cont-OR"
#network = SigOr([784, 30, 30, 10])
#or_error = runner.sgd_tracking_error(network, training_data, 10, 45, (3.0, 0.1), test_data)

#print "cont-AND"
network = SigAND([784, 30, 30, 10])
and_error = runner.sgd_tracking_error(network, training_data, 10, 45, (3.0, 2.0), test_data)

#ed.draw_error_graph("Classification Error on MNIST", [("pure sigmoid", sig_error), ("cont-OR", or_error), ("cont-AND", and_error)])
#network = SigAND([784, 30, 10])
# AND train_errors, test_errors = runner.sgd_tracking_error(network, training_data, 10, 30, (3.0, 1.0), test_data)
# OR train_errors, test_errors = runner.sgd_tracking_error(network, training_data, 10, 30, (3.0, 0.1), test_data)
#train_errors, test_errors = runner.sgd_tracking_error(network, training_data, 10, 30, 3.0, test_data)
#ed.draw_error_graph("cont - OR 0.1", train_errors, test_errors)
#import network
#net = network.Network([784, 30, 10])
#net.SGD(training_data, 30, 10, 3.0, test_data=test_data)