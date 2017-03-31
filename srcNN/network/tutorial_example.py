import srcNN.data.mnist_loader as mnist
import network_runner as nr
import network_factory as nf
import srcNN.visual.error_drawer as ed

"""This script is intended as a tutorial for those new to this framework,
we train a 3-layer MLP on MNIST and plot the test error over each epoch"""


"""Load 50,000 training images, 10,0000 validation images and 10,000 test images.
Each data set is a list of tuples (x, y) where x is the 784 pixel input vector,
and y is the digit.

For the training data, each y is a 10 element vector that uses 1-hot binary
encoding for training purposes. However in the validation and test set each y is
just an integer digit from 0-9. This means when evaluating accurarcy using
the training set or conversely training on the validation set, it is important
to convert the data set to the appropriate format first."""
training_data, validation_data, test_data = mnist.load_data_wrapper()

"""Load the network runner, this will iterate over the training set num_epochs
times. Each epoch, the training set is split into random mini-batches according
to the batch size. The netowrk runner tracks the training error after each effort
but can also be used to track the gradients, weights and decay the learning rate."""
runner = nr.NetworkRunner()

"""Create the network, the length of sizes is always one more than that of acts,
due to the input layer. See network factory for a look at all the activations that
can be constructed. The hyp_params is usually just an array of learning rates,
corresponding to the units at each layer. However, 'and' units require both a
learning rate eta, followed by a constant scale c (to scale the initial weights).

This factory does everything such as setting up the activations, initializing the
weights and building the optimizers"""
network = nf.mix_network(sizes=[784, 30, 10], acts=['tanh', 'sm'], hyp_params=[0.3, 0.3])
#network = nf.mix_network(sizes=[784, 30, 10], acts=['nor', 'and'], hyp_params=[(0.1, 0.01), (0.1, 0.001)])
#network = nf.mix_network(sizes=[784, 30, 30, 10], acts=['sig', 'and', 'sm'], hyp_params=[0.03, (0.1, 0.01), 0.03])

"""Run the network. Train it using the training set and evaluate on the validation
set"""
validation_errors = runner.sgd(network, training_data, 20, 40, validation_data)

"""Display the validation errors over time using the error drawer"""
ed.draw_error_graph("MNIST valdiation Error", ("tut network", validation_errors))