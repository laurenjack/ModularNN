import srcNN.data.mnist_loader as ml
import srcNN.network.network_factory as nf
import srcNN.network.network_runner as nr

act_strings = ['sig', 'sig', 'sig', 'sig', 'sig', 'or', 'sm']
sizes = [784, 30, 30, 60, 60, 120, 120, 10]
hypers = [0.1]*7
hypers[5] = (0.1, 0.01)

network = nf.mix_network(sizes, act_strings, hypers)

# Load the data
training_data, validation_data, test_data = ml.load_data_wrapper()
# Make validation data trainable
val_inputs = [x for x, y in validation_data]
validation_results_vec = [ml.vectorized_result(y) for x, y in validation_data]
validation_data = zip(val_inputs, validation_results_vec)
training_data.extend(validation_data)
runner = nr.NetworkRunner()

#Train the network
test_errors = runner.sgd(network, training_data, 10, 100, test_data)
