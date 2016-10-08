import srcthesis.network.network_factory as nf
import srcthesis.network.network_runner as nr
import srcthesis.run.mnist_loader as ml
import srcthesis.data.optimal_nets as on

def load_file(act_strings, sizes):
    #Prep the name
    name = ''
    for a in act_strings:
        name += a+'-'
    for s in sizes[:-1]:
        name += str(s)+'-'
    name+= str(sizes[-1])

    #Create the file
    return open(name, 'w')

def to_file(f, test_errors):
    for err in test_errors:
        f.write(str(err)+'\n')
    f.write('\n')

def run_30_times(name):
    # Load network hyper-parameters
    sizes, act_strings, hypers = on.get_optimal('sig-sm')
    network = nf.mix_network(sizes, act_strings, hypers)

    # Load the data
    training_data, validation_data, test_data = ml.load_data_wrapper()
    # Make validation data trainable
    val_inputs = [x for x, y in validation_data]
    validation_results_vec = [ml.vectorized_result(y) for x, y in validation_data]
    validation_data = zip(val_inputs, validation_results_vec)
    training_data.extend(validation_data)
    runner = nr.NetworkRunner()

    # File to store results
    f = load_file(act_strings, sizes)

    for i in xrange(15):
        network = nf.mix_network(sizes, act_strings, hypers)
        test_errors = runner.sgd_tracking_error(network, training_data, 10, 60, test_data)
        to_file(f, test_errors)

