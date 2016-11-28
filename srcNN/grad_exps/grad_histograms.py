import matplotlib.pyplot as plt

import srcNN.data.mnist_loader as ml
import srcNN.data.optimal_nets as on
import srcNN.network.network_factory as nf
import srcNN.network.network_runner as nr


def to_histogram(name, report_grad_epochs, reported_dws, reported_dbs):
    l = 1
    for epoch, dw_list, db_list in zip(report_grad_epochs, reported_dws, reported_dbs):
        num_layers = len(dw_list)
        sub = 101 + 10*num_layers
        plt.figure(l)
        for dw, db in zip(dw_list, db_list):
            ax = plt.subplot(sub)
            ax.yaxis.set_visible(False)
            plt.hist(dw.flatten(), bins=50, normed=True, stacked=True)
            sub += 1
        l += 1
    plt.show()

def to_file(name, report_grad_epochs, reported_dws, reported_dbs):
    f = open(name, 'w')
    f.write(str(report_grad_epochs)+'\n')
    f.write(str(reported_dws) + '\n')
    f.write(str(reported_dbs) + '\n')

def histograms_for(network):
    # Load the data
    training_data, validation_data, test_data = ml.load_data_wrapper()
    # Make validation data trainable
    val_inputs = [x for x, y in validation_data]
    validation_results_vec = [ml.vectorized_result(y) for x, y in validation_data]
    validation_data = zip(val_inputs, validation_results_vec)
    training_data.extend(validation_data)
    runner = nr.NetworkRunner()

    report_grad_epochs = [0, 1, 9]
    _, reported_dws, reported_dbs = runner.sgd(network, training_data, 10, 10, test_data, report_grad_epochs)
    to_histogram(network.name, report_grad_epochs, reported_dws, reported_dbs)


#Load optimal parameters for this network
# net_names = ['sig-sig-and-sm', 'sig-or-and-sm', 'sig-and-or-sm']
# for name in net_names:
sizes, act_strings, hypers = on.get_optimal('sig-sig-sig-sm')
network = nf.mix_network(sizes, act_strings, hypers)
histograms_for(network)


