import srcthesis.network.network_factory as nf
import srcthesis.network.network_runner as nr
import srcthesis.run.mnist_loader as ml
import matplotlib.pyplot as plt
import srcthesis.data.optimal_nets as on

def to_histogram(name, report_grad_epochs, reported_dws, reported_dbs):
    for epoch, dw_list, db_list in zip(report_grad_epochs, reported_dws, reported_dbs):
        l = 0
        for dw, db in dw_list, db_list:
            plt.hist(dw.flatten(), bins=50, facecolor='cyan')
            break
        break
    plt.show()

def histograms_for(network):
    # Load the data
    training_data, validation_data, test_data = ml.load_data_wrapper()
    # Make validation data trainable
    val_inputs = [x for x, y in validation_data]
    validation_results_vec = [ml.vectorized_result(y) for x, y in validation_data]
    validation_data = zip(val_inputs, validation_results_vec)
    training_data.extend(validation_data)
    runner = nr.NetworkRunner()

    report_grad_epochs = [0, 1, 4]
    _, reported_dws, reported_dbs = runner.sgd_tracking_error(network, training_data, 10, 5, test_data, report_grad_epochs)
    to_histogram(network.name, report_grad_epochs, reported_dws, reported_dbs)


#Load optimal parameters for this network
sizes, act_strings, hypers = on.get_optimal('sig-sm')
network = nf.mix_network(sizes, act_strings, hypers)
histograms_for(network)

