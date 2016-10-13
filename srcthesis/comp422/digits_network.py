import numpy as np
import srcthesis.network.network_runner as nr
import srcthesis.network.network_factory as nf
import data_processor as dp
import matplotlib.pyplot as plt

data_sets = ["20", "30", "60", "00", "05"]


runner = nr.NetworkRunner()

for ds in data_sets:
    #Load the data
    train, test = dp.load_digits("digits/digits"+ds)
    # Construct the network
    network = nf.mix_network([49, 100, 100, 10], ['relu', 'relu', 'sm'], [0.1, 0.1, 0.1], dropout=True)
    errors = runner.sgd_tracking_error(network, train, 100, 1000, test)
    #Draw the error graph
    print "Final classification error for "+ds+": "+str(errors[-1])
    plt.title(ds+" percent noise")
    plt.plot(errors)
    plt.show()




