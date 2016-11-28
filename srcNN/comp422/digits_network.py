import numpy as np
import srcNN.network.network_runner as nr
import srcNN.network.network_factory as nf
import data_processor as dp
import matplotlib.pyplot as plt

data_sets = ["60", "20", "30", "00", "05"]


runner = nr.NetworkRunner()

# Load the data
ds = '20'
train, test = dp.load_digits("digits/digits"+ds)


def run_network(drop_scheme, n):
    network = nf.mix_network([49, 200, 200, 10], ['relu', 'relu', 'sm'], [0.1] * 3, drop=(drop_scheme, n))
    # network = nf.mix_network([49, 49, 10], ['lin', 'lin'], [0.03, 0.03])
    errors = runner.sgd(network, train, 100, 3000, test)
    err = errors[-1]
    print "Final classification error for " + ds + ": " + str(err)
    # Draw the error graph
    plt.plot(errors)
    return err


schemes = ['drop_sys', 'drop_out', 'drop_connect', 'no drop scheme']
final_errs = []
for s in schemes:
    err = run_network(s, 8)
    final_errs.append(err)
print final_errs
plt.title(ds + " percent noise")
plt.legend(schemes)
plt.show()


#for ds in data_sets:
    # #Load the data
    # train, test = dp.load_digits("digits/digits"+ds)
    # # Construct the network
    # network = nf.mix_network([49, 100, 100, 10], ['relu', 'relu', 'sm'], [0.1, 0.1, 0.1], drop='drop_out')
    # errors = runner.sgd_tracking_error(network, train, 100, 1000, test)
    # #Draw the error graph
    # print "Final classification error for "+ds+": "+str(errors[-1])
    # plt.title(ds+" percent noise")
    # plt.plot(errors)
    # plt.show()




