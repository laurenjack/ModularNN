import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from srcNN.visual.error_drawer import __plot

"""Responsible for graphing detailed network information over time.
    See also: weight_tracker.py"""

def draw_weight_stats(self, weight_stats):
    """Plot a networks biases, weights and errors over time"""
    plt.figure(2)

    #Biases
    plt.subplot(131)
    self.__plot_sequences("Biases", weight_stats.get_biases_for_plot())

    #Weights
    plt.subplot(132)
    self.__plot_sequences("Weights", weight_stats.get_weights_for_plot())

    #Cost
    plt.subplot(133)
    plt.title("Training Cost")
    plt.plot(weight_stats.get_cost_for_plot())

    plt.show()

def draw_node_stats(self, node_stats):
    # output_arrays = node_stats.obs_outputs.values()[0]
    # output_scalars = []
    # for out in output_arrays:
    #     output_scalars.append(out[0,0])


    # outputs
    plt.figure(1)
    obs_outputs = iter(node_stats.obs_outputs.items())
    dim = self.__output_sub_dimensions()
    rows = dim[0]
    cols = dim[1]
    for row in xrange(1, rows + 1):
        for col in xrange(1, cols + 1):
            pos = (row - 1) * rows + col
            plt.subplot(rows, cols, pos)
            next_obs_output = next(obs_outputs)
            plt.title(next_obs_output[0])
            plt.plot(next_obs_output[1])
    # plt.subplot(311)
    # for base in obs_outputs:
    #     plt.title(base)
    #     plt.plot(obs_outputs[base])


    # weights
    plt.figure(2)
    plt.subplot(211)
    i = 0
    legends = []
    for ws in node_stats.weights_lists:
        plt.title("Weights")
        plt.plot(ws)
        legends.append("$W_" + str(i) + "$")
        i += 1
    plt.legend(legends)

    # biases
    plt.subplot(212)
    plt.title("Bias")
    plt.plot(node_stats.biases)

    # cost
    plt.figure(3)
    obs_costs = iter(node_stats.obs_costs.items())
    dim = self.__output_sub_dimensions()
    rows = dim[0]
    cols = dim[1]
    for row in xrange(1, rows + 1):
        for col in xrange(1, cols + 1):
            pos = (row - 1) * rows + col
            plt.subplot(rows, cols, pos)
            next_obs_cost = next(obs_costs)
            plt.title(next_obs_cost[0])
            plt.plot(next_obs_cost[1])

    plt.show()

def __plot_sequences(self, title, sequences):
    """Plot a list of lists, usually lists of weights over time"""
    prefix = title[0]
    i = 0
    legends = []
    for ws in sequences:
        plt.title(title)
        plt.plot(ws)
        legends.append("$"+prefix+"_" + str(i) + "$")
        print prefix+"_" + str(i) + ": " + str(ws[0])
        i += 1
    plt.legend(legends)

def __output_sub_dimensions(self):
    return (1, 3)


def plot_modes(mode_stats):
    #Take the diagonals of the mode matrix to get each mode
    diags = mode_stats.get_diagonals()
    non_diags = mode_stats.get_max_non_diags()
    __plot(diags, 1)
    __plot(non_diags, 2)
    plt.show()

def __plot(elements, fig_num):
    plt.figure(fig_num)
    axes = plt.gca()
    axes.set_ylim([-100, 700])
    for i, values in elements.items():
        plt.plot(values)