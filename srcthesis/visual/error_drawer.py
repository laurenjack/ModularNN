import matplotlib.pyplot as plt

def draw_error_graph(title, training_error, test_error):
    plt.figure(1)
    plt.title(title)
    plt.plot(test_error)
    plt.plot(training_error)
    plt.legend(["Test Error", "Training Error"])
    plt.ylim([0,1])
    plt.show()

def draw_error_graph(title, legend_train):
    """Take a title, and a list of binary tuples: Where the tuples contain:
        - the name of the network for the the legend
        - the list of training errors for the network"""
    plt.figure(1)
    plt.title(title)
    legends = []
    for legend, train in legend_train:
        plt.plot(train)
        legends.append(legend)
    plt.legend(legends)
    plt.ylim([0, 0.5])
    plt.show()

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