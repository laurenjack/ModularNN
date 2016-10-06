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
    """Take a title, and a list of binrary tuples: Where the tuples contain:
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