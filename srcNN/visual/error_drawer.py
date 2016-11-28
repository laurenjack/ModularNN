import matplotlib.pyplot as plt

"""Responsible for reporting the classification errors over time."""

def draw_error_graph(title, *legend_errors):
    """Take a title, and one or more binary tuples: Where the tuples contain:
        - the name of the network for the the legend
        - the list of errors for the network"""
    plt.figure(1)
    plt.title(title)
    legends = []
    for legend, train in legend_errors:
        plt.plot(train)
        legends.append(legend)
    plt.legend(legends)
    plt.show()


