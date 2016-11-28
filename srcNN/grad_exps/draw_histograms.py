import matplotlib.pyplot as plt


def to_histogram(name, report_grad_epochs, reported_dws, reported_dbs):
    for epoch, dw_list, db_list in zip(report_grad_epochs, reported_dws, reported_dbs):
        l = 0
        for dw, db in dw_list, db_list:
            plt.hist(dw.flatten(), bins=50, facecolor='cyan')
            break
        break
    plt.show()

def draw_hist_for(name):
    f = open(name, 'r')

