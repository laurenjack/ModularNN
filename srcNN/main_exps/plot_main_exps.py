from os import listdir
import matplotlib.pyplot as plt
import numpy as np

colours = ['g', 'b', 'tomato', 'aqua']

def averages_for(f):
    epoch_avs = np.zeros(60)
    for i in xrange(15):
        for ep in xrange(60):
            class_err = float(f.readline().rstrip())
            epoch_avs[ep]+= class_err
        f.readline() #toss
    return 1.0/15.0*epoch_avs

def plot_for(dir, title):
    one_to_60 = range(1,61)
    legends = []
    plt.figure(facecolor='white')
    for name, col in zip(listdir(dir), colours):
        f = file(dir+'/'+name)
        epoch_avs = averages_for(f)
        print name+': '+str(epoch_avs[-1])
        plt.plot(one_to_60, epoch_avs, color=col)
        legends.append(name)
    plt.title(title)
    plt.legend(legends)
    plt.show()


plot_for('two_layer', 'Two Layer Networks')
#plot_for('three_layer', 'Three Layer Networks')
#plot_for('four_layer', 'Four Layer Networks')
#plot_for('four_layer_two', 'Four Layers, with Two Logical Layers')
