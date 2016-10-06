import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "cmr"

def draw_or():
    x = np.arange(0, 5, 0.01)
    y = 1.0 - np.exp(-x)
    plt.figure(facecolor='white')
    plt.title('Continuous Or Activation Function')
    plt.plot(x, y)
    plt.show()

draw_or()