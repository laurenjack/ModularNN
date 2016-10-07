import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "cmr"

def __draw_act(title, f):
    x = np.arange(0, 5, 0.01)
    y = f(x)
    plt.figure(facecolor='white')
    plt.title(title)
    plt.plot(x, y)
    plt.show()

def draw_or():
    x = np.arange(0, 5, 0.01)
    y = 1.0 - np.exp(-x)
    plt.figure(facecolor='white')
    plt.title('Continuous Or Activation Function')
    plt.plot(x, y)
    plt.show()

def draw_and():
    f = lambda x : np.exp(-x)
    __draw_act('Continuous And Activation Function', f)

draw_and()