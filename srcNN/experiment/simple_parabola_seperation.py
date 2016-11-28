import numpy as np
import random

def parabola_points(n, c, class_form):
    data = []
    for i in xrange(n):
        x1 = np.random.normal()
        x2 = x1**2 + c
        x = np.array([[x1],[x2]])
        data.append((x, class_form(c)))
    return data

def class_vec(c):
    ys = []
    fst = 0
    scnd = 1
    if (c == 1):
        fst = 1
        scnd = 0
    return np.array([[fst],[scnd]])

def class_scalar(c):
    return c;

def make_data(n):
    c1_data = parabola_points(n, 1, class_vec)
    c0_data = parabola_points(n, 0, class_vec)
    all_data = []
    all_data.extend(c1_data)
    all_data.extend(c0_data)
    random.shuffle(all_data)
    return all_data

d = make_data(5)
print d


