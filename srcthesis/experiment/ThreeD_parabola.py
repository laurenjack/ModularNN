import numpy as np
import random

"""Data generation"""
def __make_parabola(c):
    x = np.random.uniform(-2, 2, (2, 1))
    #x[1][0] = 2*x[1][0]
    y = -0.2*np.dot(x.transpose(), x) + c
    return x, y

def make_data_set(n, c):
    ds = []
    for i in xrange(n):
        coord = __make_parabola(c)
        ds.append(coord)
    return ds

def make_bulls_eye_plain(n, dim):
    return make_bulls_eye(n, dim, 1)

def make_bulls_eye(n, dim, inner):
    ds = []
    bull = []
    outer = []
    bull_n = n/2
    outer_n = n - bull_n
    count = 0

    while count < bull_n:
        x = np.random.uniform(-inner, inner, (dim, 1))
        if(np.linalg.norm(x)<inner):
            y = np.array([[1]])
            bull.append((x, y))
            count = count + 1

    outer_start = inner+1
    outer_end = outer_start+1
    count = 0
    while(count < outer_n):
        x = np.random.uniform(-outer_end, outer_end, (dim, 1))
        if (np.linalg.norm(x) > outer_start and np.linalg.norm(x) < outer_end):
            y = np.array([[-1]])
            outer.append((x, y))
            count = count + 1

    ds.extend(bull)
    ds.extend(outer)
    random.shuffle(ds)
    return ds

def make_variable_bulls_eye(n, dim):
    a_in = []
    b_in = []
    a_out_left = []
    a_out_right = []
    b_out_left = []
    b_out_right = []
    for d in xrange(dim):
        mew = np.random.uniform(-2, 2)
        w = np.random.uniform(0.25, 1)
        a_in.append(mew - w)
        b_in.append(mew + w)
        a_out_left.append(a_in[d] - 1)
        a_out_right.append(b_in[d] + 1)
        b_out_left.append(a_out_left[d] - 2*w)
        b_out_right.append(a_out_right[d] + 2*w)

    ds = []
    n = n/2
    for i in xrange(n):
        xi_in = np.zeros((dim, 1))
        xi_out = np.zeros((dim, 1))
        for d in xrange(dim):
            xi_in[d][0] = np.random.uniform(a_in[d], b_in[d])
            if np.random.uniform(0, 1)<0.5:
                xi_out[d][0] = np.random.uniform(a_out_left[d], b_out_left[d])
            else:
                xi_out[d][0] = np.random.uniform(a_out_right[d], b_out_right[d])
        y_in = np.array([[1]])
        y_out = np.array([[-1]])
        ds.append((xi_in, y_in))
        ds.append((xi_out, y_out))
    random.shuffle(ds)
    return ds




def lop_sided_zig_zag():
    xs = [0.9, 0.6, 0.3, -0.5]
    ds = []
    y = np.array([[1]])
    for x in xs:
        x = 2*np.array([[x]])
        ds.append((x, y))
        y = -1*y
    return ds

def concave_uniform():
    """Generate three points, a uniform distance apart.
    Assign the outer two class -1 and the middle point class 1"""
    xs = []
    for i in xrange(3):
        x = np.random.uniform(-2, 2, (1, 1))
        for j in xrange(10):
         xs.append(x)
    xs.sort()
    ds = []
    for i in xrange(10):
        ds.append((xs[i], np.array([[-1]])))
    for j in xrange(10, 20):
        ds.append((xs[j], np.array([[1]])))
    for k in xrange(20, 30):
        ds.append((xs[k], np.array([[-1]])))


    return ds




