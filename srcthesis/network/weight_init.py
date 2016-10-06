import numpy as np
import math
from scipy.stats import norm

def __all_low_enough(w2):
    for w in np.nditer(w2):
        if abs(w) > 0.674:
            return False
    return True

def (w1, upper):
    for w in np.nditer(w1):
        if abs(w) <= upper:
            return False
    return True

def __all_in_range(wi, lower, upper):
    for w in np.nditer(wi):
        if not lower < abs(w) < upper:
            return False;
    return True;


def __balance(w1, w2):

    w1[0][0] = abs(w1[0][0])
    w1[1][0] = -abs(w1[1][0])
    return [w1, w2]

def __balance_124(weights):
    w1 = weights[0]
    w2 = weights[1]
    w3 = weights[2]

    #Layer 1
    w1[0][0] = abs(w1[0][0])
    w1[1][0] = -abs(w1[1][0])
    #Layer 2
    #Must have opposign splits
    w2[0][0][0] = abs(w2[0][0][0])
    w2[0][1][0] = -abs(w2[0][1][0])
    w2[1][0][0] = abs(w2[1][0][0])
    w2[1][1][0] = -abs(w2[1][1][0])

    #Make the size of each weight path 1
    weight_prod_sum = w1[0][0]* w2[0][0][0] * w3[0][0]
    weight_prod_sum += w1[0][0]* w2[0][1][0] * w3[1][0]
    weight_prod_sum += w1[1][0] * w2[1][0][0] * w3[2][0]
    weight_prod_sum += w1[1][0] * w2[1][1][0] * w3[3][0]
    weight_prod_avg = weight_prod_sum/float(4)
    w1 = 1.0 / weight_prod_avg * w1
    w2 = 1.0 / weight_prod_avg * w2
    w3 = 1.0 / weight_prod_avg * w3

    return [w1,w2,w3]



def non_flip_normal_2layer(sizes):
    """A distribution of weights such that flipping will not happen
    (assuming the bias doesn't cause it). The first layer of weights
    must be larger, laying in the outer 50% of the normal
    distribution, whilst the second layer of weights are smaller,
    lying within the middle 50%"""
    s0 = sizes[0]
    s1 = sizes[1]
    s2 = sizes[2]

    # Generate larger first layer weights
    w1 = np.random.randn(s1, s0)
    midpoint = abs(0.674*np.ones((s1, s0)))
    while not __all_high_enough(w1, 0.674):
        new = np.random.randn(s1, s0)
        w1 = np.where(abs(new) > midpoint, new, w1)
    #w1 = 2*w1

    #Generate small second layer weights
    w2 = np.random.randn(s2, s1)
    midpoint = abs(0.674 * np.ones((s2, s1)))
    while not __all_low_enough(w2):
        new = np.random.randn(s2, s1)
        w2 = np.where(abs(new) <= midpoint, new, w2)
    #w2 = w2 * 2

    return [w1, w2]

def non_flip_normal_nlayer(sizes):
    n = len(sizes)-1
    ws = []
    upper = None
    for i in xrange(n-1):
        lower = norm.ppf(0.5 + float(i)/(2*float(n)))
        upper = norm.ppf(0.5 + float(i+1)/(2*float(n)))
        wi = np.random.randn(sizes[-i-1], sizes[-i-2])
        while not __all_in_range(wi, lower, upper):
            new = np.random.randn(sizes[-i-1], sizes[-i-2])
            wi = np.where(np.logical_and(lower <= abs(new), abs(new) <= upper), new, wi)
        #wi = 0.1*wi
        ws.append(wi)

    #Edge case, first layer
    w1 = np.random.randn(sizes[1], sizes[0])
    while not __all_high_enough(w1, upper):
        new = np.random.randn(sizes[1], sizes[0])
        w1 = np.where(abs(new) > upper, new, w1)
    #w1 = 0.1 * w1
    ws.append(w1)
    ws.reverse()
    return ws


def balanced_sign_init(sizes):
    s0 = sizes[0]
    s1 = sizes[1]
    s2 = sizes[2]
    w1 = 0.1 * np.random.randn(s1, s0)
    w2 = 0.1 * np.random.randn(s2, s1)
    return __balance(w1, w2)

def balanced_non_flip(sizes):
    weights = non_flip_normal_2layer(sizes)
    return __balance(weights[0], weights[1])

def balanced_non_flip_124():
    sizes = [1,2,4]
    shapes = [(2,1,1), (2, 2, 1)]
    weights = non_flip_normal_nlayer(sizes)
    weights = __balance_124(weights)



def relu_simple_biases(scalar, sizes):
    biases = [np.ones((y, 1)) for y in sizes[1:-1]]
    for i in xrange(len(biases)):
        biases[i] = scalar*biases[i]
    biases.append(np.zeros(sizes[-1]))
    return biases

def __send_back_extra_weight(low_tot, element_tot):
    to_share = 1.0/low_tot
    num_layers_inv = 1.0/float(len(element_tot))
    split = math.pow(to_share, num_layers_inv)
    new_elem_tot = []
    for elem, tot in element_tot:
        new_elem = elem*split
        new_tot = tot*split
        new_elem_tot.append((new_elem, new_tot))
    return new_elem_tot



def elements_and_totals(weights_out):
    element_tot = []
    last_wi = 0
    for wi in weights_out:
        elem = 1.0/float(wi)
        if wi < last_wi:
            tot = float(wi)/float(last_wi)
            element_tot = __send_back_extra_weight(tot, element_tot)
            elem = 1.0/float(last_wi)
        else:
            tot = 1.0
        element_tot.append((elem, tot))
        last_wi = wi
    return element_tot

def __each_layer_one(tot_to_out, shape):
    exp_mag = 1.0 / float(tot_to_out)
    b = 2 * exp_mag
    a = -b
    return np.random.uniform(a, b, shape)

def each_layer_one_conv(shape):
    tot_to_out = shape[0]*shape[1]*shape[2]
    return __each_layer_one(tot_to_out, shape)

def each_layer_one_fc(shape):
    return __each_layer_one(shape[0], shape)

result = each_layer_one_conv([5, 5, 1, 32])

# n_in = 1000
# variance = 1.0/float(n_in)
# b_less_a = (variance*12)**0.5
# b = b_less_a/2.0
# a = -b
# ws = []
# for i in xrange(10):
#     ws.append(np.random.uniform(a, b, (n_in, n_in)))
# prod = np.identity(n_in)
# for W in ws:
#     prod = W.dot(prod)
# print np.sum(abs(prod), axis=1)/float(n_in)






