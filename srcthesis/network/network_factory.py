import numpy as np
from network import Network
from activations import *
from optimizers import *
from dropout import *


def mix_network(sizes, acts, hyp_params, dropout=False, reg=False):
    #Construct the weights and biases
    weights, biases = [], []
    for act, hp, s0, s1 in zip(acts, hyp_params, sizes[:-1], sizes[1:]):
        w, b = __construct_w_b(act, s0, s1, hp)
        weights.append(w)
        biases.append(b)

    #Construct each activation
    activations = []
    for act, hp, width in zip(acts, hyp_params, sizes[1:]):
        activations.append(__construct_act(act, hp, width, reg))

    #Create the name
    name = __create_name(acts)

    if dropout:
        drop_scheme = DropOut(sizes)
    else:
        drop_scheme = DropNull()

    return Network(weights, biases, activations, name=name, drop_scheme=drop_scheme)

def relu_with_linear_final(sizes, eta, weights=None, biases=None):
    """Construct a relu with a linear output layer, useful for regression problems"""
    if (biases == None):
        biases = [np.ones((y, 1)) for y in sizes[1:]]

    if (weights == None):
        weights = [0.1*(np.random.randn(y, x))
                        for x, y in zip(sizes[:-1], sizes[1:])]

    activations = __construct_activations(sizes[1:-1], eta)
    return Network(weights, biases, activations)

def linear_network(sizes):
    biases = [0.1 * np.random.randn(y, 1) for y in sizes[1:]]
    weights = [0.1 * np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]
    num_layers = len(sizes) - 1
    activations = [Linear() for i in xrange(num_layers)]
    return Network(weights, biases, activations)


def __construct_activations(relu_widths, eta):
    sgd = Sgd(eta)
    activations = []
    for width in relu_widths:
        activations.append(Relu(width, sgd))
    activations.append(Linear(sgd))
    return activations

# def pure_or(sizes):
#     weights = [__postive_normal(y, x)
#                for x, y in zip(sizes[:-1], sizes[1:])]
#     biases = [__postive_normal(y, 1)
#               for y in sizes[1:]]
#     num_layers = len(sizes)
#     activations = [NoisyOr() for i in xrange(num_layers)]
#     return Network(weights, biases, activations, keep_positive_sgd)

def __positive_normal(c, s1, s0):
    return c*abs(np.random.randn(s1, s0))


def __construct_w_b(act_string, s0, s1, hyp_params):
    xc = 1.0 / float(s0) ** 0.5
    if act_string == 'sig':
        weights = xc * np.random.randn(s1, s0)
        biases = xc* np.random.randn(s1, 1)
    elif act_string == 'or':
        w_scale = xc * hyp_params[1]
        weights = __positive_normal(w_scale, s1, s0)
        biases = __positive_normal(w_scale, s1, 1)
    elif act_string == 'and':
        w_scale = xc * hyp_params[1]
        weights = __positive_normal(w_scale, s1, s0)
        biases = __positive_normal(w_scale, s1, 1)
    elif act_string == 'sm':
        weights = xc * np.random.randn(s1, s0)
        biases = xc * np.random.randn(s1, 1)
    elif act_string == 'relu':
        weights = xc* np.random.randn(s1, s0)
        biases = 0.1 * np.random.randn(s1, 1)
    else:
        raise NotImplementedError('Factory does not handle the constuction of the activation: ' + act_string)
    return weights, biases


def __construct_act(act_string, hp, width, reg):

    if act_string == 'sig':
        if reg:
            sgd = Sgd_Regularisation(hp)
        else:
            sgd = Sgd(hp)
        return Sigmoid(sgd)
    if act_string == 'or':
        eta = hp[0]
        if reg:
            pos_sgd = KeepPositiveRegSgd(eta)
        else:
            pos_sgd = KeepPositiveSgd(eta)
        return NoisyOr(pos_sgd)
    if act_string == 'and':
        eta = hp[0]
        if reg:
            pos_sgd = KeepPositiveRegSgd(eta)
        else:
            pos_sgd = KeepPositiveSgd(eta)
        return NoisyAnd(pos_sgd)
    if act_string == 'sm':
        sgd = None
        if reg:
            sgd = Sgd_Regularisation(hp)
        else:
            sgd = Sgd(hp)
        return Softmax(sgd)
    if act_string == 'relu':
        if reg:
            sgd = Sgd_Regularisation(hp)
        else:
            sgd = Sgd(hp)
        return Relu(width, sgd)
    raise NotImplementedError('Factory does not handle the constuction of the activation: '+act_string)

def __create_name(act_strings):
    name = ''
    for act in act_strings[:-1]:
        name+=act+'-'
    return name+act_strings[-1]

