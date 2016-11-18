def count_linear_regions(network):
    """Takes a neural network composed of ReLU's
    (followed by a linear of softmax output layer)
    and returns the number of linear regions.

    Assumes the network has no biases and an equal
    width at each layer"""
    weights = network.weights
    w = weights[0].shape[0]
    depth = len(weights)
    #Get each row of weights, stored in a bit set (integer)
    # where the bit reflects the sign
    weights = weights_as_bits(weights, w)

    #Get the number of lienar regions from first layer
    act_combos = create_bit_set(w)

    count = 0
    for w_mat in weights:
        count += count_for_layer(act_combos, w_mat, w)

    return count

def count_for_layer(act_combos, w_mat, w):
    layer_count = 0
    for input, freq in act_combos.items():
        for i in xrange(w):
            #For the current output node
            row_bit_set = w_mat[i]
            #Can it exhibit 1 or 2 output states?
            layer_count +=

def states_exhibited(input, row_bit_set):
    """A node can be on or off, given the input.

    - If it can only exhibit off, return 0
    - If it can only exhibit on, return 1
    - If it can exhibit both, return 2"""
    has_pos_acts = row_bit_set & input
    #if not positive weights are on active inputs,
    # then the unit in question can never be active, return 0
    if has_pos_acts == 0:
        return 0
    has_neg_acts = ~row_bit_set & input
    #positive weights on positive inputs but no positive weights
    #on negative inputs, return 1
    if has_neg_acts == 0:
        return 1
    #positive weights on both negative and positive inputs, this
    #node could be on or off
    return 2


def weights_as_bits(real_weights, w):
    """Represent each row of weights as a bit set (integer)
    Where negative weights 1 and positive weights are zero
    """
    weights = []
    for weight_mat in real_weights:
        rows = []
        for i in xrange(w):
            bit_set = 0
            row = weight_mat[i]
            #enter each value in the bit set
            for j in xrange(w):
                #Posiive values are a 1 in the bit set
                if row[j] > 0:
                    bit_set += (2 >> j)
            rows.append(bit_set)
        weights.append(rows)
    return weights





def create_bit_set(w):
    """Create a bit set (using integers) representing all intial linear regions.

    The bit set maps the number of total linear regions attributed to the
    given combintation of activvations. """
    act_combos = {}
    for i in xrange(1 >> w):
        {i: 1}
    return act_combos

