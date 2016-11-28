import sklearn.datasets as ds
import sklearn.decomposition as decomp
import matplotlib.pyplot as plt
import numpy as np
import srcNN.network.network_factory as nf
import srcNN.network.network_runner as nr
import srcNN.visual.weight_stats_drawer
import srcNN.visual.weight_tracker as wt
import srcNN.visual.error_drawer as ed

#Function for standardizing the inputs
def standardize(X):
    x_standard = (X - X.mean(axis=0))/(X.std(axis=0))
    #Could have divided by zero, so set to zero in this case
    return np.nan_to_num(x_standard)

#Function that takes a single output and returns a 1 hot vector
def one_hot(dim, j):
    """Puts a one in the jth position of the vector"""
    e = np.zeros(dim)
    e[j] = 1.0
    return e

#Takes a matrix adnd returns it as a list of vectors
def list_of_vectors(X):
    r = X.shape[0]
    c = X.shape[1]
    X_list = []
    for i in xrange(c):
        X_list.append(X[:,i].reshape(r, 1))
    return X_list

def run_saxe(act_strings, l_rates, use_saxe_weights=True):
    # Load simple digits data from sklearn
    digits = ds.load_digits()
    X = digits.data
    Y_decimal = digits.target
    # Convert outputs to 1 hot encoding
    Y_list = [one_hot((10, 1), digit) for digit in np.nditer(Y_decimal)]
    # Build matrix for outputs
    Y = np.concatenate(Y_list, axis=1)
    # Standardize outputs, so that YXt = Sigma31
    # Y = standardize(Y)
    # #Rescale, so the targets are not too big
    # Y /= 3
    Y_list = [Y[:, i].reshape(10, 1) for i in xrange(Y.shape[1])]

    # Whiten the inputs
    pca = decomp.PCA(whiten=True)
    pca.fit(X)
    X = pca.transform(X)
    X = standardize(X)
    # transpose for NN
    X = X.transpose()

    # Verify X has been whittened correctly
    # XXt = np.dot(X, X.transpose())
    # plt.imshow(XXt.reshape(64, 64), interpolation='nearest', cmap='Greys')
    # plt.show()

    # Covariance matrix (OLS if the inputs were whittened)
    sigma31 = np.dot(Y, X.transpose())
    # #Display the OLS coefficients
    # plt.imshow(sigma31.reshape(10, 64), interpolation='nearest', cmap='Greys')
    # plt.show()

    # Create a nerual network
    sizes = [64, 10, 10]
    network, U, S, Vt = nf.saxe_init(sizes, act_strings, l_rates, sigma31, use_saxe_weights)

    # Transform data
    # Y = U.transpose().dot(Y)
    # Y_list = [Y[:,i].reshape(10, 1) for i in xrange(Y.shape[1])]
    # X = Vt.dot(X)

    # Put Data in NN friendly train and eval forms
    X_list = list_of_vectors(X)
    train = zip(X_list, Y_list)
    eval = zip(X_list, Y_decimal)

    # Train the network
    runner = nr.NetworkRunner()
    mode_stats = wt.ModeStats(network, train, X, U, Vt)
    errors = runner.sgd(network, train, 50, 50, eval, wt=mode_stats)

    # Plot mode strengths
    srcNN.visual.weight_stats_drawer.plot_modes(mode_stats)

run_saxe(['lin', 'lin'], [0.02, 0.02], use_saxe_weights=True)
#run_saxe(['lin', 'lin'], [0.02, 0.02], use_saxe_weights=False)
#run_saxe(['sig', 'sig'], [0.2, 0.2], use_saxe_weights=True)
#run_saxe(['relu', 'sm'], [0.1, 0.1], use_saxe_weights=True)