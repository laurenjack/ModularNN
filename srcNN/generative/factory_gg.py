import srcNN.network.network_factory as nf
from gaussian_generator import GaussianGenerator

"""Module responsible for building Gaussian Generators"""

def simple_gg(sizes, acts, hyp_params):
    network = nf.mix_network(sizes, acts, hyp_params)
    d = sizes[0]
    d_out = sizes[-1]
    return GaussianGenerator(network, d, d_out)