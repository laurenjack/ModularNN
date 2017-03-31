import numpy as np
import latent_animator as la
import kde_regularizer as kde_reg
from kde_regularizer import KdeReg
import random
import math
import matplotlib.pyplot as plt

K = 1.0 / math.sqrt(2.0 * math.pi)

def test_whole_data_set():
    """Visually test if the kde Regularizer minimizes the KL Divergence
    between an isotropic Gaussian and the Kernel Density Estimate in
    the z space"""
    d = 2
    r = 30
    m = 400
    kde = KdeReg(d)
    #Sample 5 lots of 15 different clusters from a uniform distribution
    n = 400
    zs_and_ts = []
    # gauss = [(np.random.randn(d, 1), 1) for i in xrange(n)]
    # zs_and_ts.extend(gauss)
    __sample_n(d, n, 1.0, 1.5, zs_and_ts, 1)
    #__sample_n(d, n, -0.3, 0.3, zs_and_ts, 1)
    # __sample_n(n, -0.5, 0.2, zs_and_ts, 1)
    # __sample_n(n, -0.6, 0.5, zs_and_ts, 3)
    # __sample_n(d, n, -2.0, 1.5, zs_and_ts, 1)

    h = (4.0 / (3.0 * n)) ** 0.2

    def p(z):
        """Isotropic Gaussian"""
        p_exp = -0.5 * z[1, 0] ** 2.0
        return K * np.exp(p_exp)

    def f(z, latents):
        m = len(zs)
        return K / float(m) / h * np.sum([np.exp(-0.5 * ((z[1, 0] - zi[1, 0]) / h) ** 2.0) for zi in latents])

    #Update the zs
    epochs = 120
    allLatents = []
    for i in xrange(epochs):
        random.shuffle(zs_and_ts)
        zs, targets = zip(*zs_and_ts)
        batches = [zs[k:k + m] for k in xrange(0, n, m)]
        new_zs = []
        # Compute f(z)
        density_stats = kde_reg.f(batches, d, r)
        b_index = 0
        for batch in batches:
            zs = [z for z, t in batch]
            grads = np.split(kde_reg.get_grad_for(0, density_stats), m, axis=1)
            new_batch = [z - 0.5 * g.reshape(d, 1) for z, g in zip(zs, grads)]
            #density_stats = kde_reg.update_fz(density_stats, batches, b_index)
            new_zs.extend(new_batch)
            b_index += 1
            # refs = [np.random.randn(d, 1) for k in xrange(r)]
            # dir = np.random.randn(d, 1)
            # dir = dir/ np.linalg.norm(dir)
            # zs = kde.update_gradients(zs, 10.0, refs, dir)
            # zs_and_ts = [(z, zt[1]) for z, zt, in zip(zs, zs_and_ts)]
            # with_density = [(np.array([[z[1, 0]], [f(z, zs)]]), 1) for z, t in zs_and_ts]
            # gaussians = [(np.array([[z[1, 0]], [p(z)]]), 2) for z, t in zs_and_ts]
            # with_density.extend(gaussians)
            # allLatents.append(with_density)
        zs_and_ts = zip(new_zs, targets)
        allLatents.append(zs_and_ts)

    la.animate_2D(allLatents)
    refs = np.random.randn(d, r)
    test_2D_density(refs, new_zs)



    # # Generate n z's
    # zs = [np.random.normal() for i in xrange(n)]
    # # Calculate densities for each function
    # gauss = [p(z) for z in zs]
    # l_no_targets = [z for z, t in zs_and_ts]
    # kde = [f(z, l_no_targets) for z in zs]
    # kde_to_selves = [f(z, l_no_targets) for z in l_no_targets]
    #
    # # Plot both sets of densities
    # plt.scatter(zs, gauss, color='r')
    # plt.scatter(zs, kde, color='b')
    # plt.scatter(l_no_targets, kde_to_selves, color='g')
    # plt.show()

def __sample_n(d, n, a, b, zs_and_ts, t):
    sample = [(np.random.uniform(a, b, (d, 1)), t) for i in xrange(n)]
    zs_and_ts.extend(sample)


def p_multi(refs):
    return K * np.exp(-0.5 * np.sum(refs ** 2.0, axis=0))

def f_multi(refs, zi_mat, r, d, m):
    """Multivariate kernel density estimate"""
    d4 = 1.0 / float(d+4)
    H = ((4.0 / float(d+2)) ** d4 * m ** -d4) ** 2.0
    z_tensor = np.repeat(refs.transpose().reshape(r, d, 1), m, axis=2)
    z_less_zi = z_tensor - zi_mat.reshape(1, d, m)
    distances_sq = np.sum(z_less_zi ** 2.0 / H, axis=1)
    exps = np.exp(-0.5 * distances_sq)
    return K / float(m) / (H ** float(d)) ** 0.5 * np.sum(exps, axis=1)

def test_2D_density(refs, zs):
    """Test the 2D KDE against a set of references, using the optimal H"""
    zi_mat = np.concatenate(zs, axis=1)
    d, r = refs.shape
    d, m = zi_mat.shape
    p = p_multi(refs)
    f = f_multi(refs, zi_mat, r, d, m)
    for i in xrange(r):
        print 'f: '+str(f(i))+' p: '+str(p(i))
        print '\n'



test_whole_data_set()