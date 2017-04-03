import numpy as np
import latent_animator as la
import kde_regularizer as kde_reg
from kde_regularizer import KdeReg
import random
import math
import matplotlib.pyplot as plt
from datetime import datetime

K = 1.0 / math.sqrt(2.0 * math.pi)

def test_whole_data_set():
    """Visually test if the kde Regularizer minimizes the KL Divergence
    between an isotropic Gaussian and the Kernel Density Estimate in
    the z space"""
    d = 20
    r = 100
    m = 50
    kde = KdeReg(d)
    #Sample 5 lots of 15 different clusters from a uniform distribution
    n = 100000
    zs_and_ts = []
    # gauss = [(np.random.randn(d, 1), 1) for i in xrange(n)]
    # zs_and_ts.extend(gauss)
    __sample_n(d, n, 1.0, 1.5, zs_and_ts, 1)
    #__sample_n(d, n, -0.3, 0.3, zs_and_ts, 1)
    #equi_dist = [(np.array([[0],[1]]), 1), (np.array([[-1],[-0.5]]), 1) / 1.25 ** 0.5, np.array([[1],[-0.5]]) / 1.25 /0.5]
    # __sample_n(n, -0.5, 0.2, zs_and_ts, 1)
    # __sample_n(n, -0.6, 0.5, zs_and_ts, 3)
    # __sample_n(d, n, -2.0, 1.5, zs_and_ts, 1)

    h = (4.0 / (3.0 * n)) ** 0.2

    dir = np.random.randn(d, 1)
    dir = dir/ np.linalg.norm(dir)
    #print dir

    def p(z):
        """Isotropic Gaussian"""
        p_exp = -0.5 * dir.transpose().dot(z) ** 2.0
        return K * np.exp(p_exp)

    def f(z, latents):
        m = len(zs)
        return K / float(m) / h * np.sum([np.exp(-0.5 * (dir.transpose().dot(z - zi) / h) ** 2.0) for zi in latents])

    #Update the zs
    epochs = 60
    allLatents = []
    for i in xrange(epochs):
        print "Epoch:" + str(i)
        random.shuffle(zs_and_ts)
        zs, targets = zip(*zs_and_ts)
        start = datetime.now()
        batches = [np.concatenate(zs[k:k + m], axis=1) for k in xrange(0, n, m)]
        new_zs = []
        # Compute f(z)
        reg_eta = 0.25
        density_stats = kde_reg.f(batches, d, r, n, reg_eta)
        end = datetime.now();
        print (start - end).total_seconds()
        b_index = 0
        for batch in batches:
            m = batch.shape[1]
            grads = kde_reg.get_grad_for(b_index, density_stats)
            #grads = np.array_split(, m, axis=1)
            #new_batch = [z - 20.0 * g.reshape(d, 1) for z, g in zip(batch, grads)]
            new_Z = batch - reg_eta * grads
            density_stats = kde_reg.update_fz(density_stats, new_Z, b_index)
            new_zs.extend(np.split(new_Z, m, axis=1))
            b_index += 1
        zs_and_ts = zip(new_zs, targets)
        # with_density = [(np.array([[dir.transpose().dot(z)], [f(z, zs)]]), 1) for z, t in zs_and_ts]
        # gaussians = [(np.array([[dir.transpose().dot(z)], [p(z)]]), 2) for z, t in zs_and_ts]
        # with_density.extend(gaussians)
        # allLatents.append(with_density)
        allLatents.append(zs_and_ts)

    # for i in xrange(m):
    #     print 'f: '+str(f(new_zs[i], new_zs))+' p: '+str(p(new_zs[i]))
    #     print '\n'
    la.animate_2D(allLatents)
    # refs = np.random.randn(d, r)
    # test_2D_density(refs, new_zs)



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


def p_multi(refs, d):
    return K ** float(d) * np.exp(-0.5 * np.sum(refs ** 2.0, axis=0))

def f_multi(refs, zi_mat, r, d, m):
    """Multivariate kernel density estimate"""
    d4 = 1.0 / float(d+4)
    H = ((4.0 / float(d+2)) ** d4 * (float(m) ** -d4)) ** 2.0
    HC = (H ** float(d))
    z_tensor = np.repeat(refs.transpose().reshape(r, d, 1), m, axis=2)
    z_less_zi = z_tensor - zi_mat.reshape(1, d, m)
    distances_sq = np.sum(z_less_zi ** 2.0 / HC, axis=1)
    exps = np.exp(-0.5 * distances_sq)
    return K ** float(d) / float(m) / HC ** 0.5 * np.sum(exps, axis=1)

def test_2D_density(refs, zs):
    """Test the 2D KDE against a set of references, using the optimal H"""
    zi_mat = np.concatenate(zs, axis=1)
    d, r = refs.shape
    d, m = zi_mat.shape
    p = p_multi(zi_mat, d)
    f = f_multi(zi_mat, zi_mat, m, d, m)
    plt.scatter(p, f)
    x = np.arange(0, 0.5, 0.01);
    plt.plot(x, x)
    plt.show()
    # for i in xrange(m):
    #     print 'f: '+str(f[i])+' p: '+str(p[i])
    #     print '\n'

# m = 2000
# cs = np.arange(-10.0, 10.0, 20.0/float(m)).reshape(1, m);
# refs = cs * np.array([[1.0], [0.0]])
# fz = f_multi(refs, refs, m, 2, m)
# pz = p_multi(refs, 2)
# print np.sum(pz) * 20.0/float(m)
# print np.sum(fz) * 20.0/float(m)

test_whole_data_set()