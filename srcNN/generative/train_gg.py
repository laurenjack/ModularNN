import random
import numpy as np
import kde_regularizer as kde_reg
import latent_animator as la

"""Generative model that begins with a z drawn from an isotropic Gaussian
 for each data point. These z's correspond to the latent values for
  each X. Of course, sinch they are random such an arrangement is useless,
  but each point is slid along the Gaussian (while the generator is
  trained), producing a distribution of z's that can be used to generate
  X's"""

def train(gg, xs, targets, mb_size, epochs, z_eta, reg_eta, is_mean=True):
    """Train the generative model gg, to generate X's using the data set
    x_list"""
    n = len(xs)
    d = gg.get_latent_d()
    #For each data point X, randomly initialise its corresponding latent
    # variable z according to a sample from an isotropic Gaussian
    zs = [np.random.randn(d, 1) for X in xs]
    data_and_t = zip(zs, xs, targets)
    #Calculate initial some of squares for Gaussian reguralisation
    #ss_z, z_mean = gg.calc_sum_squares(data_set)
    #track the latent variables over time
    all_latents = []
    for i in xrange(epochs):
        if (i + 1) % 60 == 0:
            z_eta /= 4.0
        print "Z epoch: " + str(i)
        random.shuffle(data_and_t)
        zs, xs, targets = zip(*data_and_t)
        new_zs = []
        #new_norm_zs = []
        # references = [np.random.randn(d, 1) for i in xrange(40)]
        # dir = np.random.randn(d, 1)
        # dir = dir / np.linalg.norm(dir)
        #Create the mini batches
        #batches = [data_set[k:k + mb_size] for k in xrange(0, n, mb_size)]
        batches = [(np.concatenate(zs[k:k + mb_size], axis=1), np.concatenate(xs[k:k + mb_size], axis=1)) for k in xrange(0, n, mb_size)]
        #Compute data used for KDE regularization
        Zs, Xs = zip(*batches)
        density_stats = kde_reg.f(Zs, d, 100, n, reg_eta)
        # norm_zs = [gg.normalize(Z, Z.shape[1])[2] for Z in Zs]
        #density_stats = kde_reg.f(norm_zs, d, 100, n, reg_eta)
        batch_index = 0
        for batch in batches:
            m = batch[0].shape[1]
            #Update both the network and the latent z's, the network is
            #updated internally, while the newly updated z's are returned
            #to new_batch. These z's are still in a tuple with their
            #original X
            new_Z, density_stats = gg.update_mini_batch(batch, batch_index, z_eta, density_stats, i)
            # if is_mean:
            #     new_batch, ss_z, z_mean = gg.update_mini_batch_mean(batch, z_eta, ss_z, z_mean, n, references, dir)
            # else:
            #     new_batch, ss_z, z_mean = gg.update_mini_batch_norm(batch, z_eta, ss_z, z_mean, n, references, dir)
            new_zs.extend(np.split(new_Z, m, axis=1))
            #new_norm_zs.extend(np.split(new_norm_Z, m, axis=1))
            batch_index += 1
        #print "SS_z: "+ str(ss_z)
        data_and_t = zip(new_zs, xs, targets)
        #norm_z_and_t = zip(new_norm_zs, xs, targets)

        # zs = [zx[0] for zx, t in data_and_t]
        # Z = np.concatenate(zs, axis=1)
        # cov = np.cov(Z)
        # cov_1 = np.linalg.inv(cov)
        # if is_mean:
        #     sd = 1.0
        # else:
        #     sd = (1.0 / float(n) * sum([(zx[0] - z_mean) ** 2 for zx, t in data_and_t])) ** 0.5
        # all_latents.append([((zx[0] - z_mean)/sd, t) for zx, t in data_and_t]) #/sd
        all_latents.append([(z, t) for z, x, t in data_and_t])
        #all_latents.append([(z, t) for z, x, t in norm_z_and_t])

    #Train network without latents for a bit
    data_set = [(z, x) for z, x, t in data_and_t]
    network = gg.network
    for i in xrange(100):
        random.shuffle(data_set)
        zs, xs = zip(*data_set)
        batches = [(np.concatenate(zs[k:k + mb_size], axis=1), np.concatenate(xs[k:k + mb_size], axis=1)) for k in xrange(0, n, mb_size)]
        print str(i)+": "+str(network.cost_data_set(batches))
        for batch in batches:
            network.update_mini_batch(batch, 21)
    #Return latent variables
    return all_latents

