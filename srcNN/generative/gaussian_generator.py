import numpy as np
import matplotlib.pyplot as plt
import math
import kde_regularizer as kde_reg

class GaussianGenerator:

    def __init__(self, network, d, d_out):
        """:param network - The generator network that takes z's and
        generates an output X.

        :param d - The number of latent variables, i.e. the dimensionality
        of z

        :param d_out - The number of output pixels
        """
        self.network = network
        self.d = d
        self.d_out = d_out

    def update_mini_batch(self, mini_batch, batch_index, z_eta, density_stats, epoch):
        """Move z's around without batch normalization, just regularizing to
        an isotropic Gaussian"""
        Z, X = mini_batch
        m = Z.shape[1]
        #zs = [z for z, x in mini_batch]
        # Train the weights using backProp, and find the gradient w.r.t
        # the weighted sums, d_phis. d_phis is a list of lists of gradients
        # for each input, where the inner list has a gradient vector for
        # each layer. As far as the latent variables are concerned, we are
        # only interested in the first weighted sum vector of each observation
        _, _, d_phis = self.network.update_mini_batch(mini_batch, epoch)
        # Push this back through the activations, and back through
        # the first weights to the input layer
        W0 = self.network.weights[0]
        #dzs = [W0.transpose().dot(d_phi[0]) for d_phi in d_phis]
        dz = W0.transpose().dot(d_phis[0])
        reg = kde_reg.get_grad_for(batch_index, density_stats)
        #regs = np.split(regs, m, axis=1)
        reg_eta = density_stats.reg_eta
        new_Z = Z - z_eta * dz - reg_eta * reg
        #new_batch = [(zx[0] - z_eta * dz -10.0*reg, zx[1]) for zx, dz, reg in zip(mini_batch, dzs, regs)]
        #new_zs = [z for z, x in new_batch]
        density_stats = kde_reg.update_fz(density_stats, new_Z, batch_index)
        #density_stats = kde_reg.update_fz(density_stats, new_zs, batch_index)
        return new_Z, density_stats



    def update_mini_batch_norm(self, mini_batch, batch_index, z_eta, density_stats):
        """Update the generator network using backProp for the given
        mini batch.

        :param mini_batch - A list of tuples (z, x), where z is the latent
        variable in its initial state and x is the image/target we are
        trying to generate

        :param z_eta - The learning rate that is specifically used to update
        the latent variables

        :param ss_z - The sum of squares for the z's, across all instances
        and dimensions

        :return The gradient w.r.t the weights and biases"""
        Z, X = mini_batch
        m = Z.shape[1]
        #Normalise the z's before putting through the network
        Mew, SD, norm_Z = self.normalize(Z, m)
        #Train the weights using backProp, and find the gradient w.r.t
        #the weighted sums, d_phis. d_phis is a list of lists of gradients
        #for each input, where the inner list has a gradient vector for
        #each layer. As far as the latent variables are concerned, we are
        #only interested in the first weighted sum vector of each observation
        _, _, d_phis = self.network.update_mini_batch((norm_Z, X))
        #Push this back through the activations, and back through
        #the first weights to the input layer
        W0 = self.network.weights[0]
        #Must take into account the normalization
        #d_z_mean = float(m - 1)/float(m)
        d_z_norm = float((m-1) ** 2)/float(m ** 2) / SD
        dz = d_z_norm * W0.transpose().dot(d_phis[0])
        #dzs = [d_z_norm*W0.transpose().dot(d_phi[0]) for d_phi in d_phis]
        #Compute reguralization gradients that make it like a Gaussian
        reg = kde_reg.get_grad_for(batch_index, density_stats)
        reg = d_z_norm * reg
        #regs = kde_reg.get_gradients([z for z, x in norm_batch], references, dir)
        #Return the update vectors of latent variables each in a tuple with
        #the corresponding character they create
        #gp_d = n*self.d - 2 gp_d/ss_z
        reg_eta = density_stats.reg_eta
        new_Z = Z - z_eta * dz - reg_eta * reg
        #new_batch = [(zx[0] - z_eta*dz - 3.0 * reg, zx[1]) for zx, dz, reg in zip(mini_batch, dzs, regs)] # 0.0005*zx[0]*(ss_z - 728)
        density_stats = kde_reg.update_fz(density_stats, new_Z, batch_index)
        # Re-normalise to display changed Z in latents
        _, _, new_norm_Z = self.normalize(new_Z, m)
        return new_Z, density_stats, new_norm_Z

    def normalize(self, Z, m):
        vec_shape = (self.d, 1)
        Mew = np.mean(Z, axis=1).reshape(vec_shape)
        SD = (1.0 / float(m) * np.sum((Z - Mew) ** 2.0, axis=1)) ** 0.5 + 0.001
        SD = SD.reshape(vec_shape)
        norm_Z = (Z - Mew) / SD
        return Mew, SD, norm_Z

    def update_mini_batch_mean(self, mini_batch, z_eta, ss_z, mean_z, n):
        """Update the generator network using backProp for the given
            mini batch.

            :param mini_batch - A list of tuples (z, x), where z is the latent
            variable in its initial state and x is the image/target we are
            trying to generate

            :param z_eta - The learning rate that is specifically used to update
            the latent variables

            :param ss_z - The sum of squares for the z's, across all instances
            and dimensions

            :return The gradient w.r.t the weights and biases"""
        m = len(mini_batch)
        # Normalise the z's before putting through the network
        zs = [z for z, x in mini_batch]
        mew = np.mean(zs, axis=0)
        norm_batch = [((z - mew), x) for z, x in mini_batch]
        # Train the weights using backProp, and find the gradient w.r.t
        # the weighted sums, d_phis. d_phis is a list of lists of gradients
        # for each input, where the inner list has a gradient vector for
        # each layer. As far as the latent variables are concerned, we are
        # only interested in the first weighted sum vector of each observation
        _, _, d_phis = self.network.update_mini_batch(norm_batch)
        # Push this back through the activations, and back through
        # the first weights to the input layer
        W0 = self.network.weights[0]
        # Must take into account the normalization
        d_z_mean = float(m - 1)/float(m)
        dzs = [d_z_mean * W0.transpose().dot(d_phi[0]) for d_phi in d_phis]
        # Return the update vectors of latent variables each in a tuple with
        # the corresponding character they create
        gp_d = n*self.d - 2
        new_batch = [(gp_d/ss_z * (zx[0] - z_eta * dz), zx[1]) for zx, dz in
                     zip(mini_batch, dzs)]  # 0.0005*zx[0]*(ss_z - 728)
        # Change the sum of squares, according to the change in the mini_batch
        old_ss_mb, old_batch_mean = self.calc_sum_squares(mini_batch)
        new_ss_mb, new_batch_mean = self.calc_sum_squares(new_batch)
        new_ss_z = ss_z + new_ss_mb - old_ss_mb
        new_mean_z = mean_z + m / n * (new_batch_mean - old_batch_mean)
        return new_batch, new_ss_z, new_mean_z

    def get_latent_d(self):
        """Get the number of dimensions for the latent variables, i.e.
        the number of latent variables"""
        return self.d

    def calc_sum_squares(self, data_set):
        """For a list of latent variables calculating the sum of squares,
        iterating over each element and each dimension"""
        zs = [z for z, x in data_set]
        mean = np.mean(zs)
        return sum([np.sum((z - 0.0) ** 2) for z, X in data_set]), mean

    def generate_as_image(self, z):
        """Returns an image matrix X than can displayed directly using numpy's
        imshow"""
        X = self.network.feedforward(z)
        width = int(float(self.d_out) ** 0.5 + 0.5)
        return X.reshape(width, width)


    def generate_image(self):
        """Generate a random image from the network"""
        z = np.random.randn(self.d, 1)
        X = self.network.feedforward(z)
        width = int(math.sqrt(self.d_out) + 0.5)
        image = X.reshape(width, width)
        plt.imshow(image, interpolation='nearest', cmap='Greys')
        plt.show()

