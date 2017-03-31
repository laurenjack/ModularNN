import math
import numpy as np
from density_stats import DensityStats

"""Responsible for regularizing the distribution of latents to fit an isotropic
Gaussian, using KL divergence on a mini batch"""



class KdeReg:

    def __init__(self, d):
        self.d = d
        self.K = 1.0 / math.sqrt(2.0 * math.pi)

    def update_gradients(self, mini_batch, eta, references, dir):
        """Update the zs purely according to the KL divergence of their KDE from an isotropic Gaussian"""
        grads = self.get_gradients(mini_batch, references, dir)
        return [z - eta * grad for z, grad in zip(mini_batch, grads)]

    def get_gradients(self, mini_batch, references, dir):
        """Return a list of gradients, corresponding to each z in mini_batch.
        These gradients push the mini batches distribution to be an isotropic Gaussian"""
        m = len(mini_batch)
        # references = [np.random.randn(self.d, 1) for i in xrange(m)]
        # dir = np.random.randn(self.d, 1)
        # dir = dir / np.linalg.norm(dir)
        grad_sum = [0.0] * m
        for z in references:
            batch_grads = self.__get_grads_for_batch(z, mini_batch, dir)
            for i in xrange(m):
                grad_sum[i] += batch_grads[i]
        return grad_sum

    # def __get_grad(self, z, zs):
    #     #Caluclate  z's density on the isotropic Gaussian p(z), and p'(z)
    #     pz = self.__p(z)
    #     dpz = self.__dp(z, pz)
    #     p_exponent = self.__p_exponent(z)
    #     #Calculate the optimal bandwith for Gaussian distributed data
    #     m = len(zs)
    #     h = (4.0 / (3.0 * m)) ** 0.2
    #     #Calculate f(z), the kernel density estimator and f'(z)
    #     kernel_exps = self.evaluate_kernel_exps(z, zs, h)
    #     fz = self.__f(z, kernel_exps, h)
    #     dfz = self.__df(z, zs, kernel_exps, h)
    #     #Combine all partials to calculate the gradient of the KL Divergence for the point z
    #     grad = dpz * (np.log(self.K) + p_exponent - np.log(fz)) - (z + dfz/fz) * pz
    #     return grad

    def __get_grads_for_batch(self, z, mini_batch, dir):
        # Caluclate  z's density on the isotropic Gaussian
        # Calculate the optimal bandwith for Gaussian distributed data
        m = len(mini_batch)
        h = (4.0 / (3.0 * m)) ** 0.2
        #d4_root = 1.0 / float(self.d + 4)
        #H = ((4.0 /(self.d + 2.0)) ** d4_root / m ** d4_root) ** 2.0
        #Calculate f(z), the kernel density estimator and df(z)/dzi for each zi in the mini batch
        pz = self.__p(z, dir)
        kernel_exps = self.evaluate_kernel_exps(z, mini_batch, h, dir)
        fz = self.__f(z, kernel_exps, h)
        dfzis = self.__df(z, mini_batch, kernel_exps, h, dir)
        return [df * (fz - pz) for df in dfzis]

    def evaluate_kernel_exps(self, z, mini_batch, h, dir):
        """Evaluation of the kernels exponentials for a given point z

        We do this evaluation seperately because it is used in the computation of f(x) and f'(x)"""
        return [self.kernel_exp(z, zi, h, dir) for zi in mini_batch]

    def kernel_exp(self, z, zi, h, dir):
        # a = np.linalg.norm(z-zi)
        # b = abs(dir.transpose().dot(z - zi))
        # scale = (a+b)/math.sqrt(a ** 2.0 +b ** 2.0)
        diff = (dir.transpose().dot(z - zi) * dir) /h #* scale
        k_exp = np.exp(self.__p_exponent(diff))
        return k_exp

    def __f(self, z, kernel_exps, h):
        """Evaluation of the kernel function"""
        m = len(kernel_exps)
        kde = self.K /float(m)/h * np.sum(kernel_exps)
        return kde

    def __df(self, z, batch, kernel_exps, h, dir):
        """Derivative of the kernel function"""
        m = len(kernel_exps)
        return [self.K/float(m)/h * dir.transpose().dot(dir) * dir.transpose().dot(z - zi) * dir / h ** 2.0 * 1.0 / (1.0 - np.log(k + 0.000000001)) for zi, k in zip(batch, kernel_exps)] # 1.0 / (1.0 - np.log(k + 0.000000001))


    def __p(self, z, dir):
        """Isotropic Gaussian"""
        p = self.K * np.exp(self.__p_exponent(dir.transpose().dot(z) * dir))
        return p

    def __p_exponent(self, z):
        """The exponent of the isotropic gaussian"""
        p_exp = -0.5 * z.transpose().dot(z)
        return p_exp

K = 1.0 / math.sqrt(2.0 * math.pi)

def f(batches, d, r, n, reg_eta):
    """Calculate the kernel density estimate for each reference, where each
    f(z) is calcualted on the entire data set (all zis in batches)"""
    refs = np.random.randn(d, r)
    fz = np.zeros((r, d))
    df_distances = []
    kernel_sums = []
    D = __rvs(d)
    pz = __pz(refs, D)
    #TESTING
    # refs = np.array([[4.0/41.0 ** 0.5, 1.0/2.0 ** 0.5, 0.8],[5.0/41.0 **0.5, 1.0/2.0 ** 0.5, 0.6]])
    # D = np.array([[2.0, -1.0],[1.0, 2.0]])
    # batches = [[np.array([[2.0], [4.0]]) / 20.0 ** 0.5, np.array([[4.0],[7.0]]) / 65.0 ** 0.5]]
    # D = D / 5.0 ** 0.5
    for batch in batches:
        #Compute the bandwith
        m = len(batch)
        h = (4.0 / (3.0 * n)) ** 0.2
        distance_along_dirs, distances_squared, kernels, kernel_sum = __compute_distance_kernels_and_kernelsums(batch, refs, D, h, n)
        # Store the kernel sum so that it may be removed when the current batch is updated
        kernel_sums.append(kernel_sum)
        # For each reference along each direction add to the overall kernel function f(z)
        # (as f(z) must be computed across the entire data set)
        fz += kernel_sum
        # Compute gradients for each zi in the batch
        df_distance = distance_along_dirs/h * 1.0 / (1.0 + distances_squared ** 0.5) #kernels
        #dfz = np.sum(np.matmul(D, move_along_dir), axis=0)
        # Hold these gradients for when the time comes to update the batch
        df_distances.append(df_distance)
    return DensityStats(fz, pz, df_distances, kernel_sums, refs, D, h, n, reg_eta)

def __compute_distance_kernels_and_kernelsums(zi_mat, refs, D, h, n):
    """Compute three key components of f(z) and dfz at once."""
    m = zi_mat.shape[1]
    d, r = refs.shape
    # Create a matrix out of all the zis in this batch
    #zi_mat = np.concatenate(batch, axis=1)
    # Expand the references to a tensor, so we may find the difference of each z and zi,
    # 1) For all dimensions d, of the latent space
    # 2) For all references r
    # 3) For all zi in the batch, m
    z_tensor = np.repeat(refs.transpose().reshape(r, d, 1), m, axis=2)
    # Relying on broadcasting, build a tensor of z - zi for each z,
    # and each zi in the batch
    z_less_zi = z_tensor - zi_mat.reshape(1, d, m)
    # Project each difference onto the d directions
    distance_along_dirs = np.matmul(D.transpose(), (z_less_zi)) / h
    # Use to calculate the distance between z and each zi
    distances_squared = distance_along_dirs ** 2.0
    #distances_squared = np.matmul(distance_along_dirs.transpose(axes=[0, 2, 1]), distance_along_dirs)
    # Put distances through the exponential
    kernels = K/h/float(n) * np.exp(-1.0 / 2.0 * distances_squared)
    # Compute the kernel sum across the batch, for each reference, and for each direction
    kernel_sum = np.sum(kernels, axis=2);
    return distance_along_dirs, distances_squared, kernels, kernel_sum

def update_fz(ds, new_batch, batch_index):
    """Update f(z) for each reference, in each direction, according to the
    newly updated batch at batch_index in batches """
    # Compute the kernels for the new batch
    _, _, _, new_ks = __compute_distance_kernels_and_kernelsums(new_batch, ds.refs, ds.D, ds.h, ds.n)
    #Retrieve the old kernel
    old_ks = ds.kernel_sums[batch_index]
    ds.fz += new_ks - old_ks
    ds.kernel_sums[batch_index] = new_ks
    return ds

def get_grad_for(batch_index, density_stats):
    """Get the full gradient f'(z)(f(z) - p(z)) with respect to the latent observations"""
    fz = density_stats.fz
    pz = density_stats.pz
    r, d = fz.shape
    D = density_stats.D
    df_distance = density_stats.df_distances[batch_index]
    dL_distance = np.sum(df_distance * (fz - pz).reshape(r, d, 1), axis=0)
    return D.dot(dL_distance)/float(r)

def __pz(refs, D):
    distances_squared = np.matmul(D.transpose(), refs) ** 2.0
    return K * np.exp(-1.0 / 2.0 * distances_squared).transpose()


def __rvs(dim):
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = np.random.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

# batches = [[np.random.randn(3, 1) for j in xrange(10)] for i in xrange(5)]
# f(batches, 3, 30)




